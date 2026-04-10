"""
Swing Trade Agent — Canada Edition
Analyzes US & TSX stocks, uses Claude AI to reason about trades,
and sends Telegram alerts so you manually execute.
"""

import os
import json
from dotenv import load_dotenv
load_dotenv('/Users/mohammadizadseta/Desktop/swing-agent/.env')
import asyncio
import requests
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime
from anthropic import Anthropic

# ── CONFIG ─────────────────────────────────────────────────────────────────
# Copy .env.example to .env and fill in your keys

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-claude-api-key")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your-telegram-bot-token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your-chat-id")

# Stocks to watch — add/remove tickers freely
# TSX stocks: append .TO (e.g. SHOP.TO, RY.TO, ENB.TO)
# US stocks: plain ticker (e.g. NVDA, AAPL, AMD)
WATCHLIST = [
    "NVDA", "AMD", "AAPL", "MSFT",   # US tech
    "SHOP.TO", "RY.TO", "CNQ.TO", "ENB.TO", "VFV.TO", "COST.TO", "XEI.TO"    # Canadian
]

# Risk settings
MAX_POSITION_PCT = 10     # Never suggest more than 10% of portfolio per trade
STOP_LOSS_PCT = 5         # Suggest stop-loss 5% below entry
TAKE_PROFIT_PCT = 12      # Suggest take-profit 12% above entry

# ── MARKET DATA ─────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Pull OHLCV data + technical indicators via yfinance."""
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Technical indicators
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BBU_20"] = bb.bollinger_hband()
    df["BBL_20"] = bb.bollinger_lband()
    df["BBM_20"] = bb.bollinger_mavg()
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    macd = ta.trend.MACD(df["Close"])
    df["MACD_12_26_9"] = macd.macd()
    df["MACDs_12_26_9"] = macd.macd_signal()

    # Volume spike (vs 20-day avg)
    df["Vol_Avg20"] = df["Volume"].rolling(20).mean()
    df["Vol_Spike"] = df["Volume"] / df["Vol_Avg20"]

    return df.dropna()


def build_signal_summary(ticker: str, df: pd.DataFrame) -> dict:
    """Extract the last few days of signals into a clean dict for Claude."""
    if df.empty or len(df) < 5:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Find Bollinger Band column names dynamically
    bb_upper = next((c for c in df.columns if "BBU" in c), None)
    bb_lower = next((c for c in df.columns if "BBL" in c), None)
    bb_mid   = next((c for c in df.columns if "BBM" in c), None)
    macd_col = next((c for c in df.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()), None)
    macd_sig = next((c for c in df.columns if "MACDs" in c), None)

    price = float(last["Close"])
    price_change_5d = (price - float(df.iloc[-5]["Close"])) / float(df.iloc[-5]["Close"]) * 100

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "price_change_5d_pct": round(price_change_5d, 2),
        "RSI": round(float(last["RSI"]), 1),
        "ATR": round(float(last["ATR"]), 2),
        "ATR_pct_of_price": round(float(last["ATR"]) / price * 100, 2),
        "volume_spike_ratio": round(float(last["Vol_Spike"]), 2),
        "bb_upper": round(float(last[bb_upper]), 2) if bb_upper else None,
        "bb_lower": round(float(last[bb_lower]), 2) if bb_lower else None,
        "bb_mid":   round(float(last[bb_mid]), 2)   if bb_mid   else None,
        "price_vs_bb": (
            "above_upper" if bb_upper and price > float(last[bb_upper]) else
            "below_lower" if bb_lower and price < float(last[bb_lower]) else
            "inside_bands"
        ),
        "macd": round(float(last[macd_col]), 4) if macd_col else None,
        "macd_signal": round(float(last[macd_sig]), 4) if macd_sig else None,
        "macd_crossover": (
            "bullish" if macd_col and macd_sig and
            float(last[macd_col]) > float(last[macd_sig]) and
            float(prev[macd_col]) <= float(prev[macd_sig])
            else
            "bearish" if macd_col and macd_sig and
            float(last[macd_col]) < float(last[macd_sig]) and
            float(prev[macd_col]) >= float(prev[macd_sig])
            else "none"
        ),
        "as_of": str(df.index[-1].date()),
    }


# ── CLAUDE AI ANALYSIS ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a disciplined swing trade analyst. You analyze technical signals for stocks and decide whether to alert the human trader.

Your job:
1. Evaluate the technical signals provided
2. Decide: BUY_ALERT, SELL_ALERT, or NO_ACTION
3. Only alert on high-conviction setups — avoid noise

Swing trade criteria to look for:
- BUY signals: RSI 30-50 recovering, price near or bouncing off BB lower, bullish MACD crossover, volume spike confirming move, ATR showing reasonable volatility
- SELL/EXIT signals: RSI >70 (overbought), price touching BB upper, bearish MACD crossover, extended run without pullback
- NO_ACTION: mixed signals, no clear setup, RSI in neutral zone with no crossover

Always output valid JSON only — no markdown, no extra text:
{
  "action": "BUY_ALERT" | "SELL_ALERT" | "NO_ACTION",
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "2-3 sentence plain-English explanation",
  "entry_price": <number or null>,
  "stop_loss": <number or null>,
  "take_profit": <number or null>,
  "key_signals": ["signal1", "signal2"]
}"""


def analyze_with_claude(signal: dict) -> dict:
    """Send signals to Claude and get a structured trade decision."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Analyze this swing trade setup and return your JSON decision:

{json.dumps(signal, indent=2)}

Stop-loss guideline: ~{STOP_LOSS_PCT}% below entry
Take-profit guideline: ~{TAKE_PROFIT_PCT}% above entry
Only flag HIGH conviction if 3+ signals align."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    return json.loads(raw)


# ── TELEGRAM ALERTS ──────────────────────────────────────────────────────────

def send_telegram(message: str):
    """Send a formatted message to your Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    r = requests.post(url, json=payload, timeout=10)
    return r.ok


def format_alert(signal: dict, decision: dict) -> str:
    """Format a clean Telegram message."""
    ticker = signal["ticker"]
    action = decision["action"]
    conviction = decision["conviction"]
    price = signal["price"]

    if action == "BUY_ALERT":
        emoji = "🟢"
        action_text = "BUY ALERT"
    elif action == "SELL_ALERT":
        emoji = "🔴"
        action_text = "SELL / EXIT ALERT"
    else:
        return ""

    entry = decision.get("entry_price") or price
    sl = decision.get("stop_loss") or round(price * (1 - STOP_LOSS_PCT/100), 2)
    tp = decision.get("take_profit") or round(price * (1 + TAKE_PROFIT_PCT/100), 2)
    risk_reward = round((tp - entry) / (entry - sl), 1) if entry != sl else "N/A"

    lines = [
        f"{emoji} <b>{action_text} — {ticker}</b> [{conviction} conviction]",
        f"",
        f"<b>Price:</b> ${price}  |  <b>As of:</b> {signal['as_of']}",
        f"<b>Entry:</b> ~${entry}",
        f"<b>Stop-loss:</b> ${sl} ({STOP_LOSS_PCT}% risk)",
        f"<b>Take-profit:</b> ${tp} ({TAKE_PROFIT_PCT}% target)",
        f"<b>Risk/Reward:</b> 1:{risk_reward}",
        f"",
        f"<b>Why:</b> {decision['reasoning']}",
        f"",
        f"<b>Signals:</b> {' · '.join(decision.get('key_signals', []))}",
        f"",
        f"<i>RSI {signal['RSI']} · ATR {signal['ATR']} ({signal['ATR_pct_of_price']}%) · Vol spike {signal['volume_spike_ratio']}x · {signal['price_vs_bb']}</i>",
        f"",
        f"⚠️ <i>This is analysis only — execute manually in Wealthsimple.</i>",
        f"<i>Max position: {MAX_POSITION_PCT}% of portfolio.</i>",
    ]

    return "\n".join(lines)


# ── MAIN LOOP ────────────────────────────────────────────────────────────────

def run_scan():
    """Scan all watchlist tickers and send alerts for actionable setups."""
    print(f"\n{'='*50}")
    print(f"Swing Agent scan — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    alerts_sent = 0

    for ticker in WATCHLIST:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_stock_data(ticker)
            if df.empty:
                print(f"  No data for {ticker}, skipping.")
                continue

            signal = build_signal_summary(ticker, df)
            if not signal:
                print(f"  Not enough data for {ticker}, skipping.")
                continue

            print(f"  RSI={signal['RSI']} | ATR%={signal['ATR_pct_of_price']} | Vol spike={signal['volume_spike_ratio']}x | {signal['price_vs_bb']}")

            decision = analyze_with_claude(signal)
            action = decision.get("action", "NO_ACTION")
            conviction = decision.get("conviction", "LOW")

            print(f"  Claude says: {action} [{conviction}]")

            # Only alert on medium/high conviction
            if action != "NO_ACTION" and conviction in ("HIGH", "MEDIUM"):
                message = format_alert(signal, decision)
                if message:
                    ok = send_telegram(message)
                    print(f"  Telegram alert sent: {ok}")
                    alerts_sent += 1

        except Exception as e:
            print(f"  Error on {ticker}: {e}")

    print(f"\nScan complete. {alerts_sent} alert(s) sent.")

    if alerts_sent == 0:
        send_telegram(
            f"📊 <b>Swing Agent scan complete</b> — {datetime.now().strftime('%b %d %H:%M')}\n"
            f"No high-conviction setups found across {len(WATCHLIST)} stocks. Watching..."
        )


if __name__ == "__main__":
    run_scan()
