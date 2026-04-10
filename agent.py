"""
Swing Trade Agent — Canada Edition
Analyzes US & TSX stocks, uses Claude AI to reason about trades,
and sends Telegram alerts so you manually execute.
"""

import os
import json
import requests
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── CONFIG ──────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WATCHLIST = [
    # US stocks
    "NVDA",   # NVIDIA — AI/chips
    "AMD",    # AMD — follows NVDA
    "AAPL",   # Apple — stable signals
    "MSFT",   # Microsoft — steady trend
    "META",   # Meta — strong momentum
    "TSLA",   # Tesla — high volatility
    "AMZN",   # Amazon — broad indicator
    "AVGO",   # Broadcom — AI/chips
    "COST",   # Costco — defensive growth
    "VOO",    # Vanguard S&P 500 ETF (US)
    # Canadian stocks — execute manually in Wealthsimple
    "SHOP.TO",  # Shopify — tech
    "ENB.TO",   # Enbridge — pipeline
    "SU.TO",    # Suncor — oil
    "CNQ.TO",   # Canadian Natural Resources
    "RY.TO",    # Royal Bank
    "TD.TO",    # TD Bank
    "BNS.TO",   # Scotiabank
    "BMO.TO",   # Bank of Montreal
    "CM.TO",    # CIBC
    "CNR.TO",   # CN Rail
    "CP.TO",    # CP Rail
    "MDA.TO",   # MDA Space — tech
    "ABX.TO",   # Barrick Gold
    "WPM.TO",   # Wheaton Precious Metals
    "BCE.TO",   # BCE Telecom
    # Canadian ETFs
    "VFV.TO",   # Vanguard S&P 500 CAD
    "ZSP.TO",   # BMO S&P 500 CAD
    "XIU.TO",   # iShares TSX 60
    "XIC.TO",   # iShares TSX Composite
    "VCN.TO",   # Vanguard Canada All Cap
    "HXT.TO",   # Horizons TSX 60
    "XEQT.TO",  # iShares All-Equity Portfolio
    "VEQT.TO",  # Vanguard All-Equity Portfolio
    "VDY.TO",   # Vanguard Canadian High Dividend
    "ZWC.TO",   # BMO High Dividend Covered Call
    "ZWB.TO",   # BMO Canadian Banks Covered Call
    "ZAG.TO",   # BMO Aggregate Bond
    "VBG.TO",   # Vanguard Global Bond CAD-hedged
    "VUN.TO",   # Vanguard US Total Market CAD
    "QQC.TO",   # Invesco NASDAQ 100 CAD
]

MAX_POSITION_PCT = 10
STOP_LOSS_PCT = 5
TAKE_PROFIT_PCT = 12

# ── MARKET DATA ──────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    bb = ta.volatility.BollingerBands(close, window=20)
    df["BBU_20"] = bb.bollinger_hband()
    df["BBL_20"] = bb.bollinger_lband()
    df["BBM_20"] = bb.bollinger_mavg()

    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    macd = ta.trend.MACD(close)
    df["MACD_12_26_9"] = macd.macd()
    df["MACDs_12_26_9"] = macd.macd_signal()

    df["Vol_Avg20"] = df["Volume"].rolling(20).mean()
    df["Vol_Spike"] = df["Volume"] / df["Vol_Avg20"]

    return df.dropna()


def build_signal_summary(ticker: str, df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 5:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last["Close"])
    price_change_5d = (price - float(df.iloc[-5]["Close"])) / float(df.iloc[-5]["Close"]) * 100

    bb_upper = float(last["BBU_20"])
    bb_lower = float(last["BBL_20"])
    bb_mid   = float(last["BBM_20"])
    macd_val = float(last["MACD_12_26_9"])
    macd_sig = float(last["MACDs_12_26_9"])
    prev_macd = float(prev["MACD_12_26_9"])
    prev_sig  = float(prev["MACDs_12_26_9"])

    if macd_val > macd_sig and prev_macd <= prev_sig:
        crossover = "bullish"
    elif macd_val < macd_sig and prev_macd >= prev_sig:
        crossover = "bearish"
    else:
        crossover = "none"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "price_change_5d_pct": round(price_change_5d, 2),
        "RSI": round(float(last["RSI"]), 1),
        "ATR": round(float(last["ATR"]), 2),
        "ATR_pct_of_price": round(float(last["ATR"]) / price * 100, 2),
        "volume_spike_ratio": round(float(last["Vol_Spike"]), 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_mid":   round(bb_mid, 2),
        "price_vs_bb": (
            "above_upper" if price > bb_upper else
            "below_lower" if price < bb_lower else
            "inside_bands"
        ),
        "macd": round(macd_val, 4),
        "macd_signal": round(macd_sig, 4),
        "macd_crossover": crossover,
        "as_of": str(df.index[-1].date()),
    }


# ── CLAUDE AI ANALYSIS ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a disciplined swing trade analyst. You analyze technical signals for stocks and decide whether to alert the human trader.

Your job:
1. Evaluate the technical signals provided
2. Decide: BUY_ALERT, SELL_ALERT, or NO_ACTION
3. Only alert on high-conviction setups — avoid noise

Swing trade criteria to look for:
- BUY signals: RSI 30-50 recovering, price near or bouncing off BB lower, bullish MACD crossover, volume spike confirming move
- SELL/EXIT signals: RSI >70 (overbought), price touching BB upper, bearish MACD crossover
- NO_ACTION: mixed signals, no clear setup

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
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }, timeout=10)
    return r.ok


def format_alert(signal: dict, decision: dict) -> str:
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
    sl = decision.get("stop_loss") or round(price * (1 - STOP_LOSS_PCT / 100), 2)
    tp = decision.get("take_profit") or round(price * (1 + TAKE_PROFIT_PCT / 100), 2)
    rr = round((tp - entry) / (entry - sl), 1) if entry != sl else "N/A"

    lines = [
        f"{emoji} <b>{action_text} — {ticker}</b> [{conviction} conviction]",
        f"",
        f"<b>Price:</b> ${price}  |  <b>As of:</b> {signal['as_of']}",
        f"<b>Entry:</b> ~${entry}",
        f"<b>Stop-loss:</b> ${sl} ({STOP_LOSS_PCT}% risk)",
        f"<b>Take-profit:</b> ${tp} ({TAKE_PROFIT_PCT}% target)",
        f"<b>Risk/Reward:</b> 1:{rr}",
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


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_scan():
    print(f"\n{'='*50}")
    print(f"Swing Agent scan — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    alerts_sent = 0

    for ticker in WATCHLIST:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_stock_data(ticker)
            if df.empty:
                print(f"  No data, skipping.")
                continue

            signal = build_signal_summary(ticker, df)
            if not signal:
                print(f"  Not enough data, skipping.")
                continue

            print(f"  RSI={signal['RSI']} | ATR%={signal['ATR_pct_of_price']} | Vol spike={signal['volume_spike_ratio']}x | {signal['price_vs_bb']}")

            decision = analyze_with_claude(signal)
            action = decision.get("action", "NO_ACTION")
            conviction = decision.get("conviction", "LOW")
            print(f"  Claude says: {action} [{conviction}]")

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
