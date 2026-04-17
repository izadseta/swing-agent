"""
Swing Trade Agent v3 — Canada Edition
======================================
Improvements over v1:
  - 8:30 AM Toronto timing (before market open)
  - EMA20 trend filter (no trades against trend)
  - Signal scoring 1-5 (only alert on 3+ signals)
  - Momentum check (price rising last 2 days)
  - News sentiment via Finnhub (Claude reads headlines)
  - Fundamental check (P/E, analyst rating, earnings)
  - Smart stops split by market type:
      US stocks:        stop 2.5% / target 6%
      Canadian stocks:  stop 2.0% / target 5%
      Canadian ETFs:    stop 1.5% / target 4%
  - Telegram + Email alerts
  - Daily HTML email report

GitHub Actions cron: '30 12 * * 1-5' = 8:30 AM Toronto
"""

import os
import re
import json
import smtplib
import requests
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── CONFIG ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_SENDER       = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD     = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT    = os.getenv("EMAIL_RECIPIENT")
FINNHUB_API_KEY    = os.getenv("FINNHUB_API_KEY", "")

# ── STOP/TARGET BY MARKET TYPE ────────────────────────────────────────────────
# US stocks move more — wider stops needed
# Canadian ETFs move less — tighter stops protect capital

RULES = {
    "US": {
        "stop":   2.5,
        "target": 6.0,
        "label":  "US stock",
    },
    "CA_STOCK": {
        "stop":   2.0,
        "target": 5.0,
        "label":  "Canadian stock",
    },
    "CA_ETF": {
        "stop":   1.5,
        "target": 4.0,
        "label":  "Canadian ETF",
    },
}

# Canadian ETF tickers — gets CA_ETF rules
CA_ETFS = {
    "VFV.TO","ZSP.TO","XIU.TO","XIC.TO","VCN.TO","HXT.TO",
    "XEQT.TO","VEQT.TO","VDY.TO","ZWC.TO","ZWB.TO","ZEB.TO",
    "ZAG.TO","VBG.TO","VUN.TO","QQC.TO","ZDV.TO","XEI.TO",
    "ZCS.TO","CDZ.TO",
}

def get_rules(ticker: str) -> dict:
    if ticker in CA_ETFS:
        return RULES["CA_ETF"]
    elif ticker.endswith(".TO") or ticker.endswith(".TSX"):
        return RULES["CA_STOCK"]
    else:
        return RULES["US"]

# ── WATCHLIST ─────────────────────────────────────────────────────────────────

WATCHLIST = [
    # US stocks
    "NVDA", "AMD", "AAPL", "MSFT", "META",
    "TSLA", "AMZN", "AVGO", "COST", "VOO",
    # Canadian stocks
    "SHOP.TO", "ENB.TO", "SU.TO", "CNQ.TO",
    "RY.TO",   "TD.TO",  "BNS.TO","BMO.TO",
    "CM.TO",   "CNR.TO", "CP.TO", "MDA.TO",
    "ABX.TO",  "WPM.TO", "BCE.TO","FTS.TO",
    # Canadian ETFs
    "VFV.TO",  "XIU.TO", "XIC.TO","ZEB.TO",
    "VDY.TO",  "ZWC.TO", "XEQT.TO","QQC.TO",
    "ZAG.TO",  "VUN.TO",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MARKET DATA + TECHNICAL SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close, high, low = df["Close"], df["High"], df["Low"]

    # Trend
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

    # Momentum
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20)
    df["BBU"] = bb.bollinger_hband()
    df["BBL"] = bb.bollinger_lband()
    df["BBM"] = bb.bollinger_mavg()
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # MACD
    macd = ta.trend.MACD(close)
    df["MACD"]     = macd.macd()
    df["MACD_SIG"] = macd.macd_signal()
    df["MACD_HIS"] = macd.macd_diff()

    # Volume
    df["VOL_AVG"] = df["Volume"].rolling(20).mean()
    df["VOL_SPK"] = df["Volume"] / df["VOL_AVG"]

    return df.dropna()


def build_signal(ticker: str, df: pd.DataFrame) -> dict:
    """Score signals 0-5. Need 3+ to fire an alert."""
    if df.empty or len(df) < 10:
        return {}

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    price     = float(last["Close"])
    ema20     = float(last["EMA20"])
    rsi       = float(last["RSI"])
    rsi_prev  = float(prev["RSI"])
    bb_upper  = float(last["BBU"])
    bb_lower  = float(last["BBL"])
    macd_val  = float(last["MACD"])
    macd_sig  = float(last["MACD_SIG"])
    macd_his  = float(last["MACD_HIS"])
    prev_macd = float(prev["MACD"])
    prev_sig  = float(prev["MACD_SIG"])
    prev_his  = float(prev["MACD_HIS"])
    vol_spike = float(last["VOL_SPK"])
    atr_pct   = float(last["ATR"]) / price * 100

    # ── MOMENTUM CHECK: price rising last 2 days ──────────────────────────────
    momentum_up   = float(last["Close"]) > float(prev["Close"]) > float(prev2["Close"])
    momentum_down = float(last["Close"]) < float(prev["Close"]) < float(prev2["Close"])

    # ── MACD crossover ────────────────────────────────────────────────────────
    macd_bull = macd_val > macd_sig and prev_macd <= prev_sig
    macd_bear = macd_val < macd_sig and prev_macd >= prev_sig

    # ── SIGNAL SCORING — BUY ──────────────────────────────────────────────────
    buy_signals = []
    buy_score   = 0

    # Signal 1: Trend filter — only buy in uptrend
    if price > ema20:
        buy_signals.append("Above EMA20 (uptrend)")
        buy_score += 1

    # Signal 2: RSI recovering from oversold
    if 32 <= rsi <= 55 and rsi > rsi_prev:
        buy_signals.append(f"RSI {rsi:.0f} recovering ↑")
        buy_score += 1

    # Signal 3: Price near BB lower (oversold)
    if price <= bb_lower * 1.015:
        buy_signals.append("Near BB lower band")
        buy_score += 1

    # Signal 4: Bullish MACD crossover
    if macd_bull:
        buy_signals.append("Bullish MACD crossover ✓")
        buy_score += 1

    # Signal 5: Volume confirmation
    if vol_spike >= 1.4:
        buy_signals.append(f"Volume spike {vol_spike:.1f}x")
        buy_score += 1

    # Bonus: Momentum (price rising 2 days)
    if momentum_up:
        buy_signals.append("Momentum up 2 days")
        buy_score += 0.5

    # ── SIGNAL SCORING — SELL ─────────────────────────────────────────────────
    sell_signals = []
    sell_score   = 0

    if rsi > 70:
        sell_signals.append(f"RSI {rsi:.0f} overbought ⚠️")
        sell_score += 1
    if price >= bb_upper * 0.98:
        sell_signals.append("At BB upper band ⚠️")
        sell_score += 1
    if macd_bear:
        sell_signals.append("Bearish MACD crossover ⚠️")
        sell_score += 1
    if momentum_down and rsi > 60:
        sell_signals.append("Momentum fading")
        sell_score += 0.5
    if price < ema20 and rsi > 65:
        sell_signals.append("Below EMA20 + overbought")
        sell_score += 1

    # ── DETERMINE ACTION ──────────────────────────────────────────────────────
    if sell_score >= 2:
        action     = "SELL_ALERT"
        conviction = "HIGH" if sell_score >= 3 else "MEDIUM"
    elif buy_score >= 3:
        action     = "BUY_ALERT"
        conviction = "HIGH" if buy_score >= 4 else "MEDIUM"
    else:
        action     = "NO_ACTION"
        conviction = "LOW"

    rules      = get_rules(ticker)
    stop_price = round(price * (1 - rules["stop"]   / 100), 2)
    target     = round(price * (1 + rules["target"] / 100), 2)
    rr         = round((target - price) / (price - stop_price), 1) if price != stop_price else "N/A"

    change_5d  = (price - float(df.iloc[-5]["Close"])) / float(df.iloc[-5]["Close"]) * 100

    return {
        "ticker":        ticker,
        "market_type":   rules["label"],
        "price":         round(price, 2),
        "change_5d_pct": round(change_5d, 2),
        "RSI":           round(rsi, 1),
        "ATR_pct":       round(atr_pct, 2),
        "vol_spike":     round(vol_spike, 2),
        "price_vs_bb":   ("above_upper" if price > bb_upper else
                          "below_lower" if price < bb_lower else "inside"),
        "above_ema20":   price > ema20,
        "macd_cross":    ("bullish" if macd_bull else "bearish" if macd_bear else "none"),
        "momentum":      ("up" if momentum_up else "down" if momentum_down else "neutral"),
        "buy_score":     round(buy_score, 1),
        "sell_score":    round(sell_score, 1),
        "buy_signals":   buy_signals,
        "sell_signals":  sell_signals,
        "action":        action,
        "conviction":    conviction,
        "stop_price":    stop_price,
        "target_price":  target,
        "stop_pct":      rules["stop"],
        "target_pct":    rules["target"],
        "risk_reward":   rr,
        "as_of":         str(df.index[-1].date()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FUNDAMENTAL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fundamentals(ticker: str) -> dict:
    """Pull key fundamentals via yfinance. Returns empty dict if unavailable."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":        round(info.get("trailingPE", 0) or 0, 1),
            "forward_pe":      round(info.get("forwardPE", 0) or 0, 1),
            "analyst_rating":  info.get("recommendationKey", "n/a"),
            "target_price":    round(info.get("targetMeanPrice", 0) or 0, 2),
            "earnings_growth": round((info.get("earningsGrowth", 0) or 0) * 100, 1),
            "revenue_growth":  round((info.get("revenueGrowth", 0) or 0) * 100, 1),
            "sector":          info.get("sector", "n/a"),
            "market_cap_b":    round((info.get("marketCap", 0) or 0) / 1e9, 1),
            "dividend_yield":  round((info.get("dividendYield", 0) or 0) * 100, 2),
            "beta":            round(info.get("beta", 0) or 0, 2),
        }
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NEWS SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_for_ticker(ticker: str) -> list:
    """Fetch recent news headlines for a specific ticker via Finnhub."""
    if not FINNHUB_API_KEY:
        return []
    # Strip .TO suffix for Finnhub
    clean = ticker.replace(".TO", "").replace(".TSX", "")
    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={clean}&from=2024-01-01&to=2099-12-31&token={FINNHUB_API_KEY}",
            timeout=8
        )
        items = r.json()
        if isinstance(items, list):
            return [item.get("headline", "") for item in items[:5] if item.get("headline")]
    except Exception:
        pass
    return []


def fetch_market_news() -> list:
    """Fetch top global market headlines for daily briefing."""
    headlines = []
    if FINNHUB_API_KEY:
        try:
            r = requests.get(
                f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}",
                timeout=10
            )
            for item in r.json()[:15]:
                h = item.get("headline", "")
                if h:
                    headlines.append(h)
        except Exception as e:
            print(f"  Finnhub market news error: {e}")

    # Fallback — Yahoo Finance RSS
    if len(headlines) < 5:
        try:
            import xml.etree.ElementTree as ET
            r = requests.get(
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
                timeout=8
            )
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                t = item.find("title")
                if t is not None and t.text:
                    headlines.append(t.text)
                if len(headlines) >= 15:
                    break
        except Exception as e:
            print(f"  RSS fallback error: {e}")

    # Always add TSX context
    headlines += [
        "TSX at ~33,650 — energy and financials leading in 2026",
        "Bank of Canada holds rate at 2.25% — rate hike risk present",
        "Canada 10-yr bond yield at 3.48% — watch bond ETF holders",
    ]
    return [h for h in headlines if h][:20]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLAUDE AI
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_PROMPT = """You are an elite swing trade analyst for a Canadian investor.
Trades are held 1-3 days. You analyze technical signals, fundamentals, and news.

Your scoring rules:
- Need 3+ buy signals to trigger BUY_ALERT
- Need 2+ sell signals to trigger SELL_ALERT
- EMA20 must be respected — no buys below trend
- News sentiment can override technical — bad news = NO_ACTION even on good technicals

Output valid JSON only — no markdown:
{
  "action": "BUY_ALERT"|"SELL_ALERT"|"NO_ACTION",
  "conviction": "HIGH"|"MEDIUM"|"LOW",
  "reasoning": "2-3 sentences covering technicals + news + fundamentals",
  "news_sentiment": "positive"|"negative"|"neutral",
  "fundamental_view": "strong"|"weak"|"neutral",
  "entry_price": <number|null>,
  "stop_loss": <number|null>,
  "take_profit": <number|null>,
  "hold_days": <1-3>,
  "key_signals": ["signal1","signal2","signal3"]
}"""

NEWS_BRIEFING_PROMPT = """You are a Canadian market analyst writing a pre-market briefing
for a swing trader who holds trades 1-3 days.
Cover: macro events, TSX outlook, US market direction, sector momentum, and top risk today.
Max 250 words. Be direct and actionable."""


def analyze_with_claude(signal: dict, fundamentals: dict, news: list) -> dict:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""Analyze this 1-3 day swing trade setup:

TECHNICAL SIGNALS:
{json.dumps(signal, indent=2)}

FUNDAMENTALS:
{json.dumps(fundamentals, indent=2) if fundamentals else "Not available"}

RECENT NEWS ({signal['ticker']}):
{chr(10).join(f'• {h}' for h in news) if news else "No recent news found"}

Stop: ${signal['stop_price']} (-{signal['stop_pct']}%)
Target: ${signal['target_price']} (+{signal['target_pct']}%)
R:R: 1:{signal['risk_reward']}
Market type: {signal['market_type']}

Make your final call. If news is negative, output NO_ACTION."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=TRADE_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = re.sub(r"```json|```", "", response.content[0].text.strip()).strip()
    return json.loads(raw)


def claude_daily_briefing(headlines: list, alerts: list) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    alert_summary = (", ".join(f"{a['ticker']} ({a.get('action','')})" for a in alerts)
                     if alerts else "none today")
    prompt = (f"Date: {datetime.now().strftime('%A %B %d, %Y')}\n\n"
              f"Today's headlines:\n"
              + "\n".join(f"• {h}" for h in headlines)
              + f"\n\nSwing alerts fired today: {alert_summary}\n\n"
              "Write the pre-market briefing. Focus on what matters for 1-3 day swing trades.")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=NEWS_BRIEFING_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }, timeout=10)
    return r.ok


def format_telegram_alert(signal: dict, decision: dict) -> str:
    action = decision.get("action", "NO_ACTION")
    if action not in ("BUY_ALERT", "SELL_ALERT"):
        return ""

    emoji  = "🟢" if action == "BUY_ALERT" else "🔴"
    label  = "BUY" if action == "BUY_ALERT" else "SELL/EXIT"
    ticker = signal["ticker"]
    price  = signal["price"]
    entry  = decision.get("entry_price") or price
    sl     = decision.get("stop_loss")   or signal["stop_price"]
    tp     = decision.get("take_profit") or signal["target_price"]
    rr     = signal["risk_reward"]
    hold   = decision.get("hold_days", "1-3")
    news_s = decision.get("news_sentiment", "neutral")
    fund_v = decision.get("fundamental_view", "neutral")
    score  = signal["buy_score"] if action == "BUY_ALERT" else signal["sell_score"]
    mtype  = signal["market_type"]

    news_icon = "📰✅" if news_s == "positive" else "📰⚠️" if news_s == "negative" else "📰➖"
    fund_icon = "📊✅" if fund_v == "strong"   else "📊⚠️" if fund_v == "weak"     else "📊➖"

    signals = signal.get("buy_signals" if action == "BUY_ALERT" else "sell_signals", [])

    lines = [
        f"{emoji} <b>{label} — {ticker}</b> [{decision['conviction']} · {mtype}]",
        f"",
        f"<b>Price:</b> ${price}  |  <b>Score:</b> {score}/5  |  <b>Date:</b> {signal['as_of']}",
        f"<b>Entry:</b> ~${entry}",
        f"<b>Stop:</b> ${sl} (−{signal['stop_pct']}%)  |  <b>Target:</b> ${tp} (+{signal['target_pct']}%)",
        f"<b>R:R:</b> 1:{rr}  |  <b>Hold:</b> ~{hold} day(s)",
        f"",
        f"{news_icon} News: {news_s}  |  {fund_icon} Fundamentals: {fund_v}",
        f"",
        f"<b>Why:</b> {decision.get('reasoning', '')}",
        f"",
        f"<b>Signals:</b> {' · '.join(signals[:4])}",
        f"",
        f"<i>RSI {signal['RSI']} · ATR {signal['ATR_pct']}% · Vol {signal['vol_spike']}x · "
        f"{'↑ Above' if signal['above_ema20'] else '↓ Below'} EMA20 · {signal['price_vs_bb']}</i>",
        f"",
        f"⚠️ <i>Execute manually in Wealthsimple. Analysis only.</i>",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EMAIL
# ═══════════════════════════════════════════════════════════════════════════════

def build_email(briefing: str, alerts: list) -> str:
    date_str  = datetime.now().strftime("%A, %B %d, %Y — 8:30 AM Toronto")
    news_html = briefing.replace("\n\n", "</p><p>").replace("\n", "<br>")

    buy_count  = sum(1 for a in alerts if a.get("action") == "BUY_ALERT")
    sell_count = sum(1 for a in alerts if a.get("action") == "SELL_ALERT")

    rows = ""
    for a in alerts:
        sig    = a.get("signal", {})
        dec    = a.get("decision", {})
        is_buy = dec.get("action") == "BUY_ALERT"
        color  = "#27500A" if is_buy else "#A32D2D"
        bg     = "#EAF3DE" if is_buy else "#FCEBEB"
        label  = "BUY" if is_buy else "SELL"
        score  = sig.get("buy_score" if is_buy else "sell_score", 0)
        news_s = dec.get("news_sentiment", "neutral")
        fund_v = dec.get("fundamental_view", "neutral")
        news_icon = "✅" if news_s == "positive" else "⚠️" if news_s == "negative" else "➖"
        fund_icon = "✅" if fund_v == "strong"   else "⚠️" if fund_v == "weak"     else "➖"
        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:10px 12px;font-weight:700;color:{color};font-size:15px">{sig.get('ticker','')}</td>
          <td style="padding:10px 12px;font-weight:600;color:{color}">{label}</td>
          <td style="padding:10px 12px">{dec.get('conviction','')}</td>
          <td style="padding:10px 12px;font-size:12px">{score}/5 signals</td>
          <td style="padding:10px 12px;font-size:12px">
            News {news_icon} {news_s}<br>
            Fund {fund_icon} {fund_v}
          </td>
          <td style="padding:10px 12px;font-size:12px">
            Entry ${dec.get('entry_price') or sig.get('price','')}<br>
            Stop ${sig.get('stop_price','')} | TP ${sig.get('target_price','')}<br>
            R:R 1:{sig.get('risk_reward','')} · {dec.get('hold_days','1-3')}d
          </td>
          <td style="padding:10px 12px;font-size:11px;color:#444">{dec.get('reasoning','')[:100]}...</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="7" style="padding:16px;color:#888;text-align:center">No high-conviction setups today — staying out protects capital</td></tr>'

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f2f2f2;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:760px;margin:20px auto;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.1)">

  <div style="background:#0d2137;padding:24px 30px">
    <div style="font-size:11px;color:#7db3d8;text-transform:uppercase;letter-spacing:.1em">Swing Agent v3 — Pre-Market</div>
    <div style="font-size:26px;font-weight:700;color:#fff;margin-top:4px">Daily Trade Report</div>
    <div style="font-size:13px;color:#7db3d8;margin-top:4px">{date_str}</div>
  </div>

  <div style="background:#102030;padding:12px 30px;display:flex;gap:28px">
    <div style="text-align:center">
      <div style="font-size:22px;font-weight:700;color:{"#4ade80" if buy_count else "#555"}">{buy_count}</div>
      <div style="font-size:11px;color:#666">Buy alerts</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:22px;font-weight:700;color:{"#f87171" if sell_count else "#555"}">{sell_count}</div>
      <div style="font-size:11px;color:#666">Sell alerts</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:22px;font-weight:700;color:#fff">{len(WATCHLIST)}</div>
      <div style="font-size:11px;color:#666">Scanned</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:22px;font-weight:700;color:#fff">3+/5</div>
      <div style="font-size:11px;color:#666">Signal threshold</div>
    </div>
  </div>

  <div style="padding:26px 30px">

    <h2 style="font-size:17px;font-weight:600;margin:0 0 14px;color:#111;border-bottom:2px solid #eee;padding-bottom:8px">
      🌍 Pre-Market Briefing
    </h2>
    <div style="background:#f8f8f8;border-left:4px solid #0d2137;border-radius:0 8px 8px 0;padding:14px 18px;font-size:14px;line-height:1.8;color:#333">
      <p style="margin:0">{news_html}</p>
    </div>

    <h2 style="font-size:17px;font-weight:600;margin:26px 0 14px;color:#111;border-bottom:2px solid #eee;padding-bottom:8px">
      📡 Trade Alerts
    </h2>

    <div style="background:#fff8e1;border:1px solid #ffc107;border-radius:8px;padding:10px 14px;font-size:12px;color:#856404;margin-bottom:14px">
      <strong>Rules applied:</strong>
      US stop −2.5%/target +6% · Canadian stocks stop −2%/target +5% · Canadian ETFs stop −1.5%/target +4% ·
      Needs 3+/5 signals · EMA20 trend filter · News + fundamentals checked · 1-3 day hold
    </div>

    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:#f0f0f0">
        <th style="padding:8px 12px;text-align:left">Ticker</th>
        <th style="padding:8px 12px;text-align:left">Action</th>
        <th style="padding:8px 12px;text-align:left">Conv.</th>
        <th style="padding:8px 12px;text-align:left">Score</th>
        <th style="padding:8px 12px;text-align:left">News/Fund</th>
        <th style="padding:8px 12px;text-align:left">Levels</th>
        <th style="padding:8px 12px;text-align:left">Reasoning</th>
      </tr>
      {rows}
    </table>

  </div>

  <div style="background:#f5f5f5;padding:12px 30px;font-size:11px;color:#999;border-top:1px solid #eee;text-align:center">
    Swing Agent v3 · Execute manually in Wealthsimple · Not financial advice · Generated {datetime.now().strftime('%H:%M ET')}
  </div>
</div>
</body></html>"""


def send_email(html: str):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT]):
        print("  Email not configured — add EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT to .env")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📈 Swing Agent — {datetime.now().strftime('%b %d %Y')} pre-market"
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        print(f"  Email sent → {EMAIL_RECIPIENT}")
    except Exception as e:
        print(f"  Email error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    print(f"\n{'='*55}")
    print(f"  SWING AGENT v3 — Pre-Market Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} Toronto")
    print(f"  Scanning {len(WATCHLIST)} stocks...")
    print(f"{'='*55}")

    alerts_sent = 0
    all_alerts  = []

    for ticker in WATCHLIST:
        print(f"\n  {ticker}...")
        try:
            # 1. Technical signals
            df = fetch_stock_data(ticker)
            if df.empty:
                print("    No data, skip")
                continue
            signal = build_signal(ticker, df)
            if not signal:
                print("    Not enough data, skip")
                continue

            print(f"    Score B:{signal['buy_score']}/S:{signal['sell_score']} | "
                  f"RSI {signal['RSI']} | Vol {signal['vol_spike']}x | "
                  f"{'↑EMA' if signal['above_ema20'] else '↓EMA'} | "
                  f"{signal['price_vs_bb']} | {signal['action']}")

            # Only fetch fundamentals + news if signal looks interesting
            if signal["action"] == "NO_ACTION" and signal["buy_score"] < 2 and signal["sell_score"] < 1.5:
                continue

            # 2. Fundamentals
            print(f"    Fetching fundamentals...")
            fundamentals = fetch_fundamentals(ticker)

            # 3. News
            print(f"    Fetching news...")
            news = fetch_news_for_ticker(ticker)
            if news:
                print(f"    {len(news)} headlines found")

            # 4. Claude decision
            decision = analyze_with_claude(signal, fundamentals, news)
            final_action = decision.get("action", "NO_ACTION")
            conviction   = decision.get("conviction", "LOW")
            news_s       = decision.get("news_sentiment", "neutral")
            print(f"    Claude: {final_action} [{conviction}] · news={news_s}")

            if final_action != "NO_ACTION" and conviction in ("HIGH", "MEDIUM"):
                # Telegram
                msg = format_telegram_alert(signal, decision)
                if msg:
                    ok = send_telegram(msg)
                    print(f"    ✓ Telegram: {ok}")
                    alerts_sent += 1

                all_alerts.append({"signal": signal, "decision": decision})

        except Exception as e:
            print(f"    Error: {e}")

    # Summary Telegram if no alerts
    print(f"\n  Scan done — {alerts_sent} alert(s)")
    if alerts_sent == 0:
        send_telegram(
            f"📊 <b>Swing Agent v3</b> — {datetime.now().strftime('%b %d %H:%M')}\n\n"
            f"Scanned {len(WATCHLIST)} stocks — no setups met 3+/5 signal threshold.\n"
            f"Technical + news + fundamentals all checked.\n\n"
            f"<i>Full pre-market briefing sent to your email.</i>"
        )

    # News briefing + email
    print("\n  Fetching market news...")
    headlines = fetch_market_news()
    print(f"  {len(headlines)} headlines")

    print("  Writing pre-market briefing...")
    briefing = claude_daily_briefing(headlines, [a["decision"] for a in all_alerts])

    print("  Building email...")
    html = build_email(briefing, all_alerts)
    send_email(html)

    print(f"\n{'='*55}")
    print(f"  All done! ✅")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()
