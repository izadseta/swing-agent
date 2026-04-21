"""
Swing Trade Agent v3 — Canada Edition
Fixed: IndentationError, KeyError ticker, model updated
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

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_SENDER       = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD     = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT    = os.getenv("EMAIL_RECIPIENT")
FINNHUB_API_KEY    = os.getenv("FINNHUB_API_KEY", "")

CA_ETFS = {
    "VFV.TO","ZSP.TO","XIU.TO","XIC.TO","VCN.TO","HXT.TO",
    "XEQT.TO","VEQT.TO","VDY.TO","ZWC.TO","ZWB.TO","ZEB.TO",
    "ZAG.TO","VBG.TO","VUN.TO","QQC.TO","ZDV.TO","XEI.TO",
}

def get_rules(ticker):
    if ticker in CA_ETFS:
        return {"stop": 1.5, "target": 4.0, "label": "Canadian ETF"}
    elif ticker.endswith(".TO"):
        return {"stop": 2.0, "target": 5.0, "label": "Canadian Stock"}
    else:
        return {"stop": 2.5, "target": 6.0, "label": "US Stock"}

WATCHLIST = [
    "NVDA","AMD","AAPL","MSFT","META","TSLA","AMZN","AVGO","COST","SPY",
    "SHOP.TO","ENB.TO","SU.TO","CNQ.TO","RY.TO","TD.TO","BNS.TO",
    "BMO.TO","CM.TO","CNR.TO","CP.TO","MDA.TO","ABX.TO","WPM.TO",
    "BCE.TO","FTS.TO","XIU.TO","ZEB.TO","VFV.TO","QQC.TO","XEQT.TO",
    "VDY.TO","ZWC.TO","XIC.TO","VCN.TO",
]

def fetch_stock_data(ticker, period="3mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c, h, l = df["Close"], df["High"], df["Low"]
        df["EMA20"] = ta.trend.EMAIndicator(c, window=20).ema_indicator()
        df["RSI"]   = ta.momentum.RSIIndicator(c, window=14).rsi()
        bb = ta.volatility.BollingerBands(c, window=20)
        df["BBU"] = bb.bollinger_hband()
        df["BBL"] = bb.bollinger_lband()
        df["BBM"] = bb.bollinger_mavg()
        df["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        macd = ta.trend.MACD(c)
        df["MACD"]     = macd.macd()
        df["MACD_SIG"] = macd.macd_signal()
        df["VOL_AVG"]  = df["Volume"].rolling(20).mean()
        df["VOL_SPK"]  = df["Volume"] / df["VOL_AVG"]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def build_signal(ticker, df):
    if df.empty or len(df) < 10:
        return {}
    try:
        last, prev = df.iloc[-1], df.iloc[-2]
        price    = float(last["Close"])
        rsi      = float(last["RSI"])
        ema20    = float(last["EMA20"])
        bb_upper = float(last["BBU"])
        bb_lower = float(last["BBL"])
        macd_val = float(last["MACD"])
        macd_sig = float(last["MACD_SIG"])
        vol_spk  = float(last["VOL_SPK"])
        atr_pct  = float(last["ATR"]) / price * 100
        macd_bull = macd_val > macd_sig and float(prev["MACD"]) <= float(prev["MACD_SIG"])
        macd_bear = macd_val < macd_sig and float(prev["MACD"]) >= float(prev["MACD_SIG"])

        buy_score, buy_signals = 0, []
        if price > ema20:
            buy_score += 1
            buy_signals.append("Above EMA20 uptrend")
        if 32 <= rsi <= 55:
            buy_score += 1
            buy_signals.append(f"RSI {rsi:.0f} recovering")
        if price <= bb_lower * 1.015:
            buy_score += 1
            buy_signals.append("Near BB lower band")
        if macd_bull:
            buy_score += 1
            buy_signals.append("Bullish MACD crossover")
        if vol_spk >= 1.4:
            buy_score += 1
            buy_signals.append(f"Volume spike {vol_spk:.1f}x")

        sell_score, sell_signals = 0, []
        if rsi > 70:
            sell_score += 1
            sell_signals.append(f"RSI {rsi:.0f} overbought")
        if price >= bb_upper * 0.98:
            sell_score += 1
            sell_signals.append("At BB upper band")
        if macd_bear:
            sell_score += 1
            sell_signals.append("Bearish MACD crossover")

        rules = get_rules(ticker)
        action = ("BUY" if buy_score >= 3 and sell_score < 2 else
                  "SELL" if sell_score >= 2 else "HOLD")
        conviction = ("HIGH" if (buy_score >= 4 or sell_score >= 3) else
                      "MEDIUM" if (buy_score >= 3 or sell_score >= 2) else "LOW")
        change_5d = (price - float(df.iloc[-5]["Close"])) / float(df.iloc[-5]["Close"]) * 100

        return {
            "ticker":       ticker,
            "market_type":  rules["label"],
            "price":        round(price, 2),
            "change_5d":    round(change_5d, 2),
            "RSI":          round(rsi, 1),
            "ATR_pct":      round(atr_pct, 2),
            "vol_spike":    round(vol_spk, 2),
            "above_ema20":  price > ema20,
            "price_vs_bb":  ("above_upper" if price > bb_upper else
                             "below_lower" if price < bb_lower else "inside"),
            "macd_cross":   ("bullish" if macd_bull else "bearish" if macd_bear else "none"),
            "buy_score":    buy_score,
            "sell_score":   sell_score,
            "buy_signals":  buy_signals,
            "sell_signals": sell_signals,
            "action":       action,
            "conviction":   conviction,
            "stop":         round(price * (1 - rules["stop"] / 100), 2),
            "target":       round(price * (1 + rules["target"] / 100), 2),
            "stop_pct":     rules["stop"],
            "target_pct":   rules["target"],
            "as_of":        str(df.index[-1].date()),
        }
    except Exception:
        return {}


def fetch_market_news():
    headlines = []
    if FINNHUB_API_KEY:
        try:
            r = requests.get(
                f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}",
                timeout=10)
            for item in r.json()[:15]:
                h = item.get("headline", "")
                if h:
                    headlines.append(h)
        except Exception:
            pass
    if len(headlines) < 5:
        try:
            import xml.etree.ElementTree as ET
            r = requests.get(
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
                timeout=8)
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                t = item.find("title")
                if t is not None and t.text:
                    headlines.append(t.text)
                if len(headlines) >= 15:
                    break
        except Exception:
            pass
    headlines += [
        "TSX at ~33,650 — energy and financials leading",
        "Bank of Canada holds rate at 2.25%",
        "Canada 10-yr bond yield at 3.48%",
    ]
    return [h for h in headlines if h][:20]


def analyze_with_claude(signal):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""Analyze this 1-3 day swing trade for a Canadian investor:

Ticker: {signal['ticker']} ({signal['market_type']})
Price: ${signal['price']} | 5-day change: {signal['change_5d']:+.1f}%
RSI: {signal['RSI']} | Volume: {signal['vol_spike']}x | {signal['price_vs_bb']}
Buy score: {signal['buy_score']}/5 | Sell score: {signal['sell_score']}/3
Buy signals: {', '.join(signal['buy_signals']) or 'none'}
Sell signals: {', '.join(signal['sell_signals']) or 'none'}
Stop: ${signal['stop']} (-{signal['stop_pct']}%) | Target: ${signal['target']} (+{signal['target_pct']}%)

Output JSON only:
{{"action":"BUY_ALERT"|"SELL_ALERT"|"NO_ACTION","conviction":"HIGH"|"MEDIUM"|"LOW","reasoning":"2-3 sentences","entry_price":0,"stop_loss":0,"take_profit":0,"key_signals":["s1","s2"]}}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = re.sub(r"```json|```", "", response.content[0].text.strip()).strip()
    return json.loads(raw)


def claude_daily_picks(headlines, all_signals):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    signals_summary = "\n".join(
        f"- {s['ticker']}: ${s['price']}, RSI {s['RSI']}, vol {s['vol_spike']}x, score {s['buy_score']}/5"
        for s in all_signals[:20]
    )
    prompt = f"""You are a Wall Street specialist with 20 years in Canadian markets.
Today: {datetime.now().strftime('%A, %B %d, %Y')}

Headlines:
{chr(10).join(f'• {h}' for h in headlines[:10])}

Technical data:
{signals_summary}

Pick TOP 5 Canadian stocks or ETFs for swing trading TODAY.
For each: ticker, why today, entry zone, stop loss, target, hold 1-3 days.
Be specific. Max 300 words."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def claude_news_briefing(headlines, alerts):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    if alerts:
        alert_summary = ", ".join(
            f"{a.get('ticker', '?')} ({a.get('action', '')})"
            for a in alerts
        )
    else:
        alert_summary = "none today"

    prompt = (
        f"Date: {datetime.now().strftime('%A %B %d, %Y')}\n\n"
        f"Headlines:\n" + "\n".join(f"• {h}" for h in headlines) +
        f"\n\nSwing alerts: {alert_summary}\n\n"
        "Write a 200-word pre-market briefing for a Canadian swing trader. "
        "Cover macro, TSX outlook, key risks today."
    )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        return r.ok
    except Exception:
        return False


def format_alert(signal, decision):
    action = decision.get("action", "NO_ACTION")
    if action not in ("BUY_ALERT", "SELL_ALERT"):
        return ""
    emoji = "🟢" if action == "BUY_ALERT" else "🔴"
    label = "BUY" if action == "BUY_ALERT" else "SELL/EXIT"
    price = signal["price"]
    entry = decision.get("entry_price") or price
    sl    = decision.get("stop_loss")   or signal["stop"]
    tp    = decision.get("take_profit") or signal["target"]
    rr    = round((tp - entry) / (entry - sl), 1) if entry != sl else "N/A"
    signals = signal.get("buy_signals" if action == "BUY_ALERT" else "sell_signals", [])
    lines = [
        f"{emoji} <b>{label} — {signal['ticker']}</b> [{decision['conviction']} · {signal['market_type']}]",
        f"",
        f"<b>Price:</b> ${price}  |  <b>Score:</b> {signal['buy_score']}/5  |  <b>Date:</b> {signal['as_of']}",
        f"<b>Entry:</b> ~${entry}",
        f"<b>Stop:</b> ${sl} (−{signal['stop_pct']}%)  |  <b>Target:</b> ${tp} (+{signal['target_pct']}%)",
        f"<b>R:R:</b> 1:{rr}",
        f"",
        f"<b>Why:</b> {decision.get('reasoning', '')}",
        f"",
        f"<b>Signals:</b> {' · '.join(signals[:4])}",
        f"",
        f"<i>RSI {signal['RSI']} · ATR {signal['ATR_pct']}% · Vol {signal['vol_spike']}x · "
        f"{'↑' if signal['above_ema20'] else '↓'} EMA20 · {signal['price_vs_bb']}</i>",
        f"",
        f"⚠️ <i>Execute manually in Wealthsimple. Analysis only.</i>",
    ]
    return "\n".join(lines)


def send_email(subject, html):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT]):
        print("  Email not configured")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        print(f"  Email sent to {EMAIL_RECIPIENT}")
    except Exception as e:
        print(f"  Email error: {e}")


def build_email(briefing, picks, alerts):
    date_str  = datetime.now().strftime("%A, %B %d, %Y — 8:30 AM Toronto")
    news_html = briefing.replace("\n\n", "</p><p>").replace("\n", "<br>")
    picks_html = picks.replace("\n\n", "</p><p>").replace("\n", "<br>")

    rows = ""
    for a in alerts:
        sig    = a.get("signal", {})
        dec    = a.get("decision", {})
        is_buy = dec.get("action") == "BUY_ALERT"
        color  = "#27500A" if is_buy else "#A32D2D"
        bg     = "#EAF3DE" if is_buy else "#FCEBEB"
        label  = "BUY" if is_buy else "SELL"
        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:8px 12px;font-weight:700;color:{color}">{sig.get('ticker','')}</td>
          <td style="padding:8px 12px;color:{color}">{label}</td>
          <td style="padding:8px 12px">{dec.get('conviction','')}</td>
          <td style="padding:8px 12px;font-size:12px">{sig.get('buy_score',0)}/5</td>
          <td style="padding:8px 12px;font-size:12px">Entry ${dec.get('entry_price') or sig.get('price','')} | Stop ${sig.get('stop','')} | TP ${sig.get('target','')}</td>
          <td style="padding:8px 12px;font-size:11px">{dec.get('reasoning','')[:80]}...</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="6" style="padding:14px;color:#888;text-align:center">No high-conviction setups today</td></tr>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f2f2f2;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:720px;margin:20px auto;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.1)">
  <div style="background:#0d2137;padding:24px 30px">
    <div style="font-size:11px;color:#7db3d8;text-transform:uppercase;letter-spacing:.1em">Swing Agent v3 — Pre-Market 8:30 AM</div>
    <div style="font-size:26px;font-weight:700;color:#fff;margin-top:4px">Daily Trade Report</div>
    <div style="font-size:13px;color:#7db3d8;margin-top:4px">{date_str}</div>
  </div>
  <div style="padding:26px 30px">
    <h2 style="font-size:17px;font-weight:600;margin:0 0 14px;color:#111;border-bottom:2px solid #eee;padding-bottom:8px">🌍 Pre-Market Briefing</h2>
    <div style="background:#f8f8f8;border-left:4px solid #0d2137;border-radius:0 8px 8px 0;padding:14px 18px;font-size:14px;line-height:1.8;color:#333">
      <p style="margin:0">{news_html}</p>
    </div>
    <h2 style="font-size:17px;font-weight:600;margin:26px 0 14px;color:#111;border-bottom:2px solid #eee;padding-bottom:8px">⭐ Wall Street Top 5 Canadian Picks Today</h2>
    <div style="background:#fff8e1;border-left:4px solid #f59e0b;border-radius:0 8px 8px 0;padding:14px 18px;font-size:14px;line-height:1.8;color:#333">
      <p style="margin:0">{picks_html}</p>
    </div>
    <h2 style="font-size:17px;font-weight:600;margin:26px 0 14px;color:#111;border-bottom:2px solid #eee;padding-bottom:8px">📡 Swing Trade Alerts</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:#f0f0f0">
        <th style="padding:8px 12px;text-align:left">Ticker</th>
        <th style="padding:8px 12px;text-align:left">Action</th>
        <th style="padding:8px 12px;text-align:left">Conv.</th>
        <th style="padding:8px 12px;text-align:left">Score</th>
        <th style="padding:8px 12px;text-align:left">Levels</th>
        <th style="padding:8px 12px;text-align:left">Reasoning</th>
      </tr>
      {rows}
    </table>
  </div>
  <div style="background:#f5f5f5;padding:12px 30px;font-size:11px;color:#999;border-top:1px solid #eee;text-align:center">
    Swing Agent v3 · Execute manually in Wealthsimple · Not financial advice · {datetime.now().strftime('%H:%M ET')}
  </div>
</div>
</body></html>"""


def run():
    print(f"\n{'='*55}")
    print(f"  SWING AGENT v3 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Scanning {len(WATCHLIST)} stocks...")
    print(f"{'='*55}")

    alerts_sent = 0
    all_alerts  = []
    all_signals = []

    for ticker in WATCHLIST:
        print(f"  {ticker}...", end=" ")
        try:
            df = fetch_stock_data(ticker)
            if df.empty:
                print("no data")
                continue
            signal = build_signal(ticker, df)
            if not signal:
                print("no signal")
                continue
            all_signals.append(signal)
            print(f"RSI={signal['RSI']} score={signal['buy_score']}/5 {signal['action']}")

            if signal["action"] != "HOLD" and signal["conviction"] in ("HIGH", "MEDIUM"):
                decision = analyze_with_claude(signal)
                final    = decision.get("action", "NO_ACTION")
                if final != "NO_ACTION":
                    msg = format_alert(signal, decision)
                    if msg:
                        ok = send_telegram(msg)
                        print(f"    → Telegram: {ok}")
                        alerts_sent += 1
                    all_alerts.append({"signal": signal, "decision": decision})
        except Exception as e:
            print(f"error: {e}")

    print(f"\n  Scan done — {alerts_sent} alert(s)")

    print("  Fetching news...")
    headlines = fetch_market_news()

    print("  Generating Wall Street picks...")
    picks = claude_daily_picks(headlines, all_signals)
    picks_msg = f"⭐ <b>Wall Street Top 5 — {datetime.now().strftime('%b %d')}</b>\n\n{picks}"
    send_telegram(picks_msg)

    if alerts_sent == 0:
        send_telegram(
            f"📊 <b>Swing Agent scan complete</b> — {datetime.now().strftime('%b %d %H:%M')}\n"
            f"No high-conviction setups across {len(WATCHLIST)} stocks today.\n"
            f"<i>See Wall Street picks above.</i>"
        )

    print("  Writing briefing...")
    alert_decisions = [a["decision"] for a in all_alerts]
    briefing = claude_news_briefing(headlines, alert_decisions)

    print("  Sending email...")
    subject = f"📈 Swing Agent — {datetime.now().strftime('%b %d %Y')} pre-market"
    html    = build_email(briefing, picks, all_alerts)
    send_email(subject, html)

    print(f"\n{'='*55}")
    print(f"  All done! ✅")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()
