import os
import time
import requests
import pandas as pd
import ta
import yfinance as yf
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
FINNHUB_API_KEY    = os.getenv("FINNHUB_API_KEY", "")

CA_ETFS = {
    "VFV.TO","XIU.TO","XIC.TO","XEQT.TO","VDY.TO",
    "ZWC.TO","ZEB.TO","QQC.TO","ZAG.TO","VCN.TO",
}

def get_rules(ticker):
    if ticker in CA_ETFS:
        return {"stop": 1.5, "target": 4.0, "label": "Canadian ETF"}
    elif ticker.endswith(".TO"):
        return {"stop": 2.0, "target": 5.0, "label": "Canadian Stock"}
    else:
        return {"stop": 2.5, "target": 6.0, "label": "US Stock"}

WATCHLIST = [
    "NVDA","AMD","AAPL","MSFT","META","TSLA","AMZN","AVGO",
    "SHOP.TO","ENB.TO","SU.TO","CNQ.TO","RY.TO","TD.TO",
    "ABX.TO","WPM.TO","XIU.TO","ZEB.TO","VFV.TO","QQC.TO",
]

def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c, h, l = df["Close"], df["High"], df["Low"]
        df["EMA20"] = ta.trend.EMAIndicator(c, window=20).ema_indicator()
        df["EMA50"] = ta.trend.EMAIndicator(c, window=50).ema_indicator()
        df["RSI"]   = ta.momentum.RSIIndicator(c, window=14).rsi()
        bb = ta.volatility.BollingerBands(c, window=20)
        df["BBU"] = bb.bollinger_hband()
        df["BBL"] = bb.bollinger_lband()
        df["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        macd = ta.trend.MACD(c)
        df["MACD"]     = macd.macd()
        df["MACD_SIG"] = macd.macd_signal()
        df["VOL_AVG"]  = df["Volume"].rolling(20).mean()
        df["VOL_SPK"]  = df["Volume"] / df["VOL_AVG"]
        return df.dropna()
    except Exception as e:
        print(f"fetch error {ticker}: {e}")
        return pd.DataFrame()

def get_price(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None

def detect_order_blocks(df, lookback=10):
    obs = []
    if len(df) < lookback + 3:
        return obs
    closes = df["Close"].values
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    vols   = df["Volume"].values
    vol_avg = df["VOL_AVG"].values
    for i in range(lookback, len(df) - 2):
        if (closes[i] < opens[i] and closes[i+1] > opens[i+1] and
                closes[i+2] > closes[i+1] and vols[i] > vol_avg[i] * 1.3):
            obs.append({
                "type": "bullish_ob",
                "high": round(float(highs[i]), 2),
                "low":  round(float(lows[i]), 2),
                "label": "Bullish OB: $" + str(round(float(lows[i]),2)) + "-$" + str(round(float(highs[i]),2))
            })
        if (closes[i] > opens[i] and closes[i+1] < opens[i+1] and
                closes[i+2] < closes[i+1] and vols[i] > vol_avg[i] * 1.3):
            obs.append({
                "type": "bearish_ob",
                "high": round(float(highs[i]), 2),
                "low":  round(float(lows[i]), 2),
                "label": "Bearish OB: $" + str(round(float(lows[i]),2)) + "-$" + str(round(float(highs[i]),2))
            })
    return obs[-3:] if obs else []

def detect_fvg(df, lookback=15):
    fvgs = []
    if len(df) < lookback + 3:
        return fvgs
    highs = df["High"].values
    lows  = df["Low"].values
    start = max(0, len(df) - lookback)
    for i in range(start, len(df) - 2):
        if lows[i+2] > highs[i]:
            fvgs.append({
                "type": "bullish_fvg",
                "label": "Bullish FVG: $" + str(round(float(highs[i]),2)) + "-$" + str(round(float(lows[i+2]),2))
            })
        if highs[i+2] < lows[i]:
            fvgs.append({
                "type": "bearish_fvg",
                "label": "Bearish FVG: $" + str(round(float(highs[i+2]),2)) + "-$" + str(round(float(lows[i]),2))
            })
    return fvgs[-3:] if fvgs else []

def detect_liquidity_sweep(df, lookback=20):
    sweeps = []
    if len(df) < lookback + 2:
        return sweeps
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    for i in range(lookback, len(df) - 1):
        prev_high = max(highs[i-lookback:i])
        prev_low  = min(lows[i-lookback:i])
        if highs[i] > prev_high and closes[i] < prev_high:
            sweeps.append({"type": "bearish_sweep",
                "label": "Bearish Sweep High: $" + str(round(float(prev_high),2))})
        if lows[i] < prev_low and closes[i] > prev_low:
            sweeps.append({"type": "bullish_sweep",
                "label": "Bullish Sweep Low: $" + str(round(float(prev_low),2))})
    return sweeps[-2:] if sweeps else []

def build_signal(ticker, df):
    if df.empty or len(df) < 10:
        return {}
    try:
        last, prev = df.iloc[-1], df.iloc[-2]
        price    = float(last["Close"])
        rsi      = float(last["RSI"])
        ema20    = float(last["EMA20"])
        ema50    = float(last["EMA50"])
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
            buy_signals.append("Above EMA20")
        if 32 <= rsi <= 55:
            buy_score += 1
            buy_signals.append("RSI " + str(round(rsi,0)) + " recovering")
        if price <= bb_lower * 1.015:
            buy_score += 1
            buy_signals.append("Near BB lower")
        if macd_bull:
            buy_score += 1
            buy_signals.append("Bullish MACD crossover")
        if vol_spk >= 1.4:
            buy_score += 1
            buy_signals.append("Volume " + str(round(vol_spk,1)) + "x")

        sell_score, sell_signals = 0, []
        if rsi > 70:
            sell_score += 1
            sell_signals.append("RSI " + str(round(rsi,0)) + " overbought")
        if price >= bb_upper * 0.98:
            sell_score += 1
            sell_signals.append("At BB upper")
        if macd_bear:
            sell_score += 1
            sell_signals.append("Bearish MACD crossover")

        obs    = detect_order_blocks(df)
        fvgs   = detect_fvg(df)
        sweeps = detect_liquidity_sweep(df)
        sm_score, sm_notes = 0, []

        for ob in obs:
            if ob["type"] == "bullish_ob" and ob["low"] <= price <= ob["high"] * 1.02:
                sm_score += 1
                sm_notes.append("In Bullish OB")
        for fvg in fvgs:
            if fvg["type"] == "bullish_fvg":
                sm_score += 1
                sm_notes.append("Bullish FVG present")
        for s in sweeps:
            if s["type"] == "bullish_sweep":
                sm_score += 1
                sm_notes.append("Bullish Liquidity Sweep")
            else:
                sm_score -= 1
                sm_notes.append("Bearish Liquidity Sweep")

        rules = get_rules(ticker)
        action = ("BUY" if buy_score >= 3 and sell_score < 2 else
                  "SELL" if sell_score >= 2 else "HOLD")
        conviction = ("HIGH" if (buy_score >= 4 or sell_score >= 3) else
                      "MEDIUM" if (buy_score >= 3 or sell_score >= 2) else "LOW")
        if sm_score >= 2 and conviction == "MEDIUM":
            conviction = "HIGH"
        change_5d = (price - float(df.iloc[-5]["Close"])) / float(df.iloc[-5]["Close"]) * 100

        return {
            "ticker": ticker, "market_type": rules["label"],
            "price": round(price, 2), "change_5d": round(change_5d, 2),
            "RSI": round(rsi, 1), "ATR_pct": round(atr_pct, 2),
            "vol_spike": round(vol_spk, 2),
            "above_ema20": price > ema20, "above_ema50": price > ema50,
            "price_vs_bb": ("above upper" if price > bb_upper else
                            "below lower" if price < bb_lower else "inside bands"),
            "macd_cross": ("bullish" if macd_bull else "bearish" if macd_bear else "none"),
            "buy_score": buy_score, "sell_score": sell_score,
            "buy_signals": buy_signals, "sell_signals": sell_signals,
            "action": action, "conviction": conviction,
            "stop": round(price*(1-rules["stop"]/100), 2),
            "target": round(price*(1+rules["target"]/100), 2),
            "stop_pct": rules["stop"], "target_pct": rules["target"],
            "as_of": str(df.index[-1].date()),
            "order_blocks": obs, "fvgs": fvgs,
            "liq_sweeps": sweeps, "sm_score": sm_score, "sm_notes": sm_notes,
        }
    except Exception as e:
        print(f"signal error: {e}")
        return {}

def fetch_news(ticker):
    if not FINNHUB_API_KEY:
        return []
    clean = ticker.replace(".TO", "")
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news?symbol=" + clean +
            "&from=2024-01-01&to=2099-12-31&token=" + FINNHUB_API_KEY,
            timeout=8)
        items = r.json()
        if isinstance(items, list):
            return [i.get("headline","") for i in items[:4] if i.get("headline")]
    except Exception:
        pass
    return []

def claude_analyze(signal, news):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    sm_text = ""
    if signal.get("liq_sweeps"):
        sm_text += " | Liquidity: " + ", ".join(s["label"] for s in signal["liq_sweeps"])
    if signal.get("order_blocks"):
        sm_text += " | OB: " + ", ".join(o["label"] for o in signal["order_blocks"][-2:])
    if signal.get("fvgs"):
        sm_text += " | FVG: " + ", ".join(f["label"] for f in signal["fvgs"][-2:])

    prompt = ("Analyze this swing trade for a Canadian investor. Respond in Farsi.\n"
        "Ticker: " + signal["ticker"] + " (" + signal["market_type"] + ")\n"
        "Price: $" + str(signal["price"]) + " | 5-day: " + str(signal["change_5d"]) + "%\n"
        "RSI: " + str(signal["RSI"]) + " | MACD: " + signal["macd_cross"] + " | Vol: " + str(signal["vol_spike"]) + "x\n"
        "EMA20: " + ("above" if signal["above_ema20"] else "below") + " | " + signal["price_vs_bb"] + "\n"
        "Buy score: " + str(signal["buy_score"]) + "/5 | Sell score: " + str(signal["sell_score"]) + "/3\n"
        "Signals: " + ", ".join(signal["buy_signals"]+signal["sell_signals"]) + "\n"
        "Smart Money score: " + str(signal["sm_score"]) + "/3" + sm_text + "\n"
        "News: " + (chr(10).join(news) if news else "none") + "\n"
        "Stop: $" + str(signal["stop"]) + " | Target: $" + str(signal["target"]) + "\n"
        "Write 4 sentences in Farsi: 1.Technical 2.Smart Money 3.Recommendation 4.Risk")

    r = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=350,
        messages=[{"role":"user","content":prompt}])
    return r.content[0].text.strip()

def send(chat_id, text):
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": chat_id, "text": text, "parse_mode": "HTML"
        }, timeout=10)
    except Exception:
        pass

def get_updates(offset=0):
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/getUpdates"
    try:
        r = requests.get(url, params={"offset":offset,"timeout":30}, timeout=35)
        return r.json().get("result", [])
    except Exception:
        return []

def handle_message(chat_id, text):
    parts = text.strip().split()
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].upper() if len(parts) > 1 else ""

    if cmd in ("/help", "/start", "help"):
        send(chat_id, "<b>Swing Agent - Smart Money Edition</b>\n\n"
            "/analyze ENB.TO - Full analysis + Smart Money\n"
            "/price ENB.TO - Live price\n"
            "/news ENB.TO - Recent news\n"
            "/watchlist - Scan all stocks\n"
            "/help - This menu\n\n"
            "Smart Money detects:\n"
            "- Order Blocks\n"
            "- Fair Value Gaps (FVG)\n"
            "- Liquidity Sweeps")

    elif cmd == "/price":
        if not arg:
            send(chat_id, "Example: /price ENB.TO")
            return
        send(chat_id, "Getting price...")
        price = get_price(arg)
        send(chat_id, "<b>" + arg + "</b>: $" + str(price) if price
             else "Price not found for " + arg)

    elif cmd == "/news":
        if not arg:
            send(chat_id, "Example: /news ENB.TO")
            return
        news = fetch_news(arg)
        if not news:
            send(chat_id, "No news found for " + arg)
        else:
            lines = ["<b>News " + arg + ":</b>"]
            for n in news[:4]:
                lines.append("- " + n[:100])
            send(chat_id, "\n".join(lines))

    elif cmd == "/analyze":
        if not arg:
            send(chat_id, "Example: /analyze ENB.TO")
            return
        send(chat_id, "Analyzing " + arg + "... (15-20 sec)")
        try:
            df = fetch_data(arg)
            if df.empty:
                send(chat_id, "No data found for " + arg)
                return
            signal = build_signal(arg, df)
            if not signal:
                send(chat_id, "Not enough data for " + arg)
                return
            news = fetch_news(arg)
            commentary = claude_analyze(signal, news)
            action = signal["action"]
            emoji = "🟢" if action=="BUY" else "🔴" if action=="SELL" else "🟡"
            label = "BUY" if action=="BUY" else "SELL" if action=="SELL" else "HOLD"
            conv = {"HIGH":"Strong","MEDIUM":"Medium","LOW":"Weak"}.get(signal["conviction"],"")
            sm = signal.get("sm_score", 0)
            sm_e = "🐋 Smart Money: Bullish" if sm >= 2 else "🐋 Smart Money: Neutral" if sm >= 0 else "🐋 Smart Money: Bearish"

            lines = [
                emoji + " <b>" + label + " - " + signal["ticker"] + "</b> [" + conv + "]",
                "<b>Type:</b> " + signal["market_type"] + " | <b>Date:</b> " + signal["as_of"],
                "",
                "<b>Price:</b> $" + str(signal["price"]) + " (" + str(signal["change_5d"]) + "% 5-day)",
                "<b>Stop:</b> $" + str(signal["stop"]) + " (-" + str(signal["stop_pct"]) + "%)",
                "<b>Target:</b> $" + str(signal["target"]) + " (+" + str(signal["target_pct"]) + "%)",
                "",
                "<b>Technical (" + str(signal["buy_score"]) + "/5):</b>",
            ]
            for s in signal["buy_signals"]:
                lines.append("  + " + s)
            for s in signal["sell_signals"]:
                lines.append("  ! " + s)

            lines += [
                "<i>RSI " + str(signal["RSI"]) + " | MACD " + signal["macd_cross"] +
                " | ATR " + str(signal["ATR_pct"]) + "% | Vol " + str(signal["vol_spike"]) + "x | " + signal["price_vs_bb"] + "</i>",
                "",
                "<b>" + sm_e + " (" + str(sm) + "/3):</b>",
            ]

            if signal.get("liq_sweeps"):
                for s in signal["liq_sweeps"]:
                    lines.append("  " + s["label"])
            else:
                lines.append("  No Liquidity Sweep detected")

            if signal.get("order_blocks"):
                for ob in signal["order_blocks"][-2:]:
                    lines.append("  " + ob["label"])
            else:
                lines.append("  No Order Block detected")

            if signal.get("fvgs"):
                for fvg in signal["fvgs"][-2:]:
                    lines.append("  " + fvg["label"])
            else:
                lines.append("  No FVG detected")

            if signal.get("sm_notes"):
                lines.append("  Note: " + " | ".join(signal["sm_notes"]))

            lines += [
                "",
                "<b>Claude Analysis:</b>",
                commentary,
            ]
            if news:
                lines += ["", "<b>News:</b>"]
                for n in news[:2]:
                    lines.append("- " + n[:80])
            lines.append("\n<i>Execute manually in Wealthsimple</i>")
            send(chat_id, "\n".join(lines))
        except Exception as e:
            send(chat_id, "Error: " + str(e)[:100])

    elif cmd == "/watchlist":
        send(chat_id, "Scanning " + str(len(WATCHLIST)) + " stocks... (~5 min)")
        alerts = []
        for ticker in WATCHLIST:
            try:
                df = fetch_data(ticker)
                if df.empty:
                    continue
                signal = build_signal(ticker, df)
                if not signal:
                    continue
                if signal["action"] != "HOLD" and signal["conviction"] in ("HIGH","MEDIUM"):
                    alerts.append(signal)
            except Exception:
                continue
        if not alerts:
            send(chat_id, "Scan done - no strong signals found.")
            return
        lines = ["<b>Watchlist Results:</b>", ""]
        for a in alerts:
            emoji = "🟢" if a["action"]=="BUY" else "🔴"
            sm_tag = " | 🐋" + str(a.get("sm_score",0)) + "/3" if a.get("sm_score",0) > 0 else ""
            lines.append(emoji + " <b>" + a["ticker"] + "</b> [" + a["conviction"] + "]" + sm_tag)
            lines.append("  $" + str(a["price"]) + " | Target $" + str(a["target"]) + " | Stop $" + str(a["stop"]))
        lines.append("\nUse /analyze [ticker] for full analysis")
        send(chat_id, "\n".join(lines))

    else:
        send(chat_id, "Unknown command. Use /help")

def run():
    print("Swing Agent Bot - Smart Money + yfinance")
    print("Ready!")
    offset = 0
    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")
                if chat_id and text:
                    print("Message: " + str(text))
                    handle_message(chat_id, text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error: " + str(e))
            time.sleep(5)

if __name__ == "__main__":
    run()
