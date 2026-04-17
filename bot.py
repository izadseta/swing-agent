import os
import json
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
    "NVDA","AMD","AAPL","MSFT","META","TSLA","AMZN","AVGO","COST",
    "SHOP.TO","ENB.TO","SU.TO","CNQ.TO","RY.TO","TD.TO","BNS.TO",
    "ABX.TO","WPM.TO","FTS.TO","MDA.TO",
    "XIU.TO","ZEB.TO","VFV.TO","QQC.TO","XEQT.TO",
]
def fetch_data(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
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
    df["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    macd = ta.trend.MACD(c)
    df["MACD"]     = macd.macd()
    df["MACD_SIG"] = macd.macd_signal()
    df["VOL_AVG"]  = df["Volume"].rolling(20).mean()
    df["VOL_SPK"]  = df["Volume"] / df["VOL_AVG"]
    return df.dropna()

def get_price(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except:
        pass
    return None

def build_signal(ticker, df):
    if df.empty or len(df) < 10:
        return {}
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
        buy_score += 1; buy_signals.append("بالای EMA20")
    if 32 <= rsi <= 55:
        buy_score += 1; buy_signals.append(f"RSI {rsi:.0f} در حال ریکاوری")
    if price <= bb_lower * 1.015:
        buy_score += 1; buy_signals.append("نزدیک باند پایین بولینگر")
    if macd_bull:
        buy_score += 1; buy_signals.append("تقاطع صعودی MACD")
    if vol_spk >= 1.4:
        buy_score += 1; buy_signals.append(f"حجم {vol_spk:.1f}x")

    sell_score, sell_signals = 0, []
    if rsi > 70:
        sell_score += 1; sell_signals.append(f"RSI {rsi:.0f} اشباع خرید")
    if price >= bb_upper * 0.98:
        sell_score += 1; sell_signals.append("نزدیک باند بالای بولینگر")
    if macd_bear:
        sell_score += 1; sell_signals.append("تقاطع نزولی MACD")

    rules = get_rules(ticker)
    action = ("BUY" if buy_score >= 3 and sell_score < 2 else
              "SELL" if sell_score >= 2 else "HOLD")
    conviction = ("HIGH" if (buy_score >= 4 or sell_score >= 3) else
                  "MEDIUM" if (buy_score >= 3 or sell_score >= 2) else "LOW")
    change_5d = (price - float(df.iloc[-5]["Close"])) / float(df.iloc[-5]["Close"]) * 100

    return {
        "ticker": ticker, "price": round(price,2),
        "change_5d": round(change_5d,2), "RSI": round(rsi,1),
        "ATR_pct": round(atr_pct,2), "vol_spike": round(vol_spk,2),
        "above_ema20": price > ema20,
        "price_vs_bb": ("بالای باند" if price > bb_upper else
                        "زیر باند" if price < bb_lower else "داخل باند"),
        "macd_cross": ("صعودی" if macd_bull else "نزولی" if macd_bear else "خنثی"),
        "buy_score": buy_score, "sell_score": sell_score,
        "buy_signals": buy_signals, "sell_signals": sell_signals,
        "action": action, "conviction": conviction,
        "stop": round(price*(1-rules["stop"]/100),2),
        "target": round(price*(1+rules["target"]/100),2),
        "stop_pct": rules["stop"], "target_pct": rules["target"],
        "market_type": rules["label"], "as_of": str(df.index[-1].date()),
    }

def fetch_news(ticker):
    if not FINNHUB_API_KEY:
        return []
    clean = ticker.replace(".TO","")
    try:
        r = requests.get(
            f"https://finnhub.io/api/v1/company-news?symbol={clean}"
            f"&from=2024-01-01&to=2099-12-31&token={FINNHUB_API_KEY}",
            timeout=8)
        items = r.json()
        if isinstance(items, list):
            return [i.get("headline","") for i in items[:4] if i.get("headline")]
    except:
        pass
    return []

def claude_analyze(signal, news):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    action_fa = "خرید" if signal["action"]=="BUY" else "فروش" if signal["action"]=="SELL" else "نگهداری"
    prompt = f"""تحلیل سوئینگ تریدینگ برای {signal['ticker']}:
قیمت: ${signal['price']} | تغییر ۵ روزه: {signal['change_5d']:+.1f}%
RSI: {signal['RSI']} | حجم: {signal['vol_spike']}x | {signal['price_vs_bb']}
امتیاز خرید: {signal['buy_score']}/5 | امتیاز فروش: {signal['sell_score']}/3
سیگنال‌های خرید: {', '.join(signal['buy_signals']) or 'ندارد'}
سیگنال‌های فروش: {', '.join(signal['sell_signals']) or 'ندارد'}
اخبار: {chr(10).join(news) if news else 'اخباری نیست'}
حد ضرر: ${signal['stop']} | هدف: ${signal['target']}
یک تحلیل کوتاه ۳ جمله‌ای به فارسی بنویس که توضیح بده چرا {action_fa} توصیه میشه."""
    r = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=300,
        messages=[{"role":"user","content":prompt}])
    return r.content[0].text.strip()
def format_analysis(signal, commentary, news):
    action = signal["action"]
    if action == "BUY":
        emoji, label = "🟢", "سیگنال خرید"
    elif action == "SELL":
        emoji, label = "🔴", "سیگنال فروش"
    else:
        emoji, label = "🟡", "نگهداری"

    conviction_fa = {"HIGH":"قوی","MEDIUM":"متوسط","LOW":"ضعیف"}.get(signal["conviction"],"")

    lines = [
        f"{emoji} <b>{label} — {signal['ticker']}</b> [{conviction_fa}]",
        f"<b>نوع:</b> {signal['market_type']}  |  <b>تاریخ:</b> {signal['as_of']}",
        f"",
        f"<b>💰 قیمت:</b> ${signal['price']}  ({signal['change_5d']:+.1f}% در ۵ روز)",
        f"<b>🛑 حد ضرر:</b> ${signal['stop']} (−{signal['stop_pct']}%)",
        f"<b>🎯 هدف:</b> ${signal['target']} (+{signal['target_pct']}%)",
        f"",
        f"<b>📊 سیگنال‌ها ({signal['buy_score']}/5):</b>",
    ]
    for s in signal["buy_signals"]:
        lines.append(f"  ✅ {s}")
    for s in signal["sell_signals"]:
        lines.append(f"  ⚠️ {s}")
    lines += [
        f"",
        f"<i>RSI {signal['RSI']} · ATR {signal['ATR_pct']}% · حجم {signal['vol_spike']}x · {signal['price_vs_bb']}</i>",
        f"",
        f"<b>🤖 تحلیل Claude:</b>",
        f"{commentary}",
    ]
    if news:
        lines += [f"", f"<b>📰 اخبار اخیر:</b>"]
        for n in news[:2]:
            lines.append(f"• {n[:80]}...")
    lines += [f"", f"⚠️ <i>فقط تحلیله — خودت در Wealthsimple اجرا کن</i>"]
    return "\n".join(lines)

def format_help():
    return """🤖 <b>راهنمای ربات</b>

<b>کامندها:</b>

/analyze ENB.TO
تحلیل کامل با سیگنال‌ها + اخبار + Claude

/price ENB.TO
فقط قیمت لایو

/news ENB.TO
اخبار اخیر سهم

/watchlist
اسکن کامل واچ‌لیست

/help
این راهنما

<b>مثال:</b>
/analyze SHOP.TO
/analyze NVDA
/price RY.TO"""

def send(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }, timeout=10)

def get_updates(offset=0):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        r = requests.get(url, params={"offset":offset,"timeout":30}, timeout=35)
        return r.json().get("result",[])
    except:
        return []
def handle_message(chat_id, text):
    text = text.strip()
    parts = text.split()
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].upper() if len(parts) > 1 else ""

    if cmd in ("/help", "help"):
        send(chat_id, format_help())

    elif cmd == "/price":
        if not arg:
            send(chat_id, "❌ مثال: /price ENB.TO")
            return
        send(chat_id, "⏳ در حال دریافت قیمت...")
        price = get_price(arg)
        if price:
            send(chat_id, f"💹 <b>{arg}</b>\nقیمت: <b>${price}</b>")
        else:
            send(chat_id, f"❌ قیمت {arg} پیدا نشد")

    elif cmd == "/news":
        if not arg:
            send(chat_id, "❌ مثال: /news ENB.TO")
            return
        send(chat_id, "⏳ در حال دریافت اخبار...")
        news = fetch_news(arg)
        if not news:
            send(chat_id, f"📰 <b>{arg}</b>\nاخباری یافت نشد")
        else:
            lines = [f"📰 <b>اخبار {arg}:</b>", ""]
            for n in news[:4]:
                lines.append(f"• {n[:100]}")
            send(chat_id, "\n".join(lines))

    elif cmd == "/analyze":
        if not arg:
            send(chat_id, "❌ مثال: /analyze ENB.TO")
            return
        send(chat_id, f"⏳ در حال تحلیل {arg}...\nچند ثانیه صبر کن...")
        try:
            df = fetch_data(arg)
            if df.empty:
                send(chat_id, f"❌ داده‌ای برای {arg} پیدا نشد.")
                return
            signal = build_signal(arg, df)
            if not signal:
                send(chat_id, f"❌ داده کافی برای {arg} وجود نداره.")
                return
            news = fetch_news(arg)
            commentary = claude_analyze(signal, news)
            msg = format_analysis(signal, commentary, news)
            send(chat_id, msg)
        except Exception as e:
            send(chat_id, f"❌ خطا: {str(e)[:100]}")

    elif cmd == "/watchlist":
        send(chat_id, f"⏳ اسکن {len(WATCHLIST)} سهم...\nحدود ۵ دقیقه طول میکشه.")
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
            except:
                continue
        if not alerts:
            send(chat_id, "📊 اسکن تموم شد — سیگنال قوی‌ای پیدا نشد.")
            return
        lines = ["📊 <b>نتایج اسکن:</b>\n"]
        for a in alerts:
            emoji = "🟢" if a["action"]=="BUY" else "🔴"
            lines.append(
                f"{emoji} <b>{a['ticker']}</b> — "
                f"{'خرید' if a['action']=='BUY' else 'فروش'} [{a['conviction']}]\n"
                f"  قیمت: ${a['price']} | هدف: ${a['target']} | ضرر: ${a['stop']}"
            )
        lines.append("\n<i>برای تحلیل کامل: /analyze [تیکر]</i>")
        send(chat_id, "\n".join(lines))

    else:
        send(chat_id, "❓ کامند نامشخص.\n/help برای راهنما")


def run():
    print("🤖 ربات آماده‌ست...")
    print("برای توقف: Ctrl+C\n")
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
                    print(f"پیام: {text}")
                    handle_message(chat_id, text)
        except KeyboardInterrupt:
            print("\nربات متوقف شد.")
            break
        except Exception as e:
            print(f"خطا: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run()

