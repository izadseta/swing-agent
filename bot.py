“””
Swing Trade Bot — Render Edition
از Alpha Vantage به جای yfinance استفاده میکنه
روی Render.com کار میکنه
“””

import os
import time
import requests
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(**file**)), “.env”))

ANTHROPIC_API_KEY   = os.getenv(“ANTHROPIC_API_KEY”)
TELEGRAM_BOT_TOKEN  = os.getenv(“TELEGRAM_BOT_TOKEN”)
TELEGRAM_CHAT_ID    = os.getenv(“TELEGRAM_CHAT_ID”)
FINNHUB_API_KEY     = os.getenv(“FINNHUB_API_KEY”, “”)
ALPHA_VANTAGE_KEY   = os.getenv(“ALPHA_VANTAGE_KEY”, “”)

CA_ETFS = {
“VFV.TO”,“XIU.TO”,“XIC.TO”,“XEQT.TO”,“VDY.TO”,
“ZWC.TO”,“ZEB.TO”,“QQC.TO”,“ZAG.TO”,“VCN.TO”,
}

def get_rules(ticker):
if ticker in CA_ETFS:
return {“stop”: 1.5, “target”: 4.0, “label”: “Canadian ETF”}
elif ticker.endswith(”.TO”):
return {“stop”: 2.0, “target”: 5.0, “label”: “Canadian Stock”}
else:
return {“stop”: 2.5, “target”: 6.0, “label”: “US Stock”}

WATCHLIST = [
“NVDA”,“AMD”,“AAPL”,“MSFT”,“META”,“TSLA”,“AMZN”,“AVGO”,
“SHOP.TO”,“ENB.TO”,“SU.TO”,“CNQ.TO”,“RY.TO”,“TD.TO”,
“ABX.TO”,“WPM.TO”,“XIU.TO”,“ZEB.TO”,“VFV.TO”,“QQC.TO”,
]

# ═══════════════════════════════════════════════════════

# MARKET DATA — Alpha Vantage

# ═══════════════════════════════════════════════════════

def fetch_data(ticker):
“”“دریافت داده از Alpha Vantage”””
try:
# برای سهام کانادایی .TO رو حذف میکنیم
symbol = ticker.replace(”.TO”, “.TRT”) if ticker.endswith(”.TO”) else ticker

```
    url = (f"https://www.alphavantage.co/query"
           f"?function=TIME_SERIES_DAILY"
           f"&symbol={symbol}"
           f"&outputsize=compact"
           f"&apikey={ALPHA_VANTAGE_KEY}")

    r = requests.get(url, timeout=15)
    data = r.json()

    if "Time Series (Daily)" not in data:
        # اگه .TRT کار نکرد، .TSX امتحان کن
        if ticker.endswith(".TO"):
            symbol2 = ticker.replace(".TO", ".TSX")
            url2 = url.replace(symbol, symbol2)
            r2 = requests.get(url2, timeout=15)
            data = r2.json()

        if "Time Series (Daily)" not in data:
            return pd.DataFrame()

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = ["Open","High","Low","Close","Volume"]
    df = df.astype(float)

    # محاسبه اندیکاتورها
    close = df["Close"]

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_MID"] = close.rolling(20).mean()
    bb_std       = close.rolling(20).std()
    df["BBU"]    = df["BB_MID"] + 2 * bb_std
    df["BBL"]    = df["BB_MID"] - 2 * bb_std

    # ATR
    df["TR"]  = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - close.shift()).abs(),
        (df["Low"]  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    # MACD
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Volume
    df["VOL_AVG"] = df["Volume"].rolling(20).mean()
    df["VOL_SPK"] = df["Volume"] / df["VOL_AVG"]

    return df.dropna()

except Exception as e:
    print(f"Alpha Vantage error for {ticker}: {e}")
    return pd.DataFrame()
```

def get_price(ticker):
try:
symbol = ticker.replace(”.TO”, “.TRT”) if ticker.endswith(”.TO”) else ticker
url = (f”https://www.alphavantage.co/query”
f”?function=GLOBAL_QUOTE”
f”&symbol={symbol}”
f”&apikey={ALPHA_VANTAGE_KEY}”)
r = requests.get(url, timeout=10)
data = r.json()
price = data.get(“Global Quote”, {}).get(“05. price”)
if price:
return round(float(price), 2)
except Exception:
pass
return None

# ═══════════════════════════════════════════════════════

# SMART MONEY

# ═══════════════════════════════════════════════════════

def detect_order_blocks(df, lookback=10):
obs = []
if len(df) < lookback + 3:
return obs
closes = df[“Close”].values
opens  = df[“Open”].values
highs  = df[“High”].values
lows   = df[“Low”].values
vols   = df[“Volume”].values
vol_avg = df[“VOL_AVG”].values
for i in range(lookback, len(df) - 2):
if (closes[i] < opens[i] and closes[i+1] > opens[i+1] and
closes[i+2] > closes[i+1] and vols[i] > vol_avg[i] * 1.3):
obs.append({
“type”: “bullish_ob”,
“high”: round(float(highs[i]), 2),
“low”:  round(float(lows[i]), 2),
“label”: f”🟦 Bullish OB: ${round(float(lows[i]),2)}–${round(float(highs[i]),2)}”
})
if (closes[i] > opens[i] and closes[i+1] < opens[i+1] and
closes[i+2] < closes[i+1] and vols[i] > vol_avg[i] * 1.3):
obs.append({
“type”: “bearish_ob”,
“high”: round(float(highs[i]), 2),
“low”:  round(float(lows[i]), 2),
“label”: f”🟥 Bearish OB: ${round(float(lows[i]),2)}–${round(float(highs[i]),2)}”
})
return obs[-3:] if obs else []

def detect_fvg(df, lookback=15):
fvgs = []
if len(df) < lookback + 3:
return fvgs
highs = df[“High”].values
lows  = df[“Low”].values
start = max(0, len(df) - lookback)
for i in range(start, len(df) - 2):
if lows[i+2] > highs[i]:
fvgs.append({
“type”: “bullish_fvg”,
“top”:    round(float(lows[i+2]), 2),
“bottom”: round(float(highs[i]), 2),
“label”:  f”🟩 Bullish FVG: ${round(float(highs[i]),2)}–${round(float(lows[i+2]),2)}”
})
if highs[i+2] < lows[i]:
fvgs.append({
“type”: “bearish_fvg”,
“top”:    round(float(lows[i]), 2),
“bottom”: round(float(highs[i+2]), 2),
“label”:  f”🟥 Bearish FVG: ${round(float(highs[i+2]),2)}–${round(float(lows[i]),2)}”
})
return fvgs[-3:] if fvgs else []

def detect_liquidity_sweep(df, lookback=20):
sweeps = []
if len(df) < lookback + 2:
return sweeps
highs  = df[“High”].values
lows   = df[“Low”].values
closes = df[“Close”].values
for i in range(lookback, len(df) - 1):
prev_high = max(highs[i-lookback:i])
prev_low  = min(lows[i-lookback:i])
if highs[i] > prev_high and closes[i] < prev_high:
sweeps.append({
“type”: “bearish_sweep”,
“label”: f”🔴 Liquidity Sweep High: ${round(float(prev_high),2)}”
})
if lows[i] < prev_low and closes[i] > prev_low:
sweeps.append({
“type”: “bullish_sweep”,
“label”: f”🟢 Liquidity Sweep Low: ${round(float(prev_low),2)}”
})
return sweeps[-2:] if sweeps else []

# ═══════════════════════════════════════════════════════

# SIGNAL BUILDER

# ═══════════════════════════════════════════════════════

def build_signal(ticker, df):
if df.empty or len(df) < 10:
return {}
try:
last, prev = df.iloc[-1], df.iloc[-2]
price    = float(last[“Close”])
rsi      = float(last[“RSI”])
ema20    = float(last[“EMA20”])
ema50    = float(last[“EMA50”])
bb_upper = float(last[“BBU”])
bb_lower = float(last[“BBL”])
macd_val = float(last[“MACD”])
macd_sig = float(last[“MACD_SIG”])
vol_spk  = float(last[“VOL_SPK”])
atr_pct  = float(last[“ATR”]) / price * 100
macd_bull = macd_val > macd_sig and float(prev[“MACD”]) <= float(prev[“MACD_SIG”])
macd_bear = macd_val < macd_sig and float(prev[“MACD”]) >= float(prev[“MACD_SIG”])

```
    buy_score, buy_signals = 0, []
    if price > ema20:
        buy_score += 1; buy_signals.append("بالای EMA20")
    if 32 <= rsi <= 55:
        buy_score += 1; buy_signals.append(f"RSI {rsi:.0f} ریکاوری")
    if price <= bb_lower * 1.015:
        buy_score += 1; buy_signals.append("نزدیک BB پایین")
    if macd_bull:
        buy_score += 1; buy_signals.append("MACD صعودی")
    if vol_spk >= 1.4:
        buy_score += 1; buy_signals.append(f"حجم {vol_spk:.1f}x")

    sell_score, sell_signals = 0, []
    if rsi > 70:
        sell_score += 1; sell_signals.append(f"RSI {rsi:.0f} اشباع")
    if price >= bb_upper * 0.98:
        sell_score += 1; sell_signals.append("BB بالا")
    if macd_bear:
        sell_score += 1; sell_signals.append("MACD نزولی")

    obs     = detect_order_blocks(df)
    fvgs    = detect_fvg(df)
    sweeps  = detect_liquidity_sweep(df)
    sm_score, sm_notes = 0, []

    for ob in obs:
        if ob["type"] == "bullish_ob" and ob["low"] <= price <= ob["high"] * 1.02:
            sm_score += 1; sm_notes.append("در Bullish OB")
    for fvg in fvgs:
        if fvg["type"] == "bullish_fvg" and fvg["bottom"] <= price <= fvg["top"]:
            sm_score += 1; sm_notes.append("در Bullish FVG")
    for s in sweeps:
        if s["type"] == "bullish_sweep":
            sm_score += 1; sm_notes.append("Bullish Sweep")
        else:
            sm_score -= 1; sm_notes.append("Bearish Sweep")

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
        "price_vs_bb": ("بالای باند" if price > bb_upper else
                        "زیر باند" if price < bb_lower else "داخل باند"),
        "macd_cross": ("صعودی" if macd_bull else "نزولی" if macd_bear else "خنثی"),
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
    return {}
```

# ═══════════════════════════════════════════════════════

# NEWS + CLAUDE

# ═══════════════════════════════════════════════════════

def fetch_news(ticker):
if not FINNHUB_API_KEY:
return []
clean = ticker.replace(”.TO”, “”)
try:
r = requests.get(
f”https://finnhub.io/api/v1/company-news?symbol={clean}”
f”&from=2024-01-01&to=2099-12-31&token={FINNHUB_API_KEY}”,
timeout=8)
items = r.json()
if isinstance(items, list):
return [i.get(“headline”,””) for i in items[:4] if i.get(“headline”)]
except Exception:
pass
return []

def claude_analyze(signal, news):
client = Anthropic(api_key=ANTHROPIC_API_KEY)
action_fa = (“خرید” if signal[“action”]==“BUY” else
“فروش” if signal[“action”]==“SELL” else “نگهداری”)
sm_text = “”
if signal.get(“liq_sweeps”):
sm_text += f”\nLiquidity: {’, ‘.join(s[‘label’] for s in signal[‘liq_sweeps’])}”
if signal.get(“order_blocks”):
sm_text += f”\nOrder Blocks: {’, ‘.join(o[‘label’] for o in signal[‘order_blocks’][-2:])}”
if signal.get(“fvgs”):
sm_text += f”\nFVG: {’, ’.join(f[‘label’] for f in signal[‘fvgs’][-2:])}”

```
prompt = f"""تحلیل سوئینگ + Smart Money برای {signal['ticker']}:
```

قیمت: ${signal[‘price’]} | RSI: {signal[‘RSI’]} | حجم: {signal[‘vol_spike’]}x
امتیاز خرید: {signal[‘buy_score’]}/5 | امتیاز فروش: {signal[‘sell_score’]}/3
سیگنال‌ها: {’, ’.join(signal[‘buy_signals’]+signal[‘sell_signals’]) or ‘ندارد’}
Smart Money Score: {signal[‘sm_score’]}/3{sm_text}
اخبار: {chr(10).join(news) if news else ‘ندارد’}
حد ضرر: ${signal[‘stop’]} | هدف: ${signal[‘target’]}
۴ جمله فارسی: ۱.تکنیکال ۲.Smart Money ۳.توصیه({action_fa}) ۴.ریسک”””

```
r = client.messages.create(
    model="claude-sonnet-4-5", max_tokens=350,
    messages=[{"role":"user","content":prompt}])
return r.content[0].text.strip()
```

# ═══════════════════════════════════════════════════════

# FORMAT

# ═══════════════════════════════════════════════════════

def format_analysis(signal, commentary, news):
action = signal[“action”]
emoji  = “🟢” if action==“BUY” else “🔴” if action==“SELL” else “🟡”
label  = “سیگنال خرید” if action==“BUY” else “سیگنال فروش” if action==“SELL” else “نگهداری”
conv   = {“HIGH”:“قوی”,“MEDIUM”:“متوسط”,“LOW”:“ضعیف”}.get(signal[“conviction”],””)
sm     = signal.get(“sm_score”,0)
sm_e   = “🐋✅” if sm>=2 else “🐋➖” if sm>=0 else “🐋⚠️”

```
lines = [
    f"{emoji} <b>{label} — {signal['ticker']}</b> [{conv}]",
    f"<b>نوع:</b> {signal['market_type']} | <b>تاریخ:</b> {signal['as_of']}",
    f"",
    f"<b>💰 قیمت:</b> ${signal['price']} ({signal['change_5d']:+.1f}% در ۵ روز)",
    f"<b>🛑 حد ضرر:</b> ${signal['stop']} (−{signal['stop_pct']}%)",
    f"<b>🎯 هدف:</b> ${signal['target']} (+{signal['target_pct']}%)",
    f"",
    f"<b>📊 تکنیکال ({signal['buy_score']}/5):</b>",
]
for s in signal["buy_signals"]:
    lines.append(f"  ✅ {s}")
for s in signal["sell_signals"]:
    lines.append(f"  ⚠️ {s}")
lines += [
    f"<i>RSI {signal['RSI']} · ATR {signal['ATR_pct']}% · حجم {signal['vol_spike']}x · {signal['price_vs_bb']}</i>",
    f"",
    f"<b>{sm_e} Smart Money ({sm}/3):</b>",
]
if signal.get("liq_sweeps"):
    for s in signal["liq_sweeps"]:
        lines.append(f"  {s['label']}")
else:
    lines.append("  ⬜ Liquidity Sweep: ندارد")
if signal.get("order_blocks"):
    for ob in signal["order_blocks"][-2:]:
        lines.append(f"  {ob['label']}")
else:
    lines.append("  ⬜ Order Block: ندارد")
if signal.get("fvgs"):
    for fvg in signal["fvgs"][-2:]:
        lines.append(f"  {fvg['label']}")
else:
    lines.append("  ⬜ FVG: ندارد")
if signal.get("sm_notes"):
    lines.append(f"  💡 {' | '.join(signal['sm_notes'])}")
lines += [
    f"",
    f"<b>🤖 تحلیل Claude:</b>",
    f"{commentary}",
]
if news:
    lines += [f"", f"<b>📰 اخبار:</b>"]
    for n in news[:2]:
        lines.append(f"• {n[:80]}...")
lines += [f"", f"⚠️ <i>فقط تحلیله — در Wealthsimple اجرا کن</i>"]
return "\n".join(lines)
```

def format_help():
return “”“🤖 <b>Swing Agent — Smart Money Edition</b>

<b>کامندها:</b>
/analyze ENB.TO — تحلیل کامل
/price ENB.TO — قیمت لایو
/news ENB.TO — اخبار
/watchlist — اسکن واچ‌لیست
/help — راهنما

<b>Smart Money:</b>
🐋 Order Blocks
🟩 Fair Value Gaps
🔴 Liquidity Sweeps”””

# ═══════════════════════════════════════════════════════

# TELEGRAM

# ═══════════════════════════════════════════════════════

def send(chat_id, text):
url = f”https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage”
try:
requests.post(url, json={
“chat_id”: chat_id, “text”: text, “parse_mode”: “HTML”
}, timeout=10)
except Exception:
pass

def get_updates(offset=0):
url = f”https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates”
try:
r = requests.get(url, params={“offset”:offset,“timeout”:30}, timeout=35)
return r.json().get(“result”, [])
except Exception:
return []

def handle_message(chat_id, text):
parts = text.strip().split()
cmd   = parts[0].lower() if parts else “”
arg   = parts[1].upper() if len(parts) > 1 else “”

```
if cmd in ("/help", "/start", "help"):
    send(chat_id, format_help())

elif cmd == "/price":
    if not arg:
        send(chat_id, "❌ مثال: /price ENB.TO")
        return
    send(chat_id, "⏳ در حال دریافت قیمت...")
    price = get_price(arg)
    send(chat_id, f"💹 <b>{arg}</b>\nقیمت: <b>${price}</b>" if price
         else f"❌ قیمت {arg} پیدا نشد")

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
    send(chat_id, f"⏳ در حال تحلیل {arg}...\n(۱۵-۲۰ ثانیه صبر کن)")
    try:
        df = fetch_data(arg)
        if df.empty:
            send(chat_id, f"❌ داده‌ای برای {arg} پیدا نشد.")
            return
        signal = build_signal(arg, df)
        if not signal:
            send(chat_id, f"❌ داده کافی برای {arg} وجود نداره.")
            return
        news       = fetch_news(arg)
        commentary = claude_analyze(signal, news)
        send(chat_id, format_analysis(signal, commentary, news))
    except Exception as e:
        send(chat_id, f"❌ خطا: {str(e)[:100]}")

elif cmd == "/watchlist":
    send(chat_id, f"⏳ اسکن {len(WATCHLIST)} سهم...\nحدود ۱۰ دقیقه طول میکشه.")
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
            time.sleep(12)  # Alpha Vantage rate limit
        except Exception:
            continue
    if not alerts:
        send(chat_id, "📊 اسکن تموم شد — سیگنال قوی‌ای پیدا نشد.")
        return
    lines = ["📊 <b>نتایج اسکن:</b>\n"]
    for a in alerts:
        emoji = "🟢" if a["action"]=="BUY" else "🔴"
        sm_tag = f" 🐋{a.get('sm_score',0)}/3" if a.get("sm_score",0)>0 else ""
        lines.append(
            f"{emoji} <b>{a['ticker']}</b> [{a['conviction']}]{sm_tag}\n"
            f"  ${a['price']} | هدف ${a['target']} | ضرر ${a['stop']}"
        )
    lines.append("\n<i>/analyze [تیکر] برای تحلیل کامل</i>")
    send(chat_id, "\n".join(lines))

else:
    send(chat_id, "❓ /help برای راهنما")
```

# ═══════════════════════════════════════════════════════

# MAIN — HTTP server برای Render

# ═══════════════════════════════════════════════════════

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
def do_GET(self):
self.send_response(200)
self.end_headers()
self.wfile.write(b”Swing Agent Bot Running”)
def log_message(self, *args):
pass

def start_health_server():
port = int(os.getenv(“PORT”, 8080))
server = HTTPServer((“0.0.0.0”, port), HealthHandler)
server.serve_forever()

def run():
print(“🤖 Swing Agent Bot — Smart Money + Alpha Vantage”)
print(“برای توقف: Ctrl+C\n”)

```
# Health check server برای Render
t = threading.Thread(target=start_health_server, daemon=True)
t.start()
print("Health server started")

offset = 0
while True:
    try:
        updates = get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            msg     = update.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text    = msg.get("text", "")
            if chat_id and text:
                print(f"📩 {chat_id}: {text}")
                handle_message(chat_id, text)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"خطا: {e}")
        time.sleep(5)
```

if **name** == “**main**”:
run()