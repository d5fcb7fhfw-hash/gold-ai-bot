import os
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

# ==========================================
# 環境変数と設定
# ==========================================
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
LINE_USER_ID = os.environ.get("LINE_USER_ID", "").strip()

LOG_FILE = "trade_log.json"

# 最優先VIP銘柄
GOLD_META = {"symbol": "XAU/USD", "name": "ゴールド（最優先VIP）", "currencies": ["USD"], "pip_unit": 1.0, "digits": 2}

# サブ監視6銘柄
SUB_SYMBOLS = [
    {"symbol": "USD/JPY", "name": "ドル円", "currencies": ["USD", "JPY"], "pip_unit": 0.01, "digits": 3},
    {"symbol": "EUR/USD", "name": "ユーロドル", "currencies": ["EUR", "USD"], "pip_unit": 0.0001, "digits": 5},
    {"symbol": "GBP/JPY", "name": "ポンド円", "currencies": ["GBP", "JPY"], "pip_unit": 0.01, "digits": 3},
    {"symbol": "GBP/USD", "name": "ポンドドル", "currencies": ["GBP", "USD"], "pip_unit": 0.0001, "digits": 5},
    {"symbol": "EUR/JPY", "name": "ユーロ円", "currencies": ["EUR", "JPY"], "pip_unit": 0.01, "digits": 3},
    {"symbol": "AUD/USD", "name": "豪ドル米ドル", "currencies": ["AUD", "USD"], "pip_unit": 0.0001, "digits": 5},
]

ALL_SYMBOLS = [GOLD_META] + SUB_SYMBOLS
JST = datetime.timezone(datetime.timedelta(hours=9))


def now_jst():
    return datetime.datetime.now(JST)


# ==========================================
# Twelve Data API 取得（レート制限完全回避）
# ==========================================
def fetch_batch_1h(symbols_list):
    joined = ",".join(symbols_list)
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={joined}&interval=1h&outputsize=60&apikey={TWELVE_DATA_API_KEY}"
    )
    for _ in range(2):
        res = requests.get(url, timeout=20)
        data = res.json()
        if data.get("code") == 429:
            print("バッチ取得レート制限待機（65秒）...")
            time.sleep(65)
            continue
        result = {}
        for sym in symbols_list:
            item = data.get(sym, {})
            if "values" in item:
                df = pd.DataFrame(item["values"]).iloc[::-1].reset_index(drop=True)
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float)
                result[sym] = df
        return result
    return {}


def fetch_single_tf(symbol, interval, outputsize=50):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    )
    for _ in range(2):
        res = requests.get(url, timeout=15)
        data = res.json()
        if "values" in data:
            df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            return df
        if data.get("code") == 429 or "run out of API credits" in data.get("message", ""):
            print(f"[{symbol} {interval}] レート制限待機（65秒）...")
            time.sleep(65)
        else:
            print(f"[{symbol} {interval}] 取得エラー: {data.get('message')}")
            return None
    return None


# ==========================================
# 精密テクニカル＆プライスアクション解析
# ==========================================
def analyze_dataframe(df, pa_bars=3):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi14 = (100 - (100 / (1 + rs))).iloc[-1]

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_smooth.replace(0, 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_smooth.replace(0, 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx14 = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

    bb_mid = close.rolling(20).mean().iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    swing_high = high.iloc[-24:].max()
    swing_low = low.iloc[-24:].min()

    pa_lines = []
    for i in range(-pa_bars, 0):
        o, h, l, c = open_p.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        candle_type = "陽線" if c >= o else "陰線"
        pa_lines.append(
            f"{df['datetime'].iloc[i][-5:]} {candle_type} (始:{o} 高:{h} 安:{l} 終:{c} | 上ヒゲ:{upper_wick:.2f} 下ヒゲ:{lower_wick:.2f} 実体:{body:.2f})"
        )

    trend = "UP" if (close.iloc[-1] > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]) else (
        "DOWN" if (close.iloc[-1] < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]) else "RANGE"
    )

    return {
        "close": close.iloc[-1],
        "ema20": ema20.iloc[-1],
        "ema50": ema50.iloc[-1],
        "rsi14": rsi14,
        "atr14": atr14,
        "adx14": adx14,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "trend": trend,
        "pa_text": "\n    ".join(pa_lines),
    }


# ==========================================
# 経済指標チェック
# ==========================================
def get_blocked_currencies_and_events():
    blocked = set()
    upcoming_list = []
    try:
        res = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        if res.status_code != 200:
            return blocked, "指標取得スキップ"
        events = res.json()
        now = now_jst()
        for ev in events:
            if ev.get("impact") != "High":
                continue
            dt_str = ev.get("date", "")
            country = ev.get("country", "")
            try:
                ev_dt = datetime.datetime.fromisoformat(dt_str).astimezone(JST)
                diff_min = (ev_dt - now).total_seconds() / 60.0
                if -45 <= diff_min <= 45:
                    blocked.add(country)
                if -30 <= diff_min <= 180:
                    upcoming_list.append(f"{ev_dt.strftime('%H:%M')} [{country}] {ev.get('title')}")
            except Exception:
                continue
        return blocked, ("\n".join(upcoming_list) if upcoming_list else "直近3時間以内の重要指標なし")
    except Exception:
        return blocked, "指標取得エラー"


# ==========================================
# 仮想トレードログ・厳密約定/決済判定・敗因学習
# ==========================================
def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def save_logs(logs):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def evaluate_past_trades(logs, m15_map):
    now = now_jst()
    for t in logs:
        if t.get("status") not in ["PENDING", "OPEN"]:
            continue
        sym = t.get("symbol", "XAU/USD")
        df = m15_map.get(sym)
        if df is None or df.empty:
            continue

        pip_unit = next((s["pip_unit"] for s in ALL_SYMBOLS if s["symbol"] == sym), 1.0)
        entry = float(t["entry"])
        tp = float(t["tp"])
        sl = float(t["sl"])
        direction = t["direction"]

        try:
            created_dt = datetime.datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        except Exception:
            created_dt = now

        if t["status"] == "PENDING" and (now - created_dt).total_seconds() > 6 * 3600:
            t["status"] = "EXPIRED"
            continue

        for _, bar in df.iloc[-4:].iterrows():
            hi, lo = bar["high"], bar["low"]
            if t["status"] == "PENDING":
                if lo <= entry <= hi:
                    t["status"] = "OPEN"

            if t["status"] == "OPEN":
                if direction == "BUY":
                    if lo <= sl:
                        t["status"] = "LOSS"
                        t["pnl"] = round((sl - entry) / pip_unit, 1)
                        break
                    elif hi >= tp:
                        t["status"] = "WIN"
                        t["pnl"] = round((tp - entry) / pip_unit, 1)
                        break
                elif direction == "SELL":
                    if hi >= sl:
                        t["status"] = "LOSS"
                        t["pnl"] = round((entry - sl) / pip_unit, 1)
                        break
                    elif lo <= tp:
                        t["status"] = "WIN"
                        t["pnl"] = round((entry - tp) / pip_unit, 1)
                        break
    return logs


def build_learning_feedback(logs):
    wins = [t for t in logs if t.get("status") == "WIN"]
    losses = [t for t in logs if t.get("status") == "LOSS"]
    expired = [t for t in logs if t.get("status") == "EXPIRED"]

    recent_failures = [
        f"・{t.get('symbol')} {t.get('direction')} @{t.get('entry')} → 結果:{t.get('status')} (根拠:{t.get('reason', '')[:45]})"
        for t in (losses + expired)[-5:]
    ]
    return (
        f"通算実績: {len(wins)}勝 {len(losses)}敗 / 未約定(EXPIRED): {len(expired)}件。\n"
        f"【過去の失敗・未約定パターン（同じミスを絶対に繰り返さないこと）】\n"
        + ("\n".join(recent_failures) if recent_failures else "まだ失敗履歴なし")
    )


# ==========================================
# LINE通知・週間レポート
# ==========================================
def send_line_message(text):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        return
    requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        },
        json={"to": LINE_USER_ID, "messages": [{"type": "text", "text": text[:4900]}]},
        timeout=15,
    )


def send_weekly_report(logs):
    wins = [t for t in logs if t.get("status") == "WIN"]
    losses = [t for t in logs if t.get("status") == "LOSS"]
    settled = len(wins) + len(losses)
    win_rate = (len(wins) / settled * 100) if settled > 0 else 0.0

    by_sym = {}
    for t in wins + losses:
        s = t.get("symbol", "XAU/USD")
        by_sym.setdefault(s, {"win": 0, "loss": 0, "pnl": 0.0})
        if t["status"] == "WIN":
            by_sym[s]["win"] += 1
        else:
            by_sym[s]["loss"] += 1
        by_sym[s]["pnl"] += float(t.get("pnl", 0))

    lines = [f"・{s}: {v['win']}勝{v['loss']}敗 ({v['pnl']:+.1f})" for s, v in by_sym.items()]
    report = (
        f"📅 週間精密AIトレード戦績レポート\n"
        f"・通算決済：{len(wins)}勝 {len(losses)}敗（勝率 {win_rate:.1f}%）\n\n"
        f"■ 銘柄別内訳\n" + ("\n".join(lines) if lines else "決済データなし")
    )
    send_line_message(report)


# ==========================================
# メイン処理（ゴールド最優先4時間足フル解析 ＋ サブ6銘柄厳選）
# ==========================================
def main():
    current_time = now_jst()
    logs = load_logs()

    if current_time.weekday() == 5 and current_time.hour == 12:
        send_weekly_report(logs)
        return

    print(f"=== {current_time.strftime('%Y-%m-%d %H:%M')} JST ゴールド最優先＆全7銘柄精密スキャン開始 ===")
    blocked_currencies, calendar_text = get_blocked_currencies_and_events()

    # 1. 全7銘柄の1時間足を1バッチで取得
    h1_map = fetch_batch_1h([s["symbol"] for s in ALL_SYMBOLS])
    time.sleep(10)

    # 2. 【最優先VIP枠】XAU/USD の 4H・15M・5M を無条件でフル取得（4重時間足解析）
    df_gold_4h = fetch_single_tf("XAU/USD", "4h", 50)
    time.sleep(10)
    df_gold_15m = fetch_single_tf("XAU/USD", "15min", 50)
    time.sleep(10)
    df_gold_5m = fetch_single_tf("XAU/USD", "5min", 50)
    time.sleep(10)

    m15_map = {}
    if df_gold_15m is not None:
        m15_map["XAU/USD"] = df_gold_15m

    # 3. サブ6銘柄からトレンド強度（ADX）上位2銘柄のみ抽出
    sub_candidates = []
    for meta in SUB_SYMBOLS:
        sym = meta["symbol"]
        if any(c in blocked_currencies for c in meta["currencies"]):
            continue
        df_1h = h1_map.get(sym)
        if df_1h is None or len(df_1h) < 50:
            continue
        ind_1h = analyze_dataframe(df_1h)
        if ind_1h["trend"] != "RANGE" and ind_1h["adx14"] >= 22.0:
            sub_candidates.append((ind_1h["adx14"], meta, ind_1h))

    sub_candidates.sort(key=lambda x: x[0], reverse=True)
    top_subs = sub_candidates[:2]

    # サブ上位2銘柄（＋保有中ポジション）の15Mと4Hを取得
    active_syms = {t.get("symbol") for t in logs if t.get("status") in ["PENDING", "OPEN"] and t.get("symbol") != "XAU/USD"}
    for sym in ({c[1]["symbol"] for c in top_subs} | active_syms):
        df_15 = fetch_single_tf(sym, "15min", 50)
        time.sleep(10)
        if df_15 is not None:
            m15_map[sym] = df_15

    sub_h4_map = {}
    for _, meta, _ in top_subs:
        sym = meta["symbol"]
        df_4 = fetch_single_tf(sym, "4h", 50)
        time.sleep(10)
        if df_4 is not None:
            sub_h4_map[sym] = analyze_dataframe(df_4)

    # 過去トレードの約定・決済採点
    logs = evaluate_past_trades(logs, m15_map)

    # 4. ゴールド最優先精密レポート構築
    gold_section = "【XAU/USD データ取得エラー】"
    if "XAU/USD" in h1_map and df_gold_4h is not None and df_gold_15m is not None and df_gold_5m is not None:
        g4 = analyze_dataframe(df_gold_4h, pa_bars=3)
        g1 = analyze_dataframe(h1_map["XAU/USD"], pa_bars=3)
        g15 = analyze_dataframe(df_gold_15m, pa_bars=4)
        g5 = analyze_dataframe(df_gold_5m, pa_bars=5)

        cur = g5["close"]
        atr15 = g15["atr14"]
        max_pullback = round(atr15 * 0.30, 2)
        min_sl_dist = round(atr15 * 1.30, 2)
        usd_blocked = "⚠️現在USD重要指標前後45分ロック中（原則見送り推奨）" if "USD" in blocked_currencies else "指標ロックなし"

        gold_section = f"""
★★★【最優先VIP銘柄：XAU/USD（ゴールド）4重時間足フル解析】★★★
・現在値: {cur:.2f} | 指標状態: {usd_blocked}
・許容指値幅（未約定防止）: 現在値から ±{max_pullback:.2f} ドル以内（これ以上離すな）
・最低損切り幅（ヒゲ狩り防止）: {min_sl_dist:.2f} ドル以上（1.3×ATR14基準）
・目標利確幅（損小利大）: 損切り幅の 2.2倍〜3.2倍以上

[4時間足(4H)] トレンド:{g4['trend']} | EMA20:{g4['ema20']:.2f} | EMA50:{g4['ema50']:.2f} | RSI14:{g4['rsi14']:.1f} | ADX14:{g4['adx14']:.1f} | 24本高値:{g4['swing_high']:.2f} / 安値:{g4['swing_low']:.2f}
[1時間足(1H)] トレンド:{g1['trend']} | EMA20:{g1['ema20']:.2f} | EMA50:{g1['ema50']:.2f} | RSI14:{g1['rsi14']:.1f} | ADX14:{g1['adx14']:.1f} | BB上限:{g1['bb_upper']:.2f} / 下限:{g1['bb_lower']:.2f}
[15分足(15M)] トレンド:{g15['trend']} | EMA20:{g15['ema20']:.2f} | RSI14:{g15['rsi14']:.1f} | ATR14:{atr15:.2f} | 24本高値:{g15['swing_high']:.2f} / 安値:{g15['swing_low']:.2f}
  15M直近プライスアクション:
    {g15['pa_text']}
[5分足(5M) エントリートリガー] トレンド:{g5['trend']} | EMA20:{g5['ema20']:.2f} | RSI14:{g5['rsi14']:.1f}
  5M直近5本プライスアクション:
    {g5['pa_text']}
"""

    # 5. サブ銘柄レポート構築（4Hと1H一致のみ）
    sub_sections = []
    for _, meta, ind_1h in top_subs:
        sym = meta["symbol"]
        ind_4h = sub_h4_map.get(sym)
        df_15 = m15_map.get(sym)
        if not ind_4h or df_15 is None:
            continue
        if ind_4h["trend"] != ind_1h["trend"]:
            continue
        ind_15m = analyze_dataframe(df_15, pa_bars=3)
        d = meta["digits"]
        atr = ind_15m["atr14"]
        sub_sections.append(
            f"【サブ候補：{sym}（{meta['name']}）】現在値: {ind_15m['close']:.{d}f} | 4H&1H:{ind_1h['trend']}一致 | ADX:{ind_1h['adx14']:.1f} | 許容指値幅:±{atr*0.3:.{d}f} | 最低SL幅:{atr*1.2:.{d}f}\n"
            f"  4H_EMA20:{ind_4h['ema20']:.{d}f}, RSI:{ind_4h['rsi14']:.1f} | 1H_EMA20:{ind_1h['ema20']:.{d}f}, RSI:{ind_1h['rsi14']:.1f} | 15M_RSI:{ind_15m['rsi14']:.1f}"
        )

    learning_text = build_learning_feedback(logs)

    prompt = f"""
現在時刻: {current_time.strftime('%Y-%m-%d %H:%M')} JST
{learning_text}

【直近の重要経済指標】
{calendar_text}

{gold_section}

【サブ監視銘柄（4H・1Hトレンド一致＆ADX22以上のみ）】
{chr(10).join(sub_sections) if sub_sections else "現在サブ銘柄に強いトレンド一致なし"}

あなたは機関投資家向けヘッジファンドの最高リスク管理責任者です。
**最優先任務は「XAU/USD（ゴールド）」の精密審査**です。ゴールドは4H・1H・15M・5Mの4時間足すべてを統合し、優位性が極めて高いチャンスがあれば必ず最優先で採用してください。サブ銘柄はゴールド以上に明確な高期待値チャンスがある場合のみ追加（合計最大2銘柄まで）してください。
どの銘柄もダマシやレンジで危険な場合は、無理に選ばず空配列 `[]` を返してください。

【絶対遵守ルール（約定率最大化＆損小利大）】
1. **未約定防止**：エントリー価格は必ず「現在値（成行）」または「現在値から許容指値幅（0.3×ATR）以内の浅い押し目・戻り目」に設定すること。
2. **ヒゲ狩り防止**：SLは必ず「最低損切り幅（ゴールドは1.3×ATR、他は1.2×ATR）以上」離し、直近15M・1Hの明確なサポレジの外側に置くこと。
3. **損小利大（RR 1:2.2以上）**：TPまでの値幅はSL幅の **必ず2.2倍〜3.2倍** に設定し、手前の4H/1Hスイング高値安値に阻まれないこと。
4. **ゴールドの5Mトリガー確認**：XAU/USDを採用する場合は、5分足と15分足の直近ローソク足（ヒゲ反発や実体ブレイク）でエントリー方向への初動が出ていることを確認すること。

JSON形式のみで出力してください：
{{
  "opportunities": [
    {{
      "symbol": "XAU/USD",
      "priority": "最優先VIP",
      "direction": "BUY",
      "order_type": "成行買い または Buy Limit",
      "entry": 2650.50,
      "tp": 2670.00,
      "sl": 2642.00,
      "rr": "1 : 2.3",
      "confidence": "特A（期待勝率82%）",
      "reason": "4H/1H/15Mすべて上昇トレンド一致。5MでEMA20反発の下ヒゲ陽線を確認し、1.3×ATR外側の15M安値下にSL配置。"
    }}
  ]
}}
"""

        Run python gold_mac_bot.py
  File "/home/runner/work/gold-ai-bot/gold-ai-bot/gold_mac_bot.py", line 481
    client = genai.Client(api_key=GEMINI_API_KEY)
IndentationError: unexpected indent
Error: Process completed with exit code 1.
