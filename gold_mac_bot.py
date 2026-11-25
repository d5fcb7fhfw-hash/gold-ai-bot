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

SYMBOLS = [
    {"symbol": "XAU/USD", "name": "ゴールド", "currencies": ["USD"], "pip_unit": 1.0, "digits": 2},
    {"symbol": "USD/JPY", "name": "ドル円", "currencies": ["USD", "JPY"], "pip_unit": 0.01, "digits": 3},
    {"symbol": "EUR/USD", "name": "ユーロドル", "currencies": ["EUR", "USD"], "pip_unit": 0.0001, "digits": 5},
    {"symbol": "GBP/JPY", "name": "ポンド円", "currencies": ["GBP", "JPY"], "pip_unit": 0.01, "digits": 3},
    {"symbol": "GBP/USD", "name": "ポンドドル", "currencies": ["GBP", "USD"], "pip_unit": 0.0001, "digits": 5},
    {"symbol": "EUR/JPY", "name": "ユーロ円", "currencies": ["EUR", "JPY"], "pip_unit": 0.01, "digits": 3},
    {"symbol": "AUD/USD", "name": "豪ドル米ドル", "currencies": ["AUD", "USD"], "pip_unit": 0.0001, "digits": 5},
]

JST = datetime.timezone(datetime.timedelta(hours=9))


def now_jst():
    return datetime.datetime.now(JST)


# ==========================================
# Twelve Data API 取得（バッチ＆個別リトライ対応）
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
# 高度なテクニカル指標・プライスアクション計算
# ==========================================
def analyze_dataframe(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi14 = (100 - (100 / (1 + rs))).iloc[-1]

    # ATR14
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]

    # ADX14（トレンド強度）
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_smooth = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_smooth.replace(0, 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_smooth.replace(0, 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx14 = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]

    # ボリンジャーバンド
    bb_mid = close.rolling(20).mean().iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # 直近スイング高値・安値（24本）
    swing_high = high.iloc[-24:].max()
    swing_low = low.iloc[-24:].min()

    # 直近3本のローソク足プライスアクション（ヒゲ比率・陽線陰線）
    pa_lines = []
    for i in range(-3, 0):
        o, h, l, c = open_p.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        candle_type = "陽線" if c >= o else "陰線"
        pa_lines.append(
            f"{df['datetime'].iloc[i][-5:]} {candle_type} (始:{o} 高:{h} 安:{l} 終:{c} | 上ヒゲ:{upper_wick:.4f} 下ヒゲ:{lower_wick:.4f} 実体:{body:.4f})"
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
# 経済指標リスクチェック（前後45分を厳格ロック）
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

        pip_unit = next((s["pip_unit"] for s in SYMBOLS if s["symbol"] == sym), 1.0)
        entry = float(t["entry"])
        tp = float(t["tp"])
        sl = float(t["sl"])
        direction = t["direction"]

        try:
            created_dt = datetime.datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M").replace(tzinfo=JST)
        except Exception:
            created_dt = now

        # 6時間経過して未約定ならEXPIRED（古い指値を引きずらない）
        if t["status"] == "PENDING" and (now - created_dt).total_seconds() > 6 * 3600:
            t["status"] = "EXPIRED"
            continue

        # 直近4本（1時間分）の15分足で判定
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
        f"・{t.get('symbol')} {t.get('direction')} @{t.get('entry')} → 結果:{t.get('status')} (根拠:{t.get('reason', '')[:40]})"
        for t in (losses + expired)[-5:]
    ]
    feedback = (
        f"通算実績: {len(wins)}勝 {len(losses)}敗 / 未約定(EXPIRED): {len(expired)}件。\n"
        f"【過去の失敗・未約定パターン（同じミスを絶対に繰り返さないこと）】\n"
        + ("\n".join(recent_failures) if recent_failures else "まだ失敗履歴なし")
    )
    return feedback


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
# メイン処理（2段階精密スクリーニング）
# ==========================================
def main():
    current_time = now_jst()
    logs = load_logs()

    if current_time.weekday() == 5 and current_time.hour == 12:
        send_weekly_report(logs)
        return

    print(f"=== {current_time.strftime('%Y-%m-%d %H:%M')} JST 精密2段階スキャン開始 ===")
    blocked_currencies, calendar_text = get_blocked_currencies_and_events()
    print("指標ロック中通貨:", blocked_currencies)

    # 第1段階：全7銘柄の1時間足をバッチ取得（わずか1クレジット消費）
    all_sym_names = [s["symbol"] for s in SYMBOLS]
    h1_map = fetch_batch_1h(all_sym_names)
    time.sleep(10)

    # トレンド強度（ADX）と指標リスクで有望な上位3銘柄に絞り込み
    candidates = []
    for meta in SYMBOLS:
        sym = meta["symbol"]
        if any(c in blocked_currencies for c in meta["currencies"]):
            print(f"[{sym}] 重要指標前後45分のため除外")
            continue
        df_1h = h1_map.get(sym)
        if df_1h is None or len(df_1h) < 50:
            continue
        ind_1h = analyze_dataframe(df_1h)
        # トレンドが明確かつADX >= 20の銘柄のみ候補化
        if ind_1h["trend"] != "RANGE" and ind_1h["adx14"] >= 20.0:
            candidates.append((ind_1h["adx14"], meta, ind_1h))

    # ADXが高い順に上位3銘柄を選出
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:3]

    # 第2段階：上位3銘柄（＋保有中ポジションがある銘柄）の4Hと15Mを精密取得
    active_syms = {t.get("symbol") for t in logs if t.get("status") in ["PENDING", "OPEN"]}
    target_syms_for_15m = {c[1]["symbol"] for c in top_candidates} | active_syms

    m15_map = {}
    h4_ind_map = {}
    m15_ind_map = {}

    for sym in target_syms_for_15m:
        df_15m = fetch_single_tf(sym, "15min", 50)
        time.sleep(10)
        if df_15m is not None:
            m15_map[sym] = df_15m
            m15_ind_map[sym] = analyze_dataframe(df_15m)

    for _, meta, _ in top_candidates:
        sym = meta["symbol"]
        df_4h = fetch_single_tf(sym, "4h", 50)
        time.sleep(10)
        if df_4h is not None:
            h4_ind_map[sym] = analyze_dataframe(df_4h)

    # 過去トレードの約定・決済採点
    logs = evaluate_past_trades(logs, m15_map)

    # 4Hと1Hの方向が完全一致する銘柄だけを最終プロンプトに投入
    final_reports = []
    for _, meta, ind_1h in top_candidates:
        sym = meta["symbol"]
        ind_4h = h4_ind_map.get(sym)
        ind_15m = m15_ind_map.get(sym)
        if not ind_4h or not ind_15m:
            continue

        # 4Hと1Hのトレンド不一致（ダマシ）を厳格除外
        if ind_4h["trend"] != ind_1h["trend"]:
            print(f"[{sym}] 4H({ind_4h['trend']})と1H({ind_1h['trend']})のトレンド不一致のため除外")
            continue

        d = meta["digits"]
        atr = ind_15m["atr14"]
        cur = ind_15m["close"]
        max_pullback = round(atr * 0.35, d)
        min_sl_dist = round(atr * 1.2, d)

        report = (
            f"【{sym}（{meta['name']}）】現在値: {cur:.{d}f} | 方向: {ind_1h['trend']}トレンド一致\n"
            f"  ・許容指値幅（現在値から最大）: ±{max_pullback:.{d}f} 以内（これ以上離すと未約定になるため厳禁）\n"
            f"  ・最低損切り幅（ノイズ回避）: {min_sl_dist:.{d}f} 以上\n"
            f"  ・4時間足(4H): EMA20={ind_4h['ema20']:.{d}f}, EMA50={ind_4h['ema50']:.{d}f}, RSI14={ind_4h['rsi14']:.1f}, 高値={ind_4h['swing_high']:.{d}f}, 安値={ind_4h['swing_low']:.{d}f}\n"
            f"  ・1時間足(1H): EMA20={ind_1h['ema20']:.{d}f}, ADX14={ind_1h['adx14']:.1f}, RSI14={ind_1h['rsi14']:.1f}, BB上限={ind_1h['bb_upper']:.{d}f}, BB下限={ind_1h['bb_lower']:.{d}f}\n"
            f"  ・15分足(15M): EMA20={ind_15m['ema20']:.{d}f}, RSI14={ind_15m['rsi14']:.1f}, ATR14={atr:.{d}f}\n"
            f"  ・15M直近プライスアクション:\n    {ind_15m['pa_text']}"
        )
        final_reports.append(report)

    if not final_reports:
        print("全条件（4H/1H一致・ADX20以上・指標クリア）を満たす銘柄なし。通知スキップ。")
        save_logs(logs)
        return

    learning_text = build_learning_feedback(logs)

    prompt = f"""
現在時刻: {current_time.strftime('%Y-%m-%d %H:%M')} JST
{learning_text}

【直近の重要経済指標】
{calendar_text}

【厳格フィルタを通過した最有望銘柄データ（4H・1Hトレンド完全一致）】
{chr(10).join(final_reports)}

あなたは機関投資家向けヘッジファンドの最高リスク管理責任者兼チーフストラテジストです。
以下の絶対ルールを守り、**本当に勝率とリスクリワードが極めて高い最高峰のチャンス**だけを最大1〜2銘柄厳選してください。条件を少しでも満たさない場合は空配列 `[]` を返してください。

【絶対遵守ルール（約定率最大化＆損小利大）】
1. **未約定防止**：エントリー価格は必ず「現在値」または「現在値から指定された許容指値幅（±0.35×ATR）以内の浅い押し目・戻り目」に設定すること。遠くの指値は絶対禁止。
2. **ヒゲ狩りロスカット防止**：損切り（SL）は必ず「最低損切り幅（1.2×ATR）以上」離し、かつ直近15M・1Hの明確なサポレジラインの外側に置くこと。
3. **損小利大（RR 1:2.2以上）**：利確（TP）までの値幅は、損切り（SL）までの値幅の **必ず2.2倍〜3.0倍** に設定し、手前の4H/1Hスイング高値安値に阻まれないことを確認すること。
4. **プライスアクション確認**：15M直近3本のローソク足で、エントリー方向への反発（下ヒゲ陽線・上ヒゲ陰線・実体確定）が出ていること。

JSON形式のみで出力してください：
{{
  "opportunities": [
    {{
      "symbol": "XAU/USD",
      "direction": "BUY",
      "order_type": "成行買い または Buy Limit",
      "entry": 2650.50,
      "tp": 2668.00,
      "sl": 2642.50,
      "rr": "1 : 2.2",
      "confidence": "特A（期待勝率80%超）",
      "reason": "4H/1H上昇トレンド一致＋ADX強気。15Mで下ヒゲ陽線反発を確認し、ATR基準の損切り外側にSLを配置。"
    }}
  ]
}}
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    try:
        result = json.loads(response.text)
        opps = result.get("opportunities", [])
    except Exception as e:
        print("JSONパースエラー:", e)
        save_logs(logs)
        return

    if not opps:
        print("AI最終審査で基準未達のため見送り（通知スキップ）")
        save_logs(logs)
        return

    msg_blocks = [f"💎 特選高期待値シグナル（精密MTF審査通過）\n（{current_time.strftime('%m/%d %H:%M')} JST）\n"]
    for op in opps:
        sym = op["symbol"]
        dir_jp = "買い（BUY）" if op["direction"] == "BUY" else "売り（SELL）"
        block = (
            f"━━━━━━━━━━━━━━━━\n"
            f"🎯 【{sym}】 {dir_jp}｜{op.get('confidence', '高')}\n"
            f"・注文方法：{op.get('order_type', op['direction'])}\n"
            f"・エントリー：{op['entry']}\n"
            f"・利確（TP）：{op['tp']}\n"
            f"・損切り（SL）：{op['sl']}\n"
            f"・RR比：{op.get('rr', '1 : 2.2')}\n"
            f"・精密根拠：{op['reason']}"
        )
        msg_blocks.append(block)

        logs.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M"),
            "symbol": sym,
            "direction": op["direction"],
            "entry": float(op["entry"]),
            "tp": float(op["tp"]),
            "sl": float(op["sl"]),
            "reason": op.get("reason", ""),
            "status": "PENDING",
            "pnl": 0.0,
        })

    send_line_message("\n".join(msg_blocks))
    save_logs(logs)
    print(f"特選シグナル {len(opps)}件 を送信・記録しました。")


if __name__ == "__main__":
    main()
