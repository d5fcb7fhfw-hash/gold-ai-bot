import os
import json
import time
import math
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from google import genai

# ==============================================================================
# 環境変数（GitHub Secrets に設定）
# ==============================================================================
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_CANDIDATES = [
    "gemini-omni-flash-preview",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]

JST = datetime.timezone(datetime.timedelta(hours=9))
NOW_JST = datetime.datetime.now(JST)
NOW_UTC = datetime.datetime.now(datetime.timezone.utc)

GOLD_SYMBOL = "XAU/USD"
SUB_PAIRS = ["USD/JPY", "EUR/USD", "GBP/JPY", "GBP/USD", "EUR/JPY", "AUD/USD"]

TRADE_LOG_FILE = "trade_log.json"
EVOLUTION_FILE = "ai_evolution.json"


# ==============================================================================
# スプレッド誤差補正ヘルパー（XMTrading / MT5 実戦基準）
# ==============================================================================
def get_spread_buffer(symbol):
    if "XAU" in symbol:
        return 0.35
    if "JPY" in symbol:
        return 0.03
    return 0.0003


# ==============================================================================
# セッション（市場時間帯）判定
# ==============================================================================
def get_market_session_info(now_jst):
    hr = now_jst.hour
    mn = now_jst.minute
    t_val = hr + mn / 60.0

    if 5.5 <= t_val < 7.5:
        return {
            "session": "早朝ロールオーバー・スプレッド極大拡大帯（05:30〜07:30 JST）",
            "entry_allowed": False,
            "advice": "XMTrading等でスプレッドが異常拡大しヒゲ狩りが多発するため新規エントリー厳禁。保有中はSL拡大に警戒。",
        }
    elif 7.5 <= t_val < 15.5:
        return {
            "session": "東京・アジア時間（07:30〜15:30 JST）",
            "entry_allowed": True,
            "advice": "実需中心でレンジになりやすい時間帯。高値・安値への飛び乗りブレイクは避け、サポレジ引きつけの浅い指値を優先。",
        }
    elif 15.5 <= t_val < 21.0:
        return {
            "session": "欧州・ロンドン市場時間（15:30〜21:00 JST）",
            "entry_allowed": True,
            "advice": "大口機関投資家が参入し東京時間の高値・安値を一度狩ってから本命トレンドが出やすい時間帯。流動性スイープ後の押し目・戻り目を狙う。",
        }
    elif 21.0 <= t_val or t_val < 2.0:
        return {
            "session": "NY市場・ロンドン重複ゴールデンタイム（21:00〜02:00 JST）",
            "entry_allowed": True,
            "advice": "1日で最も出来高と値幅が出る時間帯。順張りトレンドフォローが最も機能しやすいが、指標発表前後は急変に注意。",
        }
    else:
        return {
            "session": "NY後半・流動性低下帯（02:00〜05:30 JST）",
            "entry_allowed": True,
            "advice": "出来高が細り値動きが鈍化しやすい時間帯。新規は厳選し、保有ポジションは早朝スプレッド拡大前の利確・建値保護を検討。",
        }


# ==============================================================================
# Gemini 呼び出し（利用可能モデル自動探索・429レート制限時の自動切替付き）
# ==============================================================================
def get_available_models():
    candidates = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if (
                ("flash" in name or "pro" in name)
                and "image" not in name
                and "tts" not in name
                and "audio" not in name
                and "embedding" not in name
            ):
                candidates.append(name)
    except Exception:
        pass
    ordered = [m for m in MODEL_CANDIDATES if m in candidates]
    for m in candidates:
        if m not in ordered:
            ordered.append(m)
    return ordered if ordered else MODEL_CANDIDATES


def generate_with_retry(prompt, retries=2):
    for model_name in get_available_models():
        for attempt in range(retries):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return resp.text.strip()
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "NOT_FOUND" in err_str:
                    print(f"[Gemini Skip] {model_name} は未対応のためスキップします...")
                    break
                elif "PerDay" in err_str:
                    print(f"[Gemini DailyLimit] {model_name} の1日上限到達。次のモデルへ切り替えます...")
                    break
                elif "429" in err_str and attempt < retries - 1:
                    print(f"[Gemini RateLimit] 1分上限のため65秒待機して再試行します... ({attempt + 1}/{retries})")
                    time.sleep(65)
                else:
                    break
    raise RuntimeError("利用可能なすべてのGeminiモデルの無料枠上限に達しました。")


# ==============================================================================
# 1. JSON データ読み書き
# ==============================================================================
def load_json(path, default_data):
    if not os.path.exists(path):
        save_json(path, default_data)
        return default_data
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 2. テクニカル指標・4H/1H/15M/5M・壁（サポレジ・節目）計算
# ==============================================================================
def calc_ema(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def calc_rsi(closes, period=14):
    if len(closes) <= period:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(candles, period=14):
    if len(candles) <= period:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = float(candles[i]["high"])
        l = float(candles[i]["low"])
        pc = float(candles[i - 1]["close"])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def fetch_candles(symbol, interval, outputsize=60):
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={urllib.parse.quote(symbol)}&interval={interval}"
        f"&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    )
    for attempt in range(2):
        try:
            req = urllib.request.urlopen(url, timeout=15)
            data = json.loads(req.read().decode("utf-8"))
            if "values" in data:
                return list(reversed(data["values"]))
            if data.get("code") == 429 and attempt == 0:
                print(f"[RateLimit] {symbol} {interval} - 62秒待機します...")
                time.sleep(62)
        except Exception as e:
            print(f"[Error] fetch_candles {symbol} {interval}: {e}")
    return []


def analyze_symbol_market(symbol, is_vip=False):
    c_4h = fetch_candles(symbol, "4h", 40) if is_vip else []
    if is_vip:
        time.sleep(8)
    c_1h = fetch_candles(symbol, "1h", 60)
    time.sleep(8)
    c_15m = fetch_candles(symbol, "15min", 60)
    time.sleep(8)
    c_5m = fetch_candles(symbol, "5min", 40) if is_vip else []
    if is_vip:
        time.sleep(8)

    if not c_1h or not c_15m:
        return {"symbol": symbol, "error": "データ取得失敗"}

    closes_1h = [float(x["close"]) for x in c_1h]
    highs_1h = [float(x["high"]) for x in c_1h]
    lows_1h = [float(x["low"]) for x in c_1h]

    closes_15m = [float(x["close"]) for x in c_15m]
    highs_15m = [float(x["high"]) for x in c_15m]
    lows_15m = [float(x["low"]) for x in c_15m]
    current_price = closes_15m[-1]

    recent_2h_high = max(highs_15m[-8:]) if len(highs_15m) >= 8 else max(highs_15m)
    recent_2h_low = min(lows_15m[-8:]) if len(lows_15m) >= 8 else min(lows_15m)

    recent_24h_high = max(highs_1h[-24:]) if len(highs_1h) >= 24 else max(highs_1h)
    recent_24h_low = min(lows_1h[-24:]) if len(lows_1h) >= 24 else min(lows_1h)
    swing_6h_high = max(highs_1h[-6:])
    swing_6h_low = min(lows_1h[-6:])

    step = 10.0 if "XAU" in symbol else (0.5 if "JPY" in symbol else 0.005)
    upper_round = math.ceil(current_price / step) * step
    lower_round = math.floor(current_price / step) * step

    ema20_1h = calc_ema(closes_1h, 20)
    ema50_1h = calc_ema(closes_1h, 50)
    rsi_1h = calc_rsi(closes_1h, 14)
    atr_1h = calc_atr(c_1h, 14)
    change_1h_pct = ((closes_1h[-1] - closes_1h[-2]) / closes_1h[-2] * 100) if len(closes_1h) >= 2 else 0.0

    ema20_15m = calc_ema(closes_15m, 20)
    rsi_15m = calc_rsi(closes_15m, 14)
    atr_15m = calc_atr(c_15m, 14)

    result = {
        "symbol": symbol,
        "current_price": round(current_price, 4),
        "spread_buffer": get_spread_buffer(symbol),
        "recent_2h_high": round(recent_2h_high, 4),
        "recent_2h_low": round(recent_2h_low, 4),
        "change_1h_percent": round(change_1h_pct, 3),
        "1H": {
            "trend": "上昇" if ema20_1h > ema50_1h else "下降",
            "ema20": round(ema20_1h, 4),
            "ema50": round(ema50_1h, 4),
            "rsi": round(rsi_1h, 1),
            "atr": round(atr_1h, 4),
        },
        "15M": {
            "ema20": round(ema20_15m, 4),
            "rsi": round(rsi_15m, 1),
            "atr": round(atr_15m, 4),
            "last_3_candles": [
                {"t": x["datetime"], "o": x["open"], "h": x["high"], "l": x["low"], "c": x["close"]}
                for x in c_15m[-3:]
            ],
        },
        "walls_and_levels": {
            "24h_high": round(recent_24h_high, 4),
            "24h_low": round(recent_24h_low, 4),
            "6h_high": round(swing_6h_high, 4),
            "6h_low": round(swing_6h_low, 4),
            "upper_round_number": round(upper_round, 4),
            "lower_round_number": round(lower_round, 4),
            "dist_to_24h_high": round(recent_24h_high - current_price, 4),
            "dist_to_24h_low": round(current_price - recent_24h_low, 4),
        },
    }

    if c_4h:
        closes_4h = [float(x["close"]) for x in c_4h]
        highs_4h = [float(x["high"]) for x in c_4h]
        lows_4h = [float(x["low"]) for x in c_4h]
        ema20_4h = calc_ema(closes_4h, 20)
        ema50_4h = calc_ema(closes_4h, 50)
        result["4H"] = {
            "trend": "上昇" if ema20_4h > ema50_4h else "下降",
            "ema20": round(ema20_4h, 4),
            "ema50": round(ema50_4h, 4),
            "rsi": round(calc_rsi(closes_4h, 14), 1),
            "atr": round(calc_atr(c_4h, 14), 4),
            "4h_swing_high": round(max(highs_4h[-12:]), 4),
            "4h_swing_low": round(min(lows_4h[-12:]), 4),
        }

    if c_5m:
        closes_5m = [float(x["close"]) for x in c_5m]
        result["5M"] = {
            "ema20": round(calc_ema(closes_5m, 20), 4),
            "rsi": round(calc_rsi(closes_5m, 14), 1),
            "last_4_candles": [
                {"t": x["datetime"], "o": x["open"], "h": x["high"], "l": x["low"], "c": x["close"]}
                for x in c_5m[-4:]
            ],
        }
    return result


# ==============================================================================
# 3. 米ドル強弱（DXY代替相関）計算
# ==============================================================================
def calc_usd_strength_index(fx_data_list):
    usdjpy_chg = 0.0
    eurusd_chg = 0.0
    for item in fx_data_list:
        if item.get("symbol") == "USD/JPY":
            usdjpy_chg = item.get("change_1h_percent", 0.0)
        elif item.get("symbol") == "EUR/USD":
            eurusd_chg = item.get("change_1h_percent", 0.0)

    usd_score = round((usdjpy_chg - eurusd_chg) / 2.0, 3)
    if usd_score >= 0.12:
        bias = "明確な米ドル全面高（GOLDには強い逆風・売り優位）"
    elif usd_score <= -0.12:
        bias = "明確な米ドル全面安（GOLDには強い追い風・買い優位）"
    else:
        bias = "米ドル方向感中立（レンジ・個別材料主導）"

    return {
        "usd_strength_score_1h": usd_score,
        "usdjpy_1h_change_pct": usdjpy_chg,
        "eurusd_1h_change_pct": eurusd_chg,
        "gold_impact_assessment": bias,
    }


# ==============================================================================
# 4. 最新ニュース・経済指標取得 ＆ 指標30分前〜15分後ロック判定
# ==============================================================================
def fetch_google_news(queries, limit_per_query=4):
    headlines = []
    for q in queries:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ja&gl=JP&ceid=JP:ja"
        try:
            req = urllib.request.urlopen(url, timeout=10)
            xml_data = req.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")[:limit_per_query]
            for item in items:
                title = item.findtext("title", "")
                pub_date = item.findtext("pubDate", "")
                headlines.append(f"- {title} ({pub_date})")
        except Exception as e:
            print(f"[News Error] {q}: {e}")
    return "\n".join(headlines) if headlines else "最新ニュース取得なし"


def fetch_economic_calendar_and_lock():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    lock_active = False
    lock_reason = "直近30分以内に重要指標の発表はありません（通常稼働OK）。"
    high_events = []

    try:
        req = urllib.request.urlopen(url, timeout=10)
        events = json.loads(req.read().decode("utf-8"))
        for ev in events:
            impact = ev.get("impact", "")
            country = ev.get("country", "")
            date_str = ev.get("date", "")
            if impact in ["High", "Medium"] and country in ["USD", "JPY", "EUR", "GBP", "AUD"]:
                high_events.append(
                    f"- [{country}] {ev.get('title')} | 日時(UTC): {date_str} | 重要度: {impact} | 予想: {ev.get('forecast')} / 前回: {ev.get('previous')}"
                )
            if impact == "High" and country in ["USD", "JPY", "EUR"] and date_str:
                try:
                    ev_dt = datetime.datetime.fromisoformat(date_str)
                    diff_min = (ev_dt - NOW_UTC).total_seconds() / 60.0
                    if -15.0 <= diff_min <= 35.0:
                        lock_active = True
                        lock_reason = f"⚠️【重要指標ロック発動中】[{country}] {ev.get('title')} が約{int(diff_min)}分後に発表（または直後）のため、新規エントリーを強制禁止します。保有ポジションは建値保護・縮小を推奨。"
                except Exception:
                    pass
    except Exception as e:
        return f"指標カレンダー取得エラー: {e}", False, lock_reason

    cal_text = "\n".join(high_events[:20]) if high_events else "直近の主要指標情報なし"
    return cal_text, lock_active, lock_reason


# ==============================================================================
# 5. 市場オープン・土日判定
# ==============================================================================
def is_weekend_closed(now_jst):
    wd = now_jst.weekday()
    hr = now_jst.hour
    if wd == 5 and hr >= 7:
        return True
    if wd == 6:
        return True
    if wd == 0 and hr < 7:
        return True
    return False


# ==============================================================================
# 6. 指値約定チェック・建値保護アラート・スプレッド補正決着＆自己進化
# ==============================================================================
def check_positions_and_evolve(trade_log, evolution_data, market_snapshots, news_summary):
    open_positions = trade_log.get("open_positions", [])
    closed_history = trade_log.get("closed_history", [])
    still_active = []
    newly_closed = []
    status_notices = []

    for pos in open_positions:
        sym = pos["symbol"]
        snap = market_snapshots.get(sym)
        if not snap or "current_price" not in snap:
            still_active.append(pos)
            continue

        cur = float(snap["current_price"])
        high_2h = float(snap.get("recent_2h_high", cur))
        low_2h = float(snap.get("recent_2h_low", cur))
        spread = get_spread_buffer(sym)

        direction = pos["direction"]
        tp = float(pos["tp"])
        sl = float(pos["sl"])
        entry = float(pos["entry_price"])
        status = pos.get("status", "OPEN")

        if status == "PENDING":
            filled = False
            if direction == "BUY" and (low_2h <= (entry - spread * 0.5) or cur <= (entry - spread * 0.5)):
                filled = True
            elif direction == "SELL" and (high_2h >= (entry + spread * 0.5) or cur >= (entry + spread * 0.5)):
                filled = True

            if filled:
                pos["status"] = "OPEN"
                pos["filled_at"] = NOW_JST.strftime("%Y-%m-%d %H:%M JST")
                status = "OPEN"
                status_notices.append(f"✅【指値約定】{sym} ({direction}) が指値 {entry} に到達しポジション保有を開始しました。")
            else:
                try:
                    opened_dt = datetime.datetime.strptime(pos["opened_at"].replace(" JST", ""), "%Y-%m-%d %H:%M").replace(tzinfo=JST)
                    elapsed_min = (NOW_JST - opened_dt).total_seconds() / 60.0
                except Exception:
                    elapsed_min = 0
                if elapsed_min > 70:
                    status_notices.append(f"⏳【鮮度切れキャンセル】{sym} ({direction}) 指値 {entry} は60分経過のため未約定キャンセルしました。")
                    continue
                pos["current_price"] = round(cur, 4)
                still_active.append(pos)
                continue

        result = None
        close_price = cur

        if direction == "BUY":
            if (high_2h - spread) >= tp or (cur - spread) >= tp:
                result = "WIN"
                close_price = tp
            elif low_2h <= sl or cur <= sl:
                result = "LOSS"
                close_price = sl
        elif direction == "SELL":
            if (low_2h + spread) <= tp or (cur + spread) <= tp:
                result = "WIN"
                close_price = tp
            elif (high_2h + spread) >= sl or (cur + spread) >= sl:
                result = "LOSS"
                close_price = sl

        if result:
            pnl = (close_price - entry) if direction == "BUY" else (entry - close_price)
            pos["status"] = "CLOSED"
            pos["result"] = result
            pos["close_price"] = round(close_price, 4)
            pos["pnl"] = round(pnl, 4)
            pos["closed_at"] = NOW_JST.strftime("%Y-%m-%d %H:%M JST")
            newly_closed.append(pos)
            closed_history.append(pos)
        else:
            unrealized = (cur - entry) if direction == "BUY" else (entry - cur)
            target_dist = abs(tp - entry)
            progress_pct = round((unrealized / target_dist) * 100, 1) if target_dist > 0 else 0.0

            be_advice = "通常ホールド継続"
            if progress_pct >= 50.0 and not pos.get("breakeven_triggered", False):
                pos["breakeven_triggered"] = True
                pos["sl"] = round(entry, 4)
                be_advice = f"🛡️【建値移動・半分利確推奨】TPまで{progress_pct}%到達！SLを建値({entry})へ移動し負けゼロを確定させてください。"
                status_notices.append(f"🛡️【建値移動発動】{sym} ({direction}) が目標の{progress_pct}%まで伸びたため、SLを建値({entry})へ引き上げました。")
            elif pos.get("breakeven_triggered", False):
                be_advice = f"🛡️ 建値保護済み（SL={entry}で負けゼロ確定・残りTP狙い）"

            pos["current_price"] = round(cur, 4)
            pos["unrealized_pnl"] = round(unrealized, 4)
            pos["tp_progress_percent"] = progress_pct
            pos["dist_to_tp"] = round(abs(tp - cur), 4)
            pos["dist_to_sl"] = round(abs(cur - float(pos["sl"])), 4)
            pos["management_advice"] = be_advice
            still_active.append(pos)

    trade_log["open_positions"] = still_active
    trade_log["closed_history"] = closed_history

    consecutive_losses = 0
    for item in reversed(closed_history):
        if item.get("result") == "LOSS":
            consecutive_losses += 1
        else:
            break
    evolution_data["consecutive_losses"] = consecutive_losses
    evolution_data["strict_mode"] = True if consecutive_losses >= 2 else False

    evolution_reports = list(status_notices)
    for closed_pos in newly_closed:
        evo_prompt = f"""
あなたは自己進化型AIトレーディングシステムの「戦績解剖・ルール改善エンジン」です。
たった今、以下の仮想トレードが取引終了（決着）となりました。結果を冷徹に解剖し「進化ルール」を更新してください。

- 銘柄：{closed_pos['symbol']} / 方向：{closed_pos['direction']} / 時間軸：{closed_pos.get('timeframe_type', '')}
- エントリー → 決着：{closed_pos['entry_price']} → {closed_pos['close_price']} / 結果：{closed_pos['result']} ({closed_pos['pnl']:+})
- 根拠メモ：{closed_pos.get('reason', '')}
- 直近ニュース：{news_summary}
- 現在の進化ルール帳：{json.dumps(evolution_data, ensure_ascii=False)}

【敗因5分類タグ】TAG_1(上位足逆行・壁衝突) / TAG_2(指標・急変・ドル急変) / TAG_3(指値刺さり遅れ) / TAG_4(SL狭すぎ・ヒゲ狩り) / TAG_5(レンジダマシ)

必ず純粋なJSONのみ出力してください：
{{
  "win_or_loss": "{closed_pos['result']}",
  "primary_failure_tag": "LOSSの場合は TAG_1〜TAG_5 のいずれか（WINの場合は NONE）",
  "analysis_summary": "なぜ勝てたか／なぜ負けたかの詳しい解剖コメント",
  "new_or_updated_rule": "次回以降の判断プロンプトへ反映する具体的な改善ルール",
  "gold_short_rules": ["更新後のGOLD短期ルールリスト（最大6個）"],
  "gold_long_rules": ["更新後のGOLD超厳選長期ルールリスト（最大4個）"],
  "fx_pairs_rules": ["更新後のサブ6通貨ペア共通ルールリスト（最大6個）"]
}}
"""
        try:
            raw = generate_with_retry(evo_prompt).replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            tag = parsed.get("primary_failure_tag", "NONE")
            if tag in evolution_data.get("tag_counts", {}):
                evolution_data["tag_counts"][tag] += 1

            evolution_data["gold_short_rules"] = parsed.get("gold_short_rules", evolution_data.get("gold_short_rules", []))
            evolution_data["gold_long_rules"] = parsed.get("gold_long_rules", evolution_data.get("gold_long_rules", []))
            evolution_data["fx_pairs_rules"] = parsed.get("fx_pairs_rules", evolution_data.get("fx_pairs_rules", []))
            evolution_data["last_evolution_note"] = parsed.get("new_or_updated_rule", "")

            closed_pos["failure_tag"] = tag
            closed_pos["post_analysis"] = parsed.get("analysis_summary", "")

            evolution_reports.append(
                f"📌【決着＆自己進化】{closed_pos['symbol']} ({closed_pos['direction']}) → {closed_pos['result']} ({closed_pos['pnl']:+})\n"
                f"・分類タグ: {tag}\n"
                f"・解剖: {parsed.get('analysis_summary')}\n"
                f"・新ルール反映: {parsed.get('new_or_updated_rule')}"
            )
        except Exception as e:
            print(f"[Evolution Parse Error] {e}")

    save_json(TRADE_LOG_FILE, trade_log)
    save_json(EVOLUTION_FILE, evolution_data)
    return evolution_reports


# ==============================================================================
# 7. LINE プッシュ通知送信
# ==============================================================================
def send_line_message(text):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("[LINE Skip] トークンまたはUSER_ID未設定\n", text)
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
    for chunk in chunks:
        payload = json.dumps({
            "to": LINE_USER_ID,
            "messages": [{"type": "text", "text": chunk}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            urllib.request.urlopen(req, timeout=15)
        except Exception as e:
            print(f"[LINE Push Error] {e}")


# ==============================================================================
# 8. 土日専用：週末集中学習＆来週の作戦レポート実行
# ==============================================================================
def run_weekend_study_mode(trade_log, evolution_data):
    hr = NOW_JST.hour
    wd = NOW_JST.weekday()
    if hr % 4 != 0 and not (wd == 6 and hr == 22) and not (wd == 5 and hr == 10):
        print(f"[Weekend] {NOW_JST.strftime('%H:%M')} は週末待機スロットのためスキップ")
        return

    analyst_news = fetch_google_news(
        ["ドル円 来週 見通し FX", "ゴールド 金相場 来週 予想", "FRB 金利 見通し 週間", "XAUUSD weekly outlook"],
        limit_per_query=4,
    )
    calendar_data, _, _ = fetch_economic_calendar_and_lock()

    weekend_prompt = f"""
あなたは週末集中学習モードで稼働中の「自己進化型AIトレーディング研究官」です。
土日休場時間を使って「①過去の全トレードデータの徹底振り返り」と「②外部のトレーダー・アナリスト見解および来週の市場テーマの学習」を行い、月曜朝7時の市場オープンに向けた知識と戦略をアップデートしてください。

- 学習実行日時（日本時間）：{NOW_JST.strftime('%Y-%m-%d %H:%M JST')}
- 累計の全件約定トレード履歴：{json.dumps(trade_log.get('closed_history', [])[-30:], ensure_ascii=False)}
- 5分類敗因タグの累計データと現在の進化ルール帳：{json.dumps(evolution_data, ensure_ascii=False)}
- 土日に収集した週間見通しニュース：
{analyst_news}
- 来週の重要経済指標カレンダー：
{calendar_data}
- 金曜クローズ時点の持ち越しポジション：{json.dumps(trade_log.get('open_positions', []), ensure_ascii=False)}

最初に必ず `===EVOLUTION_JSON_START===` と `===EVOLUTION_JSON_END===` で囲んで更新後の進化ルールJSONを出力し、その後にLINE送信用の「📚 週末の集中学習＆来週の作戦レポート」を詳しく書いてください。

===EVOLUTION_JSON_START===
{{
  "weekend_market_outlook": "来週の相場テーマ・トレーダー意識価格帯の要約",
  "gold_short_rules": ["更新後のGOLD短期ルールリスト"],
  "gold_long_rules": ["更新後のGOLD超厳選長期ルールリスト"],
  "fx_pairs_rules": ["更新後のサブ6通貨ペア共通ルールリスト"]
}}
===EVOLUTION_JSON_END===

### 📚 週末の集中学習＆来週の作戦レポート
（1.今週の振り返り＆新ルール / 2.外部トレーダー見解＆市場テーマ / 3.GOLD来週の狙い方 / 4.サブ6通貨ペア注目銘柄 を詳しく記載）
"""
    full_text = generate_with_retry(weekend_prompt)

    report_text = full_text
    if "===EVOLUTION_JSON_START===" in full_text and "===EVOLUTION_JSON_END===" in full_text:
        json_part = full_text.split("===EVOLUTION_JSON_START===")[1].split("===EVOLUTION_JSON_END===")[0].strip()
        report_text = full_text.split("===EVOLUTION_JSON_END===")[1].strip()
        try:
            parsed_evo = json.loads(json_part)
            evolution_data["weekend_market_outlook"] = parsed_evo.get("weekend_market_outlook", "")
            evolution_data["gold_short_rules"] = parsed_evo.get("gold_short_rules", evolution_data.get("gold_short_rules", []))
            evolution_data["gold_long_rules"] = parsed_evo.get("gold_long_rules", evolution_data.get("gold_long_rules", []))
            evolution_data["fx_pairs_rules"] = parsed_evo.get("fx_pairs_rules", evolution_data.get("fx_pairs_rules", []))
            evolution_data["last_weekend_study"] = NOW_JST.strftime("%Y-%m-%d %H:%M JST")
            save_json(EVOLUTION_FILE, evolution_data)
        except Exception as e:
            print(f"[Weekend JSON Parse Error] {e}")

    if (wd == 6 and hr in [20, 21, 22]) or (wd == 5 and hr in [8, 9, 10]):
        send_line_message(report_text)
    else:
        print("[Weekend] 知識アップデート完了（LINE配信時間外のため保存のみ）")


# ==============================================================================
# 9. 平日メイン分析（LINEで提示した新規指値・エントリーをすべて新規取引登録）
# ==============================================================================
def run_weekday_analysis(trade_log, evolution_data):
    gold_data = analyze_symbol_market(GOLD_SYMBOL, is_vip=True)
    fx_data_list = [analyze_symbol_market(sym, is_vip=False) for sym in SUB_PAIRS]

    market_snapshots = {}
    if "current_price" in gold_data:
        market_snapshots[GOLD_SYMBOL] = gold_data
    for item in fx_data_list:
        if "current_price" in item:
            market_snapshots[item["symbol"]] = item

    usd_strength = calc_usd_strength_index(fx_data_list)
    session_info = get_market_session_info(NOW_JST)

    news_headlines = fetch_google_news(
        ["ゴールド 金価格 相場", "ドル円 為替 FRB", "XAUUSD gold price news", "米国債 金利 地政学"],
        limit_per_query=3,
    )
    calendar_data, indicator_lock, indicator_lock_reason = fetch_economic_calendar_and_lock()

    evo_reports = check_positions_and_evolve(trade_log, evolution_data, market_snapshots, news_headlines)

    closed = trade_log.get("closed_history", [])
    wins = sum(1 for x in closed if x.get("result") == "WIN")
    losses = sum(1 for x in closed if x.get("result") == "LOSS")
    win_rate = round((wins / len(closed)) * 100, 1) if closed else 0.0

    stats_block = {
        "total_trades": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate_percent": win_rate,
        "consecutive_losses": evolution_data.get("consecutive_losses", 0),
        "strict_mode_active": evolution_data.get("strict_mode", False),
        "evolution_rules": evolution_data,
    }

    main_prompt = f"""
あなたはプロ機関投資家レベルの専属FX・ゴールド（XAU/USD）ディーラー兼AIアナリストです。
最優先目標は「トータル利益の最大化（高いリスクリワード比と期待値プラスの徹底）」です。表面的な一言コメントは厳禁とし、4H/1H/15M/5Mチャート構造・米ドル強弱スコア・市場時間帯・大口投資家動向・価格設定の数学的根拠まで詳しく言語化して回答してください。

================================================================
【BLOCK 1：監視対象と分離出力の絶対ルール】
================================================================
1. 最優先VIP銘柄は「XAU/USD（ゴールド）」とする。
2. サブ監視銘柄は「USD/JPY, EUR/USD, GBP/JPY, GBP/USD, EUR/JPY, AUD/USD」の主要6通貨ペアとする。
3. 回答は必ず「👑 1. GOLD（XAU/USD）の取引判断」と「🎯 2. それ以外（サブ6通貨ペア）の取引判断」の2大セクションに完全に分けて出力すること。
4. サブ6通貨ペアの相関ダブり防止ルール：通貨強弱の重複を厳禁とし、最も期待値が高い「最良の1銘柄だけ」に厳選すること。優位な場面がなければ「見送り（様子見）」と明記すること。

================================================================
【BLOCK 2：利益最大化のための数学的・構造的絶対条件】
================================================================
1. リスクリワード比（RR） 1 : 1.5 以上の絶対厳守：
   - すべての新規指値注文・新規エントリー提案は、必ず「利確までの値幅（|TP - Entry|） ÷ 損切りまでの値幅（|Entry - SL|） ≧ 1.5」を満たすように設計すること。
   - もしRR 1.5未満になる場合は、無理に指値やエントリーを提示せず「見送り（静観）」とすること。
2. スプレッド誤差の事前織り込み：
   - XMTrading等の実口座スプレッドを考慮し、SLは直近ヒゲの少し外側へ余裕を持たせ、TPは壁の少し手前に設定すること。
3. 建値保護（ブレークイーブン）＆分割利確ルール：
   - 保有中ポジションの含み益がTPの50%以上に到達した場合、SLを建値へ移動して負けゼロを確定させ、半分利確して残りを伸ばす指示を出すこと。
4. 指標ロック＆早朝スプレッド拡大ロック：
   - `indicator_lock` が `true`、または `session_info.entry_allowed` が `false` の場合は、新規エントリーを絶対に推奨せず「見送り（ロック中）」とすること。

================================================================
【BLOCK 3：時間軸・注文スタイル・大口投資家（スマートマネー）視点】
================================================================
1. 基本スタイル：「短期デイトレ（1〜3時間程度で決着を狙う値幅）」かつ「浅い指値注文（現在値から引きつけた押し目買い・戻り売り）」を基本軸とする（有効期限45〜60分）。
2. 超厳選の長期（スイング）枠：4H・1Hの大局トレンド、米ドル強弱スコア、世界情勢の3つが完全一致し RR 1:2.5 以上が狙える特選場面に限り「【超厳選・長期スイング枠】」を提示すること。
3. 大口投資家視点：現在の市場セッション（{session_info['session']}）の特性を踏まえ、個人投資家のストップ狩り（流動性スイープ）に巻き込まれない指値・SL配置を行うこと。

================================================================
【BLOCK 4：リアルタイム入力データ】
================================================================
- 現在日時（日本時間）：{NOW_JST.strftime('%Y-%m-%d %H:%M JST')}
- 現在の市場セッション特性：{json.dumps(session_info, ensure_ascii=False)}
- 経済指標ロック状態：lock={indicator_lock} / {indicator_lock_reason}
- 米ドル強弱インデックス（1H連動算出）：{json.dumps(usd_strength, ensure_ascii=False)}
- XAU/USD 最新チャート指標（4H・1H・15M・5M）：{json.dumps(gold_data, ensure_ascii=False)}
- サブ6通貨ペア 最新チャート指標：{json.dumps(fx_data_list, ensure_ascii=False)}
- 最新ニュースヘッドライン：
{news_headlines}
- 重要経済指標カレンダー：
{calendar_data}
- 現在のAIポジション一覧（OPENは保有中、PENDINGは指値約定待ち）：{json.dumps(trade_log.get('open_positions', []), ensure_ascii=False)}
- 累計戦績＆5分類敗因タグ学習済み進化ルール：{json.dumps(stats_block, ensure_ascii=False)}

================================================================
【BLOCK 5：出力フォーマット（新規取引の完全同期ルール）】
================================================================
【絶対厳守】
LINEレポート本文で「新規指値注文（指値エントリー推奨・押し目/戻り目待機で具体的な指値価格・TP・SLを提示）」または「新規エントリー推奨」を出したものは、**すべて「新規取引」として扱い、必ず最初の `===SIGNALS_JSON_START===` 内にJSONとして出力してください！**
本文に具体的な指値価格・TP・SLを書いたのにJSONへ入れないことは絶対に禁止します。完全見送りの場合のみ空配列 `[]` としてください。

===SIGNALS_JSON_START===
[
  {{
    "symbol": "XAU/USD または サブ最良1銘柄",
    "timeframe_type": "短期デイトレ（1〜3時間・浅い指値） または 超厳選長期スイング",
    "direction": "BUY または SELL",
    "entry_price": 0.0,
    "tp": 0.0,
    "sl": 0.0,
    "reason": "エントリー根拠要約"
  }}
]
===SIGNALS_JSON_END===

続いて、LINEへそのまま送信するユーザー向けレポートを以下の構成で詳しく出力してください。

### 👑 1. GOLD（XAU/USD）の取引判断

**① 現在の保有ポジション・指値待機状況（AI仮想約定ベース）**
※`status="OPEN"`（約定済み・保有中）がある場合：売買方向、買った値段 → 今の値段、現在の損益（＋利益／−損）、TP進捗率（%）、TP・SLまでの残り距離、建値移動・保有アドバイスを明記。
※`status="PENDING"`（指値約定待ち）がある場合：待機中の指値価格、現在値との差、TP・SL予定を明記。
※どちらもない場合：「現在、GOLDの保有中・指値待機中ポジションはありません。」と記載。

**② 今の相場状況（詳しい分析）**
- 【上位足4H・1H＆短期15M・5M構造・壁との距離】
- 【米ドル強弱スコア・世界情勢・大口投資家動向・セッション特性】
- 【指標ロック・学習ルール・安全装置の状態】

**③ 次の一手・新規取引の具体策（未保有・乗り遅れ向け対応）**
- 【短期デイトレ（1〜3時間完結・浅い指値メイン）】
  - 判断アクション：『新規指値エントリー推奨』 / 『押し目・戻り目待機（指値設定）』 / 『見送り（静観）』
  - すでに保有中の場合でも、「約定に乗れていない人が今から新規で取引するならどうすべきか」を必ず提示すること。
  - エントリー目安（浅い指値・有効期限45〜60分）：〇〇.〇〇 ／ 利確（TP）：〇〇.〇〇 ／ 損切り（SL）：〇〇.〇〇（リスクリワード比 1 : 〇.〇）
  - 詳しい判断根拠
- 【超厳選・長期スイング枠】（該当する高確度時のみ記載。なければ「現在、超厳選の長期枠に該当する高確度セットアップはありません」と記載）

---
### 🎯 2. それ以外（サブ6通貨ペア）の取引判断

**① 現在の保有ポジション・指値待機状況（AI仮想約定ベース）**
（保有中・指値待機中があれば詳細、なければ「現在、サブ6通貨ペアの保有中・指値待機中ポジションはありません。」）

**② 今の相場状況＆次の一手（相関ダブり防止・最良1銘柄厳選）**
（サブ6通貨全体の地合い、最良1銘柄の短期戦略・エントリー・TP・SL・RR比・詳しい判断根拠、または見送り理由）
"""
    full_text = generate_with_retry(main_prompt)

    message_body = full_text
    registered_count = 0
    if "===SIGNALS_JSON_START===" in full_text and "===SIGNALS_JSON_END===" in full_text:
        sig_raw = full_text.split("===SIGNALS_JSON_START===")[1].split("===SIGNALS_JSON_END===")[0].strip()
        message_body = full_text.split("===SIGNALS_JSON_END===")[1].strip()
        try:
            new_signals = json.loads(sig_raw)
            existing_keys = {(p["symbol"], p["direction"], round(float(p["entry_price"]), 2)) for p in trade_log.get("open_positions", [])}
            for sig in new_signals:
                if indicator_lock or not session_info["entry_allowed"]:
                    print(f"[Filter Skip] 指標ロックまたは時間帯制限のため新規登録をブロック: {sig.get('symbol')}")
                    continue

                entry_val = float(sig.get("entry_price", 0))
                tp_val = float(sig.get("tp", 0))
                sl_val = float(sig.get("sl", 0))
                if entry_val <= 0 or tp_val <= 0 or sl_val <= 0:
                    continue

                reward = abs(tp_val - entry_val)
                risk = abs(entry_val - sl_val)
                rr_ratio = (reward / risk) if risk > 0 else 0.0

                if rr_ratio < 1.45:
                    print(f"[RR Filter Skip] {sig.get('symbol')} RR={rr_ratio:.2f} のため期待値不足でスキップ")
                    continue

                dedup_key = (sig["symbol"], sig["direction"], round(entry_val, 2))
                if dedup_key not in existing_keys:
                    snap = market_snapshots.get(sig["symbol"], {})
                    cur_val = float(snap.get("current_price", entry_val))
                    high_2h = float(snap.get("recent_2h_high", cur_val))
                    low_2h = float(snap.get("recent_2h_low", cur_val))
                    spread = get_spread_buffer(sig["symbol"])

                    is_already_filled = False
                    if sig["direction"] == "BUY" and (cur_val <= (entry_val - spread * 0.5) or low_2h <= (entry_val - spread * 0.5)):
                        is_already_filled = True
                    elif sig["direction"] == "SELL" and (cur_val >= (entry_val + spread * 0.5) or high_2h >= (entry_val + spread * 0.5)):
                        is_already_filled = True

                    trade_log["open_positions"].append({
                        "symbol": sig["symbol"],
                        "timeframe_type": sig.get("timeframe_type", "短期デイトレ（1〜3時間・浅い指値）"),
                        "direction": sig["direction"],
                        "entry_price": round(entry_val, 4),
                        "current_price": round(cur_val, 4),
                        "tp": round(tp_val, 4),
                        "sl": round(sl_val, 4),
                        "rr_ratio": round(rr_ratio, 2),
                        "breakeven_triggered": False,
                        "unrealized_pnl": round((cur_val - entry_val) if sig["direction"] == "BUY" else (entry_val - cur_val), 4) if is_already_filled else 0.0,
                        "reason": sig.get("reason", ""),
                        "opened_at": NOW_JST.strftime("%Y-%m-%d %H:%M JST"),
                        "status": "OPEN" if is_already_filled else "PENDING",
                    })
                    existing_keys.add(dedup_key)
                    registered_count += 1
            save_json(TRADE_LOG_FILE, trade_log)
        except Exception as e:
            print(f"[Signal JSON Parse Error] {e}")

    header = f"⏰ 【定期AI相場レポート】{NOW_JST.strftime('%m/%d %H:%M')} JST\n"
    if registered_count > 0:
        header += f"📝 新規取引 {registered_count}件 をシステム登録しました（指値到達で自動約定・TP/SLで自動決着）\n"
    if indicator_lock:
        header += f"{indicator_lock_reason}\n\n"
    if evo_reports:
        header += "\n" + "\n\n".join(evo_reports) + "\n\n" + "=" * 24 + "\n\n"

    send_line_message(header + message_body)


# ==============================================================================
# エントリーポイント
# ==============================================================================
if __name__ == "__main__":
    default_trade_log = {"open_positions": [], "closed_history": []}
    default_evolution = {
        "consecutive_losses": 0,
        "strict_mode": False,
        "tag_counts": {"TAG_1": 0, "TAG_2": 0, "TAG_3": 0, "TAG_4": 0, "TAG_5": 0},
        "gold_short_rules": [
            "初期設定：RR 1:1.5以上を確保しつつ1H・15Mの押し目/戻り目を浅い指値（有効期限45〜60分）で狙う",
            "エントリー前に4Hトレンド、米ドル強弱、直近24H高値・安値・節目の壁までの距離を必ず確認する",
        ],
        "gold_long_rules": [
            "4H・1Hの大局トレンド、米ドル強弱、世界情勢の3つが完全一致しRR 1:2.5以上が狙える特選場面のみ提示する",
        ],
        "fx_pairs_rules": [
            "通貨強弱の相関ダブりを避け、サブ6通貨ペアから最も期待値が高い最良1銘柄のみ厳選する",
        ],
        "weekend_market_outlook": "",
        "last_evolution_note": "",
    }

    trade_log = load_json(TRADE_LOG_FILE, default_trade_log)
    evolution_data = load_json(EVOLUTION_FILE, default_evolution)

    if is_weekend_closed(NOW_JST):
        print("[Mode] 土日休場：週末集中学習モードを実行します")
        run_weekend_study_mode(trade_log, evolution_data)
    else:
        print("[Mode] 平日市場オープン：利益最大化プロ仕様分析を実行します")
        run_weekday_analysis(trade_log, evolution_data)
