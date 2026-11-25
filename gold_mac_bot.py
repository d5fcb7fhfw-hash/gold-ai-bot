import os
import json
import time
import math
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import google.generativeai as genai

# ==============================================================================
# 環境変数（GitHub Secrets に設定）
# ==============================================================================
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]


# ==============================================================================
# Gemini 呼び出し（429レート制限時のモデル自動切替 ＆ 待機リトライ付き）
# ==============================================================================
def generate_with_retry(prompt, retries=2):
    for model_name in MODEL_CANDIDATES:
        model = genai.GenerativeModel(model_name)
        for attempt in range(retries):
            try:
                resp = model.generate_content(prompt)
                return resp.text.strip()
            except Exception as e:
                err_str = str(e)
                if "PerDay" in err_str:
                    print(f"[Gemini DailyLimit] {model_name} の1日上限到達。次の予備モデルへ切り替えます...")
                    break
                elif "429" in err_str and attempt < retries - 1:
                    print(f"[Gemini RateLimit] 1分上限のため65秒待機して再試行します... ({attempt + 1}/{retries})")
                    time.sleep(65)
                else:
                    if model_name == MODEL_CANDIDATES[-1]:
                        raise e
                    break
    raise RuntimeError("すべてのGeminiモデルの無料枠上限に達しました。")


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
# 2. テクニカル指標・壁（サポレジ・節目）計算
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
    current_price = closes_15m[-1]

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

    ema20_15m = calc_ema(closes_15m, 20)
    rsi_15m = calc_rsi(closes_15m, 14)
    atr_15m = calc_atr(c_15m, 14)

    result = {
        "symbol": symbol,
        "current_price": round(current_price, 4),
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
# 3. 最新ニュース・週末アナリスト見解・経済指標取得
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


def fetch_economic_calendar():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        req = urllib.request.urlopen(url, timeout=10)
        events = json.loads(req.read().decode("utf-8"))
        high_events = []
        for ev in events:
            if ev.get("impact") in ["High", "Medium"] and ev.get("country") in ["USD", "JPY", "EUR", "GBP", "AUD"]:
                high_events.append(
                    f"- [{ev.get('country')}] {ev.get('title')} | 日時(UTC): {ev.get('date')} | 重要度: {ev.get('impact')} | 予想: {ev.get('forecast')} / 前回: {ev.get('previous')}"
                )
        return "\n".join(high_events[:20]) if high_events else "直近の主要指標情報なし"
    except Exception as e:
        return f"指標カレンダー取得エラー: {e}"


# ==============================================================================
# 4. 市場オープン・土日判定
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
# 5. 決着（TP/SL到達）チェック ＆ 5分類タグ敗因解剖・自己進化
# ==============================================================================
def check_positions_and_evolve(trade_log, evolution_data, latest_prices, news_summary):
    open_positions = trade_log.get("open_positions", [])
    closed_history = trade_log.get("closed_history", [])
    still_open = []
    newly_closed = []

    for pos in open_positions:
        sym = pos["symbol"]
        cur = latest_prices.get(sym)
        if cur is None:
            still_open.append(pos)
            continue

        direction = pos["direction"]
        tp = float(pos["tp"])
        sl = float(pos["sl"])
        entry = float(pos["entry_price"])

        result = None
        close_price = cur

        if direction == "BUY":
            if cur >= tp:
                result = "WIN"
                close_price = tp
            elif cur <= sl:
                result = "LOSS"
                close_price = sl
        elif direction == "SELL":
            if cur <= tp:
                result = "WIN"
                close_price = tp
            elif cur >= sl:
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
            pos["current_price"] = round(cur, 4)
            pos["unrealized_pnl"] = round(unrealized, 4)
            pos["dist_to_tp"] = round(abs(tp - cur), 4)
            pos["dist_to_sl"] = round(abs(cur - sl), 4)
            still_open.append(pos)

    trade_log["open_positions"] = still_open
    trade_log["closed_history"] = closed_history

    consecutive_losses = 0
    for item in reversed(closed_history):
        if item.get("result") == "LOSS":
            consecutive_losses += 1
        else:
            break
    evolution_data["consecutive_losses"] = consecutive_losses
    evolution_data["strict_mode"] = True if consecutive_losses >= 2 else False

    evolution_reports = []
    for closed_pos in newly_closed:
        evo_prompt = f"""
あなたは自己進化型AIトレーディングシステムの「戦績解剖・ルール改善エンジン」です。
たった今、以下の仮想トレードが利確（TP）または損切り（SL）に到達して取引終了（決着）となりました。
このトレードの結果（特に損切りの場合の敗因）を冷徹かつ論理的に解剖し、次回以降のトレード精度を高めるための「進化ルール」を更新してください。

================================================================
【BLOCK 1：決着したトレードの事実データ】
================================================================
- 銘柄：{closed_pos['symbol']}
- 時間軸区分：{closed_pos.get('timeframe_type', '短期デイトレ（1〜3時間・浅い指値）')}
- 売買方向：{closed_pos['direction']}
- エントリー価格 → 決着価格：{closed_pos['entry_price']} → {closed_pos['close_price']}
- 決着結果：{closed_pos['result']}（WIN：利確到達 / LOSS：損切り到達）
- 獲得・損失値幅：{closed_pos['pnl']}
- エントリー日時：{closed_pos.get('opened_at')} → 決着日時：{closed_pos.get('closed_at')}
- エントリー時の根拠メモ：{closed_pos.get('reason', '')}
- 直近ニュース背景：{news_summary}
- 現在の進化ルール帳（更新前）：{json.dumps(evolution_data, ensure_ascii=False)}
- 現在の連敗カウント：{consecutive_losses}

================================================================
【BLOCK 2：敗因の5分類タグ特定ルール（LOSS確定時は必須）】
================================================================
結果が「LOSS（損切り到達）」だった場合、必ず以下の5つの敗因タグの中から最も当てはまる【主因タグ】を1つ特定し、具体的な発生メカニズムを分析してください。

- TAG_1【上位足逆行・壁衝突】：1時間足などの上位トレンドに逆らっていた、または直近高値・安値・キリのいい節目（壁）のすぐ手前でエントリーして跳ね返された。
- TAG_2【指標・急変ニュース・ドル急変】：経済指標の発表、要人発言・地政学ヘッドライン、または米ドルの急激な方向転換に巻き込まれた。
- TAG_3【指値刺さり遅れ・鮮度切れ】：浅い指値を置いた後、45〜60分近く経って相場環境が変わってから遅れて約定し、不利な位置で捕まった。
- TAG_4【SL狭すぎ・ヒゲ狩り】：方向性の見立て自体は合っていたが、損切り（SL）幅がATRや直近ノイズに対して狭すぎたため、一時的なヒゲで刈られた後に本来の方向へ伸びた。
- TAG_5【レンジのダマシブレイク】：方向感のないレンジ相場の端でブレイクや押し目と誤認して飛び乗り、レンジ内へ引き戻された。

================================================================
【BLOCK 3：ルール自動進化・書き換え方針】
================================================================
1. 勝った場合（WIN）：どの要素が勝因になったかを抽出し、「勝ちパターン」として強化・記録すること。
2. 負けた場合（LOSS）：特定した5分類タグの累計傾向を確認し、次回からそれを防ぐ具体的な改善ルールを1〜2文で生成して進化ルール帳に反映すること。
3. 探索バランス維持（引き締めすぎ防止）：「チャンスを逃さない広め設定」を土台にしつつ、「明らかに負ける罠だけを避けるフィルター」へ洗練させること。

================================================================
【BLOCK 4：出力フォーマット（必ず純粋なJSONのみ出力）】
================================================================
{{
  "win_or_loss": "{closed_pos['result']}",
  "primary_failure_tag": "LOSSの場合は TAG_1〜TAG_5 のいずれか（WINの場合は NONE）",
  "analysis_summary": "なぜ勝てたか／なぜ負けたかの詳しい解剖コメント",
  "new_or_updated_rule": "次回以降のトレード判断プロンプトへ反映する具体的な改善ルール",
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
# 6. LINE プッシュ通知送信
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
# 7. 土日専用：週末集中学習＆来週の作戦レポート実行
# ==============================================================================
def run_weekend_study_mode(trade_log, evolution_data):
    hr = NOW_JST.hour
    wd = NOW_JST.weekday()
    if hr % 3 != 0:
        print(f"[Weekend] {NOW_JST.strftime('%H:%M')} は週末待機スロットのためスキップ")
        return

    analyst_news = fetch_google_news(
        ["ドル円 来週 見通し FX", "ゴールド 金相場 来週 予想", "FRB 金利 見通し 週間", "XAUUSD weekly outlook"],
        limit_per_query=4,
    )
    calendar_data = fetch_economic_calendar()

    weekend_prompt = f"""
あなたは週末集中学習モードで稼働中の「自己進化型AIトレーディング研究官」です。
現在、外国為替（FX）およびゴールド（XAU/USD）市場は土日休場中のため、新規の売買シグナルは出しません。
その代わり、この休場時間を使って「①過去の全トレードデータの徹底振り返り」と「②外部のトレーダー・アナリスト見解および来週の市場テーマの学習」を行い、月曜朝7時の市場オープンに向けた知識と戦略をアップデートしてください。

================================================================
【BLOCK 1：週末学習用の入力データ】
================================================================
- 学習実行日時（日本時間）：{NOW_JST.strftime('%Y-%m-%d %H:%M JST')}
- 累計の全件約定トレード履歴：{json.dumps(trade_log.get('closed_history', [])[-30:], ensure_ascii=False)}
- 5分類敗因タグの累計データと現在の進化ルール帳：{json.dumps(evolution_data, ensure_ascii=False)}
- 土日に収集した国内外FXアナリスト・トレーダー見解・週間見通しニュース：
{analyst_news}
- 来週の重要経済指標カレンダー：
{calendar_data}
- 金曜クローズ時点の持ち越しポジション：{json.dumps(trade_log.get('open_positions', []), ensure_ascii=False)}

================================================================
【BLOCK 2：週末に実行すべき3つの深掘り学習タスク】
================================================================
1. 【内部データ解剖】今週・過去の勝ちパターンと負けパターンの総点検：
   - 銘柄別（GOLD vs サブ6通貨）、短期 vs 長期ごとの勝率と損益傾向を分析する。
   - 5分類敗因タグ（TAG_1〜TAG_5）のうち、最も損失の原因になっている弱点を特定し、来週から適用する「具体的な改善ルール」をアップグレードする。
2. 【外部知識の吸収】プロトレーダー・市場参加者の目線の言語化：
   - 収集した週間見通しやアナリスト意見から、「来週、世界中のトレーダーが最も意識している価格帯（サポート・レジスタンス・心理的節目）」と「市場のメインテーマ（FRB金利観測、インフレ指標、地政学リスク、日銀動向など）」を抽出して知識ベースに整理する。
3. 【持ち越しポジション＆月曜窓開け対策】：
   - 金曜から持ち越している保有ポジションがある場合、週末のニュースを踏まえた月曜朝のリスクと対応方針を整理する。

================================================================
【BLOCK 3：出力フォーマット】
================================================================
最初に必ず `===EVOLUTION_JSON_START===` と `===EVOLUTION_JSON_END===` で囲んで更新後の進化ルールJSONを出力し、その後にLINE送信用の「📚 週末の集中学習＆来週の作戦レポート」を詳しく書いてください。

===EVOLUTION_JSON_START===
{{
  "weekend_market_outlook": "来週の相場テーマ・トレーダー意識価格帯の要約（来週の判断プロンプトへ自動注入される知識メモ）",
  "gold_short_rules": ["更新後のGOLD短期ルールリスト"],
  "gold_long_rules": ["更新後のGOLD超厳選長期ルールリスト"],
  "fx_pairs_rules": ["更新後のサブ6通貨ペア共通ルールリスト"]
}}
===EVOLUTION_JSON_END===

### 📚 週末の集中学習＆来週の作戦レポート

**1. 今週のトレード振り返り＆自己進化アップデート**
（累計成績サマリー、機能した勝ちパターン、5分類敗因タグの弱点分析と来週の新ルールを詳しく記載）

**2. 外部トレーダー見解＆来週の市場メインテーマ**
（アナリスト・トレーダーが注目している焦点と意識されている価格帯・節目マップを詳しく記載）

**3. 👑 来週のGOLD（XAU/USD）の狙い方（短期メイン ＋ 超厳選長期の目線）**
（持ち越しポジション対応、月曜朝7時〜来週前半の短期1〜3H・浅い指値戦略、超厳選長期スイング条件を詳しく記載）

**4. 🎯 来週のサブ6通貨ペアの注目銘柄と戦略**
（相関ダブりを避けた来週の最注目通貨ペアと具体的な狙い目を詳しく記載）
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

    if (wd == 6 and hr == 21) or (wd == 5 and hr == 9):
        send_line_message(report_text)
    else:
        print("[Weekend] 知識アップデート完了（LINE配信時間外のため保存のみ）")


# ==============================================================================
# 8. 平日メイン分析（1時間ごとの自動通知・全件約定記録）
# ==============================================================================
def run_weekday_analysis(trade_log, evolution_data):
    gold_data = analyze_symbol_market(GOLD_SYMBOL, is_vip=True)
    fx_data_list = [analyze_symbol_market(sym, is_vip=False) for sym in SUB_PAIRS]

    latest_prices = {}
    if "current_price" in gold_data:
        latest_prices[GOLD_SYMBOL] = gold_data["current_price"]
    for item in fx_data_list:
        if "current_price" in item:
            latest_prices[item["symbol"]] = item["current_price"]

    news_headlines = fetch_google_news(
        ["ゴールド 金価格 相場", "ドル円 為替 FRB", "XAUUSD gold price news", "米国債 金利 地政学"],
        limit_per_query=3,
    )
    calendar_data = fetch_economic_calendar()

    evo_reports = check_positions_and_evolve(trade_log, evolution_data, latest_prices, news_headlines)

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
ユーザーはあなたの提示したトレード判断を信頼して実戦の参考にします。表面的な一言コメントは厳禁とし、チャート構造・世界情勢・米ドル強弱・価格設定の理由・過去の学習ルールまで、根拠を具体的かつ詳しく言語化して回答してください。

================================================================
【BLOCK 1：監視対象と分離出力の絶対ルール】
================================================================
1. 最優先VIP銘柄は「XAU/USD（ゴールド）」とする。
2. サブ監視銘柄は「USD/JPY, EUR/USD, GBP/JPY, GBP/USD, EUR/JPY, AUD/USD」の主要6通貨ペアとする。
3. 回答は必ず「👑 1. GOLD（XAU/USD）の取引判断」と「🎯 2. それ以外（サブ6通貨ペア）の取引判断」の2大セクションに完全に分けて出力すること。両者を混ぜてはならない。
4. サブ6通貨ペアの相関ダブり防止ルール：
   - サブ6通貨ペアから新規エントリーを提示する場合は、通貨強弱の重複（実質同じ賭けになる二重リスク）を厳禁とし、最も期待値と優位性が高い「最良の1銘柄だけ」に厳選すること。優位な場面がなければ無理に出さず「見送り（様子見）」と明記すること。

================================================================
【BLOCK 2：時間軸・注文スタイル・決着基準のルール】
================================================================
1. 基本スタイル（メイン戦略）：
   - 「短期デイトレ（1〜3時間程度で決着を狙う値幅）」かつ「浅い指値注文（現在値から引きつけた押し目買い・戻り売り）」を基本軸とする。
2. 浅い指値の鮮度ルール（45〜60分制限）：
   - 提示する浅い指値は「提示から45〜60分以内に約定しなければ鮮度切れ（無効・キャンセル）」とする前提で設計すること。
3. 超厳選の長期（スイング）枠：
   - 長期トレードは損切り幅が大きくなるため、普段は提示しない。
   - ただし、「4時間足・日足レベルの大局トレンド」「最新の世界情勢・FRB金利観測」「米ドルの強弱」の3つが完全に同じ方向へ揃い、勝率・優位性が極めて高いと判断できる特選場面に限り、短期枠とは別に「【超厳選・長期スイング枠】」としてしっかり提示すること。
4. 取引終了（決着）の基準：
   - 時間が経過しただけの曖昧な途中決済は行わない。提示した利確（TP）または損切り（SL）の価格ラインに到達した時点を「取引終了（WIN / LOSS確定）」とする。

================================================================
【BLOCK 3：仕掛けの姿勢・安全装置・自己進化ルール】
================================================================
1. 探索と進化の基本姿勢：
   - 初期および通常時は「チャンスを逃さない広めの基準」で積極的に優位な場面を探し出すこと。
   - 過去の全件約定データから得られた勝ちパターン・負けパターン（5分類敗因タグ学習済みルール）を常に参照し、過去に負けた状況と同じ罠にはまらないよう条件を洗練させて判断すること。
2. 目の前の壁（サポレジ・節目）チェック：
   - 買いエントリーのすぐ上に強いレジスタンスの壁がある場合、または売りエントリーのすぐ下に強いサポートの壁がある場合、TPに届く前に跳ね返されるリスクが高いため「見送り」とするか、「壁の手前に現実的なTPを設定する」こと。
3. 2連敗時の自動厳格化ブレーキ：
   - `strict_mode_active` が `true`（直近2連敗）の場合は、「チャンスを逃さない広め設定」を一時解除し、1H・15M・5M・壁までの距離・ドル強弱のすべてが綺麗に揃った「A級チャンスのみ」に基準を厳格化して提示すること。

================================================================
【BLOCK 4：リアルタイム入力データ】
================================================================
- 現在日時（日本時間）：{NOW_JST.strftime('%Y-%m-%d %H:%M JST')}
- XAU/USD 最新チャート指標：{json.dumps(gold_data, ensure_ascii=False)}
- サブ6通貨ペア 最新チャート指標：{json.dumps(fx_data_list, ensure_ascii=False)}
- 最新の世界情勢・FRB・地政学ニュースヘッドライン：
{news_headlines}
- 重要経済指標カレンダー：
{calendar_data}
- 現在のAI保有中ポジション一覧（全件約定扱い）：{json.dumps(trade_log.get('open_positions', []), ensure_ascii=False)}
- 累計戦績＆5分類敗因タグ学習済み進化ルール＆週末学習メモ：{json.dumps(stats_block, ensure_ascii=False)}

================================================================
【BLOCK 5：出力フォーマット（システム登録用JSON ＋ LINE配信テキスト）】
================================================================
最初に必ず `===SIGNALS_JSON_START===` と `===SIGNALS_JSON_END===` で囲んで、今回「新規エントリー推奨」としたシグナルだけをJSON配列で出力してください（見送り・待機のみの場合は空配列 `[]`）。
全件約定ルールに基づき、ここに含めた新規シグナルは自動的に仮想保有ポジションへ登録されます（同銘柄・同方向の重複登録は避けること）。

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

**① 現在の保有ポジション状況（AI仮想約定ベース）**
※保有中がある場合：売買方向、買った値段 → 今の値段、現在の損益（利益／損）、利確（TP）・損切り（SL）までの残り距離、保有継続・撤退アドバイスを明記。なければ「現在、GOLDの保有中ポジションはありません。」と記載。

**② 今の相場状況（詳しい分析）**
- 【チャート構造（1H・15M・5M）】トレンド方向、押し目/戻り目、直近高値・安値・節目（壁）との距離を詳しく解説。
- 【世界情勢・米ドル強弱・指標】最新ニュースやFRB金利観測、ドル強弱がGOLDに追い風か逆風か、重要指標の接近有無を詳しく解説。
- 【学習ルール・安全装置の状態】過去の5分類敗因学習から今回特に警戒している点、2連敗ブレーキの作動有無を明記。

**③ 次の一手・新規取引の具体策（未保有・乗り遅れ向け対応）**
- 【短期デイトレ（1〜3時間完結・浅い指値メイン）】
  - 判断アクション：『新規指値エントリー推奨』 / 『押し目・戻り目待機』 / 『見送り（静観）』を明記。
  - すでに保有中の場合でも、「約定に乗れていない人が今から新規で取引するならどうすべきか」を必ず提示すること。
  - 価格がすでに伸びきっていて今から入ると高値掴み・安値掴みになる場合は、「今は飛び乗らず、〇〇.〇〇ドルまでの押し目（戻り目）を待つべき」と明確な待機ラインを示すこと。
  - エントリー目安（浅い指値・有効期限45〜60分）：〇〇.〇〇 ／ 利確（TP）：〇〇.〇〇 ／ 損切り（SL）：〇〇.〇〇
  - 詳しい判断根拠：なぜこの方向・この価格設定なのかを論理的に詳述。
- 【超厳選・長期スイング枠】（該当する高確度時のみ詳細を記載。該当しない場合は「現在、超厳選の長期枠に該当する高確度セットアップはありません」と記載）

---
### 🎯 2. それ以外（サブ6通貨ペア）の取引判断

**① 現在の保有ポジション状況（AI仮想約定ベース）**
※保有中がある場合：銘柄名、売買方向、買った値段 → 今の値段、現在の損益、TP・SLまでの残り距離を記載。なければ「現在、サブ6通貨ペアの保有中ポジションはありません。」と記載。

**② 今の相場状況＆次の一手（相関ダブり防止・最良1銘柄厳選）**
- サブ6通貨ペア全体の地合いを解説し、優位なチャンスがあれば相関ダブりを避けた**最良の1銘柄だけ**について短期（1〜3時間・浅い指値・鮮度45〜60分）のエントリー・TP・SL・壁チェック・詳しい判断根拠（乗り遅れ向け対応含む）を記載。なければ「現在、サブ6通貨ペアに無理に入るべき優位な場面はありません（様子見）」とその理由を詳しく記載。
"""
    full_text = generate_with_retry(main_prompt)

    message_body = full_text
    if "===SIGNALS_JSON_START===" in full_text and "===SIGNALS_JSON_END===" in full_text:
        sig_raw = full_text.split("===SIGNALS_JSON_START===")[1].split("===SIGNALS_JSON_END===")[0].strip()
        message_body = full_text.split("===SIGNALS_JSON_END===")[1].strip()
        try:
            new_signals = json.loads(sig_raw)
            existing_keys = {(p["symbol"], p["direction"]) for p in trade_log.get("open_positions", [])}
            for sig in new_signals:
                key = (sig["symbol"], sig["direction"])
                if key not in existing_keys and float(sig.get("entry_price", 0)) > 0:
                    trade_log["open_positions"].append({
                        "symbol": sig["symbol"],
                        "timeframe_type": sig.get("timeframe_type", "短期デイトレ（1〜3時間・浅い指値）"),
                        "direction": sig["direction"],
                        "entry_price": round(float(sig["entry_price"]), 4),
                        "current_price": round(float(sig["entry_price"]), 4),
                        "tp": round(float(sig["tp"]), 4),
                        "sl": round(float(sig["sl"]), 4),
                        "unrealized_pnl": 0.0,
                        "reason": sig.get("reason", ""),
                        "opened_at": NOW_JST.strftime("%Y-%m-%d %H:%M JST"),
                        "status": "OPEN",
                    })
                    existing_keys.add(key)
            save_json(TRADE_LOG_FILE, trade_log)
        except Exception as e:
            print(f"[Signal JSON Parse Error] {e}")

    header = f"⏰ 【定期AI相場レポート】{NOW_JST.strftime('%m/%d %H:%M')} JST\n"
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
            "初期設定：チャンスを逃さない広めの基準で1H・15Mの押し目/戻り目を浅い指値（有効期限45〜60分）で狙う",
            "エントリー前に直近24H高値・安値・キリのいい価格帯（壁）までの距離を必ず確認する",
        ],
        "gold_long_rules": [
            "4H・日足の大局トレンド、FRB金利/世界情勢、米ドル強弱の3つが完全一致した特選場面のみ提示する",
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
        print("[Mode] 平日市場オープン：定期分析＆自己進化を実行します")
        run_weekday_analysis(trade_log, evolution_data)
