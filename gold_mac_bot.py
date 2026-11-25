import os
import json
import time
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
from google import genai
from google.genai import types

import os
# （その他のimportは先ほどと同じ）

# ==========================================
# 1. 設定項目（GitHub Secretsから自動読み込み）
# ==========================================
TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_USER_ID = os.environ["LINE_USER_ID"]


# ==========================================
# 2. 経済指標カレンダー取得（ForexFactory無料XML → 日本時間変換）
# ==========================================
def get_economic_calendar_jst() -> str:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        root = ET.fromstring(res.content)
        
        now_jst = datetime.now(JST)
        today_str = now_jst.strftime("%m-%d-%Y")
        tomorrow_str = (now_jst + timedelta(days=1)).strftime("%m-%d-%Y")
        
        events = []
        for event in root.findall(".//event"):
            country = event.findtext("country", default="").strip()
            impact = event.findtext("impact", default="").strip()
            # ゴールドに直結するUSDのHigh/Medium指標のみ抽出
            if country != "USD" or impact not in ["High", "Medium"]:
                continue
                
            date_str = event.findtext("date", default="").strip()
            time_str = event.findtext("time", default="").strip()
            title = event.findtext("title", default="").strip()
            forecast = event.findtext("forecast", default="-").strip()
            previous = event.findtext("previous", default="-").strip()
            
            if date_str in [today_str, tomorrow_str]:
                # ForexFactoryは基本UTC（またはEST）表記のため日本時間目安を付記
                events.append(f"- [{date_str} {time_str} UTC] 【重要度:{impact}】{title} (予想:{forecast} / 前回:{previous})")
                
        if not events:
            return "本日〜明日の重要USD経済指標は特にありません。"
        return "\n".join(events)
    except Exception as e:
        return f"経済指標カレンダー取得スキップ（ニュース見出しで代用）: {e}"

# ==========================================
# 3. GoogleニュースRSS取得（米国市場＋日本市場）
# ==========================================
def get_google_news_rss() -> str:
    queries = [
        ("XAUUSD OR \"gold price\" OR Fed OR \"Treasury yield\" OR \"US dollar\" when:1d", "en-US", "US", "US:en", 8),
        ("ゴールド相場 OR 金価格 OR ドル円 OR FOMC OR 米金利 when:1d", "ja", "JP", "JP:ja", 5),
    ]
    news_list = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for q, hl, gl, ceid, limit in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl={hl}&gl={gl}&ceid={ceid}"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            root = ET.fromstring(res.content)
            for item in root.findall(".//item")[:limit]:
                title = item.findtext("title", default="").strip()
                pub_date = item.findtext("pubDate", default="").strip()
                if title:
                    news_list.append(f"- [{pub_date}] {title}")
        except Exception:
            pass

    return "\n".join(news_list) if news_list else "主要ニュースなし"

# ==========================================
# 4. 高度テクニカル＆プライスアクション数値化データ取得
# ==========================================
def get_market_data(symbol: str, interval: str, outputsize: int = 80) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY
    }
    res = requests.get(url, params=params).json()
    if "values" not in res:
        raise ValueError(f"{symbol} ({interval}) データ取得エラー: {res}")

    df = pd.DataFrame(res["values"]).iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # トレンド指標（EMA20, EMA50）
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ボリンジャーバンド（20期間 ±2σ）
    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["BB_Upper"] = sma20 + (std20 * 2)
    df["BB_Lower"] = sma20 - (std20 * 2)
    
    # RSI14
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ATR14（ボラティリティ）
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(window=14).mean()

    # プライスアクション（ヒゲの割合％：大口の反発サイン検出）
    candle_range = (df["high"] - df["low"]).replace(0, 0.01)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["UpWick%"] = (upper_wick / candle_range * 100).round(0)
    df["LowWick%"] = (lower_wick / candle_range * 100).round(0)

    cols = ["datetime", "open", "high", "low", "close", "EMA20", "EMA50", "BB_Upper", "BB_Lower", "RSI14", "ATR14", "UpWick%", "LowWick%"]
    return df[cols].round(2)

# ==========================================
# 5. 市場構造（重要サポレジ・ラウンドナンバー）の自動抽出
# ==========================================
def extract_key_levels(df_d1: pd.DataFrame, df_h1: pd.DataFrame, current_price: float) -> str:
    # 前日高値・安値・終値
    prev_day = df_d1.iloc[-2]
    pdh = prev_day["high"]
    pdl = prev_day["low"]
    pdc = prev_day["close"]
    
    # 直近5日間の最高値・最安値
    w_high = df_d1.tail(5)["high"].max()
    w_low = df_d1.tail(5)["low"].min()
    
    # 直近24時間（1時間足24本）の高値・安値
    h24_high = df_h1.tail(24)["high"].max()
    h24_low = df_h1.tail(24)["low"].min()
    
    # ラウンドナンバー（10ドル刻みの節目）
    base_10 = round(current_price / 10) * 10
    rn_above = base_10 if base_10 > current_price else base_10 + 10
    rn_below = base_10 if base_10 < current_price else base_10 - 10

    return f"""
    ・現在値：{current_price}
    ・直近24時間 高値：{h24_high} / 安値：{h24_low}
    ・前日高値(PDH)：{pdh} / 前日安値(PDL)：{pdl} / 前日終値：{pdc}
    ・直近5日間 最高値：{w_high} / 最安値：{w_low}
    ・直近の意識されるラウンドナンバー（節目）：上値 {rn_above:.2f} / 下値 {rn_below:.2f}
    """

# ==========================================
# 6. 仮想トレード記録・自動採点（答え合わせ）
# ==========================================
def load_trade_logs() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_trade_logs(logs: list):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def evaluate_past_trades(logs: list, df_m15_full: pd.DataFrame) -> tuple[list, str, int]:
    if not logs:
        return logs, "まだ過去の仮想トレード記録がありません（今回が初回記録になります）。", 0

    df_check = df_m15_full.copy()
    df_check["dt"] = pd.to_datetime(df_check["datetime"])
    newly_settled = []

    for trade in logs:
        if trade["status"] not in ["PENDING", "OPEN"]:
            continue

        trade_time = pd.to_datetime(trade["timestamp"])
        future_bars = df_check[df_check["dt"] > trade_time]
        if future_bars.empty:
            continue

        direction = trade["direction"]
        entry = float(trade["entry"])
        tp = float(trade["tp"])
        sl = float(trade["sl"])
        status = trade["status"]

        for _, row in future_bars.iterrows():
            high = float(row["high"])
            low = float(row["low"])

            if status == "PENDING":
                if direction == "BUY" and low <= entry:
                    status = "OPEN"
                elif direction == "SELL" and high >= entry:
                    status = "OPEN"

            if status == "OPEN":
                if direction == "BUY":
                    if low <= sl:
                        status = "LOSS"
                        trade["pnl"] = round(sl - entry, 2)
                        newly_settled.append(f"❌ 負け（BUY {entry} → SL {sl} 被弾 / {trade['pnl']}ドル）")
                        break
                    elif high >= tp:
                        status = "WIN"
                        trade["pnl"] = round(tp - entry, 2)
                        newly_settled.append(f"✅ 勝ち（BUY {entry} → TP {tp} 到達 / +{trade['pnl']}ドル）")
                        break
                elif direction == "SELL":
                    if high >= sl:
                        status = "LOSS"
                        trade["pnl"] = round(entry - sl, 2)
                        newly_settled.append(f"❌ 負け（SELL {entry} → SL {sl} 被弾 / {trade['pnl']}ドル）")
                        break
                    elif low <= tp:
                        status = "WIN"
                        trade["pnl"] = round(entry - tp, 2)
                        newly_settled.append(f"✅ 勝ち（SELL {entry} → TP {tp} 到達 / +{trade['pnl']}ドル）")
                        break

        if status == "PENDING" and (datetime.now() - trade_time).total_seconds() > 43200:
            status = "EXPIRED"

        trade["status"] = status

    wins = [t for t in logs if t["status"] == "WIN"]
    losses = [t for t in logs if t["status"] == "LOSS"]
    active_trades = [t for t in logs if t["status"] in ["PENDING", "OPEN"]]
    
    total_settled = len(wins) + len(losses)
    win_rate = round((len(wins) / total_settled * 100), 1) if total_settled > 0 else 0.0
    total_pnl = round(sum(t.get("pnl", 0) for t in wins + losses), 2)

    summary_lines = [
        f"【AI仮想トレード通算成績】{len(wins)}勝 {len(losses)}敗（勝率: {win_rate}% / 累計損益: {total_pnl:+}ドル幅 / 現在保有・待機中: {len(active_trades)}件）"
    ]
    if newly_settled:
        summary_lines.append("【直近の答え合わせ結果】")
        summary_lines.extend(newly_settled)

    recent_losses = losses[-3:]
    if recent_losses:
        summary_lines.append("【過去の敗因データ（同じ失敗を絶対に繰り返さないこと）】")
        for l in recent_losses:
            summary_lines.append(f"- {l['timestamp']} {l['direction']} Entry:{l['entry']} SL:{l['sl']} TP:{l['tp']} → SL被弾")

    return logs, "\n".join(summary_lines), len(active_trades)

# ==========================================
# 7. プロ仕様・全情報統合AI分析
# ==========================================
def analyze_master_strategy(
    data_dict: dict,
    df_dxy: pd.DataFrame,
    df_xag: pd.DataFrame,
    key_levels_text: str,
    calendar_text: str,
    news_text: str,
    feedback_text: str
) -> str:
    now_jst_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    
    system_instruction = """
    あなたはヘッジファンドのチーフXAU/USD（ゴールド）ストラテジストです。
    以下の全データを統合し、圧倒的な精度と優位性を持つ実践的トレード戦略を立案してください。

    【分析の5大柱】
    1. 市場構造（Key Levels）：前日高値安値(PDH/PDL)・直近24時間高値安値・ラウンドナンバーを重視し、大口のストップ狩り（流動性回収）が起きやすい位置を避けて指値とSLを配置する。
    2. マルチタイムフレーム（MTF）：1時間足（40%）・15分足（40%）を主軸とし、5分足（20%）のヒゲ比率（UpWick%/LowWick%）とボリンジャーバンド（±2σ）から反発・ブレイクの優位点を特定する。日足・4Hは大局の壁として意識する。
    3. 相関分析：米ドル指数（DXY）との逆相関、および銀（XAG/USD）との同調性を確認する。
    4. ファンダメンタルズ＆指標ガード：経済指標カレンダーを確認し、重要度Highの米ドル指標発表が「現在時刻から前後30分以内」に迫っている場合のみ【様子見】とする。それ以外の時間帯は強気に【買い目線】または【売り目線】の優位な指値・成行プランを必ず提示する。
    5. 自己学習フィードバック：過去の仮想トレード成績でSL被弾（負け）がある場合は、ATR14の1.5倍〜2.0倍の余裕を持たせる・重要な防衛ラインの外側にSLを置くなど、明確に改善策を反映する。

    【重要：出力フォーマット厳守（自動採点システム連携のため絶対厳守）】
    1行目は必ず以下のいずれか完全一致で記載すること：
    ■ 総合判定：【買い目線】
    ■ 総合判定：【売り目線】
    ■ 総合判定：【様子見】

    2行目以降は以下の形式で出力すること（価格・数値は半角数字・小数点2桁のみで記載し、単位や「〜」は書かないこと）：
    ■ 自己検証＆改善アクション：
    （過去成績の振り返りと、今回のSL/指値配置の工夫を1〜2行で記載）
    ■ 市場環境（MTF・相関・指標リスク）：
    ・トレンド構造：（日足〜5分足の整合性と意識されるサポレジ）
    ・ドル＆銀相関：（DXY・XAGの動きとの一致度）
    ・経済指標＆ニュース：（直近の指標時刻とニュース材料の要約）
    ■ 実践トレードプラン：
    ・注文方法：Buy Limit / Sell Limit / 成行買い / 成行売り（様子見の場合は「なし」）
    ・エントリー価格：2650.50
    ・利確（TP）：2678.00
    ・損切り（SL）：2636.00
    ・リスクリワード比：1 : 2.0
    ・根拠：（なぜこの価格が機関投資家目線で優位なのか1〜2行で解説）
    """

    prompt = f"""
    現在時刻：{now_jst_str}

    {feedback_text}

    ■ 重要サポレジ・市場構造（Key Levels）
    {key_levels_text}

    ■ 本日〜明日の重要経済指標カレンダー（ForexFactory）
    {calendar_text}

    ■ 直近24時間の最新金融ニュース（GoogleニュースRSS）
    {news_text}

    ■ 1. XAU/USD 日足（1D・直近8本）
    {data_dict["1day"].tail(8).to_string(index=False)}

    ■ 2. XAU/USD 4時間足（4H・直近10本）
    {data_dict["4h"].tail(10).to_string(index=False)}

    ■ 3. XAU/USD 1時間足（1H・直近12本）
    {data_dict["1h"].tail(12).to_string(index=False)}

    ■ 4. XAU/USD 15分足（15M・直近12本）
    {data_dict["15min"].tail(12).to_string(index=False)}

    ■ 5. XAU/USD 5分足（5M・直近12本）
    {data_dict["5min"].tail(12).to_string(index=False)}

    ■ 6. 相関データ：米ドル指数(DXY) 1時間足（直近5本）
    {df_dxy[["datetime", "close", "EMA20", "RSI14"]].tail(5).to_string(index=False)}

    ■ 7. 相関データ：銀(XAG/USD) 1時間足（直近5本）
    {df_xag[["datetime", "close", "EMA20", "RSI14"]].tail(5).to_string(index=False)}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        ),
    )
    return response.text

# ==========================================
# 8. 仮想トレード新規登録＆LINE通知
# ==========================================
def register_new_trade(logs: list, result_text: str, current_price: float) -> tuple[list, bool]:
    if "【買い目線】" in result_text:
        direction = "BUY"
    elif "【売り目線】" in result_text:
        direction = "SELL"
    else:
        return logs, False

    entry_match = re.search(r"エントリー価格[：:]\s*([0-9]+\.?[0-9]*)", result_text)
    tp_match = re.search(r"利確（TP）[：:]\s*([0-9]+\.?[0-9]*)", result_text)
    sl_match = re.search(r"損切り（SL）[：:]\s*([0-9]+\.?[0-9]*)", result_text)

    if entry_match and tp_match and sl_match:
        entry_val = float(entry_match.group(1))
        tp_val = float(tp_match.group(1))
        sl_val = float(sl_match.group(1))

        # すでに同じ方向・近い価格帯の有効ポジションがある場合は重複登録をスキップ
        for t in logs:
            if t["status"] in ["PENDING", "OPEN"] and t["direction"] == direction and abs(t["entry"] - entry_val) < 5.0:
                print("ℹ️ すでに同方向・近接価格のポジションを監視中のため、新規登録と重複通知をスキップします。")
                return logs, False

        initial_status = "OPEN" if abs(current_price - entry_val) <= 2.0 else "PENDING"
        new_trade = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "entry": entry_val,
            "tp": tp_val,
            "sl": sl_val,
            "status": initial_status,
            "pnl": 0.0
        }
        logs.append(new_trade)
        print(f"📝 新規仮想トレードを記録: {direction} Entry={entry_val} TP={tp_val} SL={sl_val} ({initial_status})")
        return logs, True
    return logs, False

def send_line(message: str, current_price: float, feedback_text: str):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    summary_header = feedback_text.split("\n")[0]
    text = f"🏆 XAU/USD プロ統合AI戦略\n（現在値: {current_price}）\n{summary_header}\n\n{message}"
    
    payload = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": text[:4900]}]
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"LINE送信エラー: {res.status_code} {res.text}")

# ==========================================
# 9. メイン実行処理
# ==========================================
if __name__ == "__main__":
    now = datetime.now(JST)
    weekday = now.weekday()
    if (weekday == 5 and now.hour >= 7) or (weekday == 6) or (weekday == 0 and now.hour < 6):
        print("💤 FX市場休場中のため処理をスキップします。")
        exit(0)

    print("1/4 経済指標カレンダー＆最新金融ニュースを取得中...")
    calendar_text = get_economic_calendar_jst()
    news_text = get_google_news_rss()

    print("2/4 XAU/USD 全時間足（1D〜5M）データを取得中...")
    timeframes = ["1day", "4h", "1h", "15min", "5min"]
    data_dict = {}
    for tf in timeframes:
        out_size = 100 if tf == "15min" else 60
        data_dict[tf] = get_market_data("XAU/USD", tf, outputsize=out_size)
        time.sleep(1.5)

    print("3/4 相関資産（DXY・XAG/USD）を取得中...")
    try:
        df_dxy = get_market_data("DXY", "1h", outputsize=30)
        time.sleep(1.5)
    except Exception:
        df_dxy = data_dict["1h"]

    try:
        df_xag = get_market_data("XAG/USD", "1h", outputsize=30)
    except Exception:
        df_xag = data_dict["1h"]

    current_price = data_dict["5min"]["close"].iloc[-1]
    key_levels_text = extract_key_levels(data_dict["1day"], data_dict["1h"], current_price)

    # 過去の答え合わせ
    logs = load_trade_logs()
    logs, feedback_text, active_count = evaluate_past_trades(logs, data_dict["15min"])
    print("\n--- AI自己検証・成績レポート ---")
    print(feedback_text)
    print("--------------------------------\n")

    print("4/4 全データを統合してAIストラテジストが分析中...")
    result = analyze_master_strategy(
        data_dict, df_dxy, df_xag, key_levels_text, calendar_text, news_text, feedback_text
    )
    
    print("\n--- AI総合戦略レポート ---")
    print(result)
    print("--------------------------\n")

    if "【買い目線】" in result or "【売り目線】" in result:
        logs, is_new_signal = register_new_trade(logs, result, current_price)
        save_trade_logs(logs)
        if is_new_signal:
            send_line(result, current_price, feedback_text)
            print("🔥 新規エントリー戦略をLINEへ送信＆検証ログに記録しました！")
    else:
        save_trade_logs(logs)
        print("⏸️ 判定が「様子見」（重要指標警戒等）のため、LINE通知をスキップしました。")

