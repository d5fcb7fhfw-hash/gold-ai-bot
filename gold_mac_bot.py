export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("OK", { status: 200 });
    }

    const body = await request.json();
    const events = body.events || [];

    for (const event of events) {
      if (event.type === "message" && event.message.type === "text") {
        const userText = event.message.text.trim();

        if (userText.includes("戦績") || userText.includes("成績")) {
          const replyText = await buildStatsReport(env);
          await replyToLine(event.replyToken, replyText, env);
        } else if (userText.includes("今は") || userText.includes("今どう") || userText.includes("分析")) {
          const replyText = await scanGoldPriorityNow(env);
          await replyToLine(event.replyToken, replyText, env);
        }
      }
    }

    return new Response("OK", { status: 200 });
  },
};

// ==========================================
// 世界情勢・FRB・地政学ニュース取得（RSS）
// ==========================================
async function fetchWorldNews() {
  try {
    const rssUrl =
      "https://news.google.com/rss/search?q=Gold+price+OR+XAUUSD+OR+Federal+Reserve+OR+geopolitics&hl=en-US&gl=US&ceid=US:en";
    const res = await fetch(rssUrl);
    if (!res.ok) return "ニュース取得スキップ";
    const xml = await res.text();
    const titles = [...xml.matchAll(/<item>[\s\S]*?<title>(.*?)<\/title>/g)]
      .slice(0, 8)
      .map((m) => "・" + m[1].replace(/<!\[CDATA\[|\]\]>/g, "").trim());
    return titles.length > 0 ? titles.join("\n") : "最新ニュースなし";
  } catch {
    return "ニュース取得スキップ";
  }
}

// ==========================================
// ゴールド最優先・ハイブリッド即時判断（「今は」用）
// ==========================================
async function scanGoldPriorityNow(env) {
  const tdKey = (env.TWELVE_DATA_API_KEY || "").trim();
  const geminiKey = (env.GEMINI_API_KEY || "").trim();

  if (!tdKey || !geminiKey) {
    return "⚠️ 環境変数エラー：TWELVE_DATA_API_KEY または GEMINI_API_KEY が未設定です。";
  }

  const allSymbols = ["XAU/USD", "USD/JPY", "EUR/USD", "GBP/JPY", "GBP/USD", "EUR/JPY", "AUD/USD"];
  const subSymbols = ["USD/JPY", "EUR/USD", "GBP/JPY", "GBP/USD", "EUR/JPY", "AUD/USD"];

  try {
    const [res1h, res4h, resGold15m, worldNews] = await Promise.all([
      fetch(`https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(allSymbols.join(","))}&interval=1h&outputsize=35&apikey=${tdKey}`),
      fetch(`https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(allSymbols.join(","))}&interval=4h&outputsize=35&apikey=${tdKey}`),
      fetch(`https://api.twelvedata.com/time_series?symbol=XAU%2FUSD&interval=15min&outputsize=35&apikey=${tdKey}`),
      fetchWorldNews(),
    ]);

    const json1h = await res1h.json();
    const json4h = await res4h.json();
    const jsonGold15m = await resGold15m.json();

    if (json1h.code === 429 || json4h.code === 429 || jsonGold15m.code === 429) {
      return "⏳ API制限中です。1分ほど待ってからもう一度「今は」と送ってください。";
    }

    let usdSummary = "米ドル強弱データ不足";
    const uj = json1h["USD/JPY"]?.values;
    const eu = json1h["EUR/USD"]?.values;
    if (uj && eu && uj.length >= 4 && eu.length >= 4) {
      const jpyChg = ((parseFloat(uj[0].close) - parseFloat(uj[3].close)) / parseFloat(uj[3].close)) * 100;
      const eurChg = ((parseFloat(eu[0].close) - parseFloat(eu[3].close)) / parseFloat(eu[3].close)) * 100;
      const diff = jpyChg - eurChg;
      usdSummary =
        diff > 0.15
          ? "米ドル全面高（ゴールドに逆風）"
          : diff < -0.15
          ? "米ドル全面安（ゴールドに追い風）"
          : "米ドル中立";
    }

    const g4h = calcMetrics(json4h["XAU/USD"]);
    const g1h = calcMetrics(json1h["XAU/USD"]);
    const g15m = calcMetrics(jsonGold15m);

    let goldText = "【XAU/USD データ取得失敗】";
    if (g4h && g1h && g15m) {
      const maxPullback = (g15m.atr14 * 0.45).toFixed(2);
      const minSl = (g15m.atr14 * 1.30).toFixed(2);
      goldText = `
★★★【最優先：XAU/USD（ゴールド）4H・1H・15M＋マクロ環境】★★★
・現在値: ${g15m.latest.toFixed(2)}
・米ドル相関環境: ${usdSummary}
・許容指値幅（未約定防止）: ±${maxPullback} ドル以内
・最低SL幅（ヒゲ狩り防止）: ${minSl} ドル以上（1.3×ATR14）
・4H: EMA20=${g4h.ema20.toFixed(2)}, RSI14=${g4h.rsi14}, 20本高値=${g4h.high20.toFixed(2)}, 安値=${g4h.low20.toFixed(2)}
・1H: EMA20=${g1h.ema20.toFixed(2)}, RSI14=${g1h.rsi14}, 20本高値=${g1h.high20.toFixed(2)}, 安値=${g1h.low20.toFixed(2)}
・15M: EMA20=${g15m.ema20.toFixed(2)}, RSI14=${g15m.rsi14}, ATR14=${g15m.atr14.toFixed(2)}
・15M直近3本ローソク足:
${g15m.recentCandles}`;
    }

    const subAligned = [];
    for (const sym of subSymbols) {
      const d1 = calcMetrics(json1h[sym]);
      const d4 = calcMetrics(json4h[sym]);
      if (!d1 || !d4) continue;
      const t4 = d4.latest > d4.ema20 ? "UP" : "DOWN";
      const t1 = d1.latest > d1.ema20 ? "UP" : "DOWN";
      if (t4 !== t1) continue;

      const digits = sym.includes("JPY") ? 3 : 5;
      subAligned.push(
        `・${sym}（${t1}一致）: 現在値=${d1.latest.toFixed(digits)}, 許容指値幅=±${(d1.atr14 * 0.45).toFixed(digits)}, 最低SL幅=${(d1.atr14 * 1.2).toFixed(digits)}, 4H_RSI=${d4.rsi14}, 1H_RSI=${d1.rsi14}`
      );
    }

    const nowJst = new Date(Date.now() + 9 * 60 * 60 * 1000)
      .toISOString()
      .replace("T", " ")
      .slice(0, 16) + " JST";

    const prompt = `
現在時刻: ${nowJst}

【最新の世界情勢・地政学・FRBニュース】
${worldNews}

${goldText}

【サブ6銘柄（4H・1Hトレンド一致のみ）】
${subAligned.length > 0 ? subAligned.join("\n") : "サブ銘柄にトレンド一致なし"}

あなたは世界最高峰のトレードストラテジストです。
「ハイブリッド・ブレイク＆押し目戦略（強い初動は成行優先、緩やかなトレンドは0.45×ATR以内の浅い指値）」に基づき、最も信頼できる判断を返してください。
1. **必ず最初に「XAU/USD（ゴールド）」の判定（🔥即エントリー / 👀接近中・監視ライン提示 / 🛡️様子見 のいずれか）を回答すること。**
2. 様子見や接近中の場合も、「〇〇円を上抜けたら成行買い」「〇〇円まで下がったら反発確認で買い」という具体的な監視ラインを必ず提示すること。
3. サブ銘柄に強いチャンスがあれば1つだけ「サブ特選」として併記すること。

【出力フォーマット】
⚡ ゴールド最優先・リアルタイム精密判断
（${nowJst}）

■ XAU/USD 判定：【🔥即エントリー】or【👀接近中・監視待機】or【🛡️様子見】
・現在値：〇〇.〇〇
・注文方法：成行 / Buy Limit / Sell Limit / ライン待ち
・基準エントリー：〇〇.〇〇
・利確（TP）：〇〇.〇〇
・損切り（SL）：〇〇.〇〇
・RR比：1 : 〇.〇
・発動条件：（今すぐ入るか、どの価格を抜けたら入るか）
・統合根拠（テクニカル×世界情勢）：（3行以内で簡潔に）

（※サブ銘柄に強いチャンスがある場合のみ追記）
🎯 サブ特選：【銘柄名】買い or 売り
・エントリー：〇〇 / TP：〇〇 / SL：〇〇（RR 1:〇.〇）
・根拠：（1行）
`;

    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key=${geminiKey}`;
    const aiRes = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.1 },
      }),
    });

    if (!aiRes.ok) {
      const errText = await aiRes.text();
      return `⚠️ AI分析エラー (${aiRes.status}): ${errText.slice(0, 150)}`;
    }

    const aiJson = await aiRes.json();
    return aiJson.candidates?.[0]?.content?.parts?.[0]?.text || "分析結果の生成に失敗しました。";
  } catch (err) {
    return `⚠️ スキャンエラー: ${err.message}`;
  }
}

function calcMetrics(item) {
  if (!item || !item.values || item.values.length < 25) return null;
  const bars = item.values.slice().reverse().map((b) => ({
    dt: b.datetime,
    open: parseFloat(b.open),
    high: parseFloat(b.high),
    low: parseFloat(b.low),
    close: parseFloat(b.close),
  }));

  const latest = bars[bars.length - 1].close;
  const k = 2 / 21;
  let ema20 = bars[0].close;
  for (let i = 1; i < bars.length; i++) {
    ema20 = bars[i].close * k + ema20 * (1 - k);
  }

  let gains = 0, losses = 0;
  for (let i = bars.length - 14; i < bars.length; i++) {
    const diff = bars[i].close - bars[i - 1].close;
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }
  const rsi14 = (100 - 100 / (1 + (losses === 0 ? 100 : gains / losses))).toFixed(1);

  let trSum = 0;
  for (let i = bars.length - 14; i < bars.length; i++) {
    const tr = Math.max(
      bars[i].high - bars[i].low,
      Math.abs(bars[i].high - bars[i - 1].close),
      Math.abs(bars[i].low - bars[i - 1].close)
    );
    trSum += tr;
  }
  const atr14 = trSum / 14;

  const high20 = Math.max(...bars.slice(-20).map((b) => b.high));
  const low20 = Math.min(...bars.slice(-20).map((b) => b.low));

  const recentCandles = bars.slice(-3).map(
    (b) => `  ${b.dt.slice(11, 16)} 始:${b.open} 高:${b.high} 安:${b.low} 終:${b.close}`
  ).join("\n");

  return { latest, ema20, rsi14, atr14, high20, low20, recentCandles };
}

// ==========================================
// 戦績表示（「戦績を表示」用）
// ==========================================
async function buildStatsReport(env) {
  const repo = (env.GITHUB_REPO || "").trim();
  const token = (env.GITHUB_TOKEN || "").trim();

  if (!repo || !token) {
    return "⚠️ 環境変数エラー：GITHUB_REPO または GITHUB_TOKEN が未設定です。";
  }

  const url = `https://api.github.com/repos/${repo}/contents/trade_log.json`;
  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
      "User-Agent": "Cloudflare-Worker-Gold-Bot",
      Accept: "application/vnd.github.v3+json",
    },
  });

  if (!res.ok) {
    if (res.status === 404) return "📊 まだ仮想トレードの戦績データがありません。";
    return `⚠️ GitHubデータ取得エラー (Status: ${res.status})`;
  }

  const data = await res.json();
  const binaryString = atob(data.content.replace(/\n/g, ""));
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const logs = JSON.parse(new TextDecoder("utf-8").decode(bytes));

  if (!Array.isArray(logs) || logs.length === 0) {
    return "📊 まだ仮想トレードの記録がありません。";
  }

  const wins = logs.filter((t) => t.status === "WIN");
  const losses = logs.filter((t) => t.status === "LOSS");
  const active = logs.filter((t) => t.status === "PENDING" || t.status === "OPEN");
  const expired = logs.filter((t) => t.status === "EXPIRED");

  const settledCount = wins.length + losses.length;
  const winRate = settledCount > 0 ? ((wins.length / settledCount) * 100).toFixed(1) : "0.0";

  const recentLines = logs.slice(-8).reverse().map((t) => {
    const icon = {
      WIN: "✅利確",
      LOSS: "❌損切",
      OPEN: "⏳保有中",
      PENDING: "🕒指値待",
      EXPIRED: "⚪️未約定",
    }[t.status] || t.status;
    const sym = t.symbol || "XAU/USD";
    const pnlStr = t.status === "WIN" || t.status === "LOSS" ? ` (${t.pnl > 0 ? "+" : ""}${t.pnl})` : "";
    return `${icon} [${sym}] ${t.timestamp.slice(5, 16)} ${t.direction} @${t.entry}${pnlStr}`;
  });

  return `📊 ゴールド最優先・精密AIトレード戦績

■ 通算トータル成績
・決済戦績：${wins.length}勝 ${losses.length}敗（勝率 ${winRate}%）
・保有/待機：${active.length}件 / 未約定：${expired.length}件

■ 直近のシグナル履歴（最新8件）
${recentLines.join("\n")}`;
}

async function replyToLine(replyToken, text, env) {
  const lineToken = (env.LINE_CHANNEL_ACCESS_TOKEN || "").trim();
  await fetch("https://api.line.me/v2/bot/message/reply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${lineToken}`,
    },
    body: JSON.stringify({
      replyToken: replyToken,
      messages: [{ type: "text", text: text.slice(0, 4900) }],
    }),
  });
}
