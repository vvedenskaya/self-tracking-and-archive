"""Relationship Constellations — every chat is a star in a night sky.

Reads telegram_messages.parquet and writes a single self-contained HTML file
(data embedded, vanilla JS canvas, no network needed) to visualizations/.

Visual encoding:
  x          when the relationship lived (center of mass of its messages)
  y          who carried the conversation (top = they wrote more, bottom = me)
  size       total messages (log scale)
  warmth     lifespan: long companions glow gold, brief encounters burn blue
  brightness recency: still-active stars shine, faded ones dim into the dark

Hover preview per star: monthly volume histogram (me vs them), signature
words (TF-IDF against all other chats, lemmatized RU+EN), and a tone score
from a small RU+EN sentiment lexicon + emoticons. All computed locally.

Usage: python art/constellation.py
"""
import json
import re
import sys
from collections import Counter
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.word_histograms import EN_STOP, RU_STOP, TOKEN_RE, URL_RE, lemmatize_ru

OUT = config.VISUALIZATIONS / "constellation.html"

CYRILLIC_RE = re.compile(r"[а-яё]")

# tone lexicons: matched against lemmas (RU) / raw lowercase tokens (EN).
# Deliberately small and transparent — a rough aggregate gauge, not real NLP.
RU_POS = set(
    """хороший классный отличный прекрасный замечательный чудесный любить
    нравиться обожать рад радость радоваться счастливый счастье круто здорово
    супер кайф приятный приятно весёлый весело смешной смеяться улыбаться
    милый добрый красивый умный интересный успех ура поздравлять благодарить
    целовать обнимать скучать""".split()
)
RU_NEG = set(
    """плохой ужасный ужас кошмар ненавидеть злой злиться бесить раздражать
    грустный грусть печальный печально плакать страшный страшно страх бояться
    болеть боль больно устать усталый тяжело сложно проблема обидно обида
    жаль хуже отвратительный дурацкий тупой говно дерьмо хрень фигня провал
    умирать смерть болезнь""".split()
)
EN_POS = set(
    """good great love loved awesome amazing nice happy glad thanks thank cool
    perfect wonderful beautiful excellent fun enjoy enjoyed excited best win
    won congrats sweet cute""".split()
)
EN_NEG = set(
    """bad terrible awful hate hated sad angry mad annoying annoyed problem
    problems hard tired sick pain hurt worst fail failed sorry worried worry
    scared fear afraid lose lost cry crying broken""".split()
)
POS_EMO = re.compile(r"\){2,}|:\)|:D|❤|💛|😂|😍|🥰|😊|😄|👍|🎉")
NEG_EMO = re.compile(r"\({2,}|:\(|😢|😭|💔|😡|😞|👎")

MIN_TONE_HITS = 20  # below this the tone score is too noisy to show
N_THEME_WORDS = 6


def month_series(sub: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Monthly message counts (mine, theirs), zero-filled between first and last."""
    per = sub.groupby([sub.ts_local.dt.to_period("M"), "is_me"]).size().unstack(fill_value=0)
    idx = pd.period_range(per.index.min(), per.index.max(), freq="M")
    per = per.reindex(idx, fill_value=0)
    me = per[True].astype(int).tolist() if True in per.columns else [0] * len(idx)
    them = per[False].astype(int).tolist() if False in per.columns else [0] * len(idx)
    return me, them


def text_features(texts) -> tuple[Counter, int, int]:
    """One pass over a chat's messages: theme word counts + pos/neg tone hits."""
    counts: Counter = Counter()
    pos = neg = 0
    for text in texts:
        if not text:
            continue
        pos += len(POS_EMO.findall(text))
        neg += len(NEG_EMO.findall(text))
        for w in TOKEN_RE.findall(URL_RE.sub(" ", text.lower())):
            if len(w) < 3:
                continue
            if CYRILLIC_RE.match(w):
                lemma = lemmatize_ru(w)
                if lemma in RU_POS:
                    pos += 1
                elif lemma in RU_NEG:
                    neg += 1
                if len(lemma) >= 3 and lemma not in RU_STOP:
                    counts[lemma] += 1
            else:
                if w in EN_POS:
                    pos += 1
                elif w in EN_NEG:
                    neg += 1
                if w not in EN_STOP:
                    counts[w] += 1
    return counts, pos, neg


def build_word_index(theme_counts: list[Counter]) -> dict[str, list]:
    """Inverted index over all messages: lemma -> [[star_idx, count], ...].

    Powers full-history word search in the browser without embedding any
    actual message text. Words seen only once in a chat are dropped.
    """
    index: dict[str, list] = {}
    for i, counts in enumerate(theme_counts):
        for w, c in counts.items():
            if c >= 2:
                index.setdefault(w, []).append([i, c])
    return index


def build_stars() -> tuple[list[dict], dict, dict]:
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    now = df.ts_utc.max()

    stars, theme_counts = [], []
    groups = [(name, sub) for name, sub in df.groupby("chat_name") if len(sub) >= 3]
    for i, (name, sub) in enumerate(groups):
        if i % 100 == 0:
            print(f"  {i}/{len(groups)} chats...")
        ts = sub.ts_utc
        span_days = max((ts.max() - ts.min()).days, 1)
        me, them = month_series(sub)
        counts, pos, neg = text_features(sub.text)
        tone = round((pos - neg) / (pos + neg), 2) if pos + neg >= MIN_TONE_HITS else None
        theme_counts.append(counts)
        stars.append(
            {
                "name": name,
                "n": int(len(sub)),
                "center": ts.mean().isoformat(),
                "first": ts.min().strftime("%Y-%m"),
                "last": ts.max().strftime("%Y-%m"),
                "spanDays": int(span_days),
                "idleDays": int((now - ts.max()).days),
                "myShare": round(float(sub.is_me.mean()), 3),
                "me": me,
                "them": them,
                "tone": tone,
            }
        )

    # signature words: TF-IDF of each chat against all the others
    docfreq: Counter = Counter()
    for counts in theme_counts:
        docfreq.update(counts.keys())
    n_docs = len(theme_counts)
    for star, counts in zip(stars, theme_counts):
        scored = sorted(
            ((c * log(n_docs / docfreq[w]), w) for w, c in counts.items() if c >= 3),
            reverse=True,
        )
        star["words"] = [w for _, w in scored[:N_THEME_WORDS]]

    meta = {
        "tMin": df.ts_utc.min().isoformat(),
        "tMax": now.isoformat(),
        "totalMessages": int(len(df)),
        "totalChats": len(stars),
    }
    return stars, meta, build_word_index(theme_counts)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Relationship Constellations</title>
<style>
  html, body { margin: 0; height: 100%; background: #060610; overflow: hidden;
               font: 13px/1.45 "Helvetica Neue", Helvetica, Arial, sans-serif; }
  canvas { display: block; }
  #tip { position: fixed; pointer-events: none; background: rgba(12,12,26,.95);
         color: #e8e4d8; padding: 12px 14px; border: 1px solid #3a3650;
         border-radius: 6px; opacity: 0; transition: opacity .15s; width: 248px; }
  #tip b { color: #e8b84b; font-size: 14px; }
  #tip .dim { color: #9a96a8; }
  #tip canvas { display: block; margin: 8px 0 2px; }
  #tip .words { margin-top: 6px; line-height: 1.7; }
  #tip .word { display: inline-block; background: rgba(232,184,75,.13); color: #e8b84b;
               border-radius: 4px; padding: 0 6px; margin-right: 4px; font-size: 12px; }
  #tip .tonebar { height: 5px; border-radius: 3px; margin-top: 8px; position: relative;
                  background: linear-gradient(90deg, #7a4a6a, #55516e, #4a8a7a); }
  #tip .tonedot { position: absolute; top: -3px; width: 11px; height: 11px;
                  border-radius: 50%; background: #e8e4d8; border: 2px solid #101018; }
  #tip .tonelabel { display: flex; justify-content: space-between; color: #66627a;
                    font-size: 11px; margin-top: 3px; }
  canvas { cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #panel { position: fixed; left: 20px; top: 20px; width: 264px; color: #9a96a8;
           background: rgba(9,9,20,.88); border: 1px solid #2e2a44; border-radius: 10px;
           padding: 14px 16px; backdrop-filter: blur(5px); }
  #panel h1 { font-size: 15px; font-weight: 600; color: #e8e4d8; margin: 0; letter-spacing: .2px; }
  #subtitle { font-size: 11.5px; color: #66627a; margin: 3px 0 12px; }
  #search { background: rgba(20,20,38,.9); color: #e8e4d8; border: 1px solid #3a3650;
            border-radius: 6px; padding: 8px 11px; width: 100%; box-sizing: border-box;
            outline: none; font: 13px "Helvetica Neue", Helvetica, Arial, sans-serif;
            transition: border-color .2s; }
  #search:focus { border-color: #e8b84b; }
  #search::placeholder { color: #66627a; }
  #count { color: #e8b84b; margin-top: 5px; font-size: 12px; min-height: 14px; }
  #results { margin-top: 4px; max-height: 44vh; overflow-y: auto; }
  #results::-webkit-scrollbar { width: 4px; }
  #results::-webkit-scrollbar-thumb { background: #2e2a44; border-radius: 2px; }
  .row { padding: 8px 4px 7px; border-top: 1px solid #1e1a30; }
  .row:hover { background: rgba(232,184,75,.06); }
  .rowname { display: flex; justify-content: space-between; gap: 8px;
             font-size: 12.5px; color: #e8e4d8; }
  .rowname .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .rowstats { color: #66627a; font-size: 11px; margin: 1px 0 4px; }
  .row canvas { display: block; }
  .more { color: #56526a; font-size: 11px; padding: 7px 4px 2px; border-top: 1px solid #1e1a30; }
  #help { font-size: 11px; line-height: 1.75; color: #76728a; margin-top: 10px;
          border-top: 1px solid #262236; padding-top: 9px; }
  #help b { color: #b0acc4; font-weight: 500; }
  #help .keys { color: #56526a; margin-top: 4px; }
  #axis { position: fixed; right: 20px; top: 50%; transform: translateY(-50%);
          text-align: right; color: #8a86a8; font-size: 12px; line-height: 1.5;
          background: rgba(9,9,20,.75); border: 1px solid #2e2a44;
          border-radius: 8px; padding: 10px 13px; pointer-events: none; }
</style>
</head>
<body>
<canvas id="sky"></canvas>
<div id="panel">
  <h1>Relationship Constellations</h1>
  <div id="subtitle">__SUBTITLE__</div>
  <input id="search" type="text" placeholder="search people or words…" autocomplete="off">
  <div id="count"></div>
  <div id="results"></div>
  <div id="help">
    <b>x</b> — when the relationship lived<br>
    <b>y</b> — who wrote more (top: them, bottom: me)<br>
    <b>size</b> — total messages · <b>gold</b> — long companion<br>
    <b>blue</b> — brief encounter · <b>bright</b> — recently active
    <div class="keys">scroll — zoom · drag — pan · dbl-click — reset · esc — clear</div>
  </div>
</div>
<div id="tip"></div>
<div id="axis">↑ they wrote<br>most of it<br><br>balanced<br><br>I wrote<br>most of it ↓</div>
<script>
const STARS = __STARS__;
const META = __META__;
const WORDS = __WORDS__;  // lemma -> [[star index, count], ...] over all messages

const cv = document.getElementById("sky"), cx = cv.getContext("2d");
const tip = document.getElementById("tip");
let W, H, PADX, placed = [];

const tMin = Date.parse(META.tMin), tMax = Date.parse(META.tMax);

// camera: stars live in world coords, screen = world * k + offset
const view = { k: 1, ox: 0, oy: 0, tk: 1, tox: 0, toy: 0 };

function clampView() {
  view.tk = Math.min(Math.max(view.tk, 1), 30);
  view.tox = Math.min(0, Math.max(W * (1 - view.tk), view.tox));
  view.toy = Math.min(0, Math.max(H * (1 - view.tk), view.toy));
}

function resize() {
  W = cv.width = innerWidth * devicePixelRatio;
  H = cv.height = innerHeight * devicePixelRatio;
  cv.style.width = innerWidth + "px";
  cv.style.height = innerHeight + "px";
  PADX = 90 * devicePixelRatio;
  clampView();
  layout();
}

function layout() {
  const padX = 90 * devicePixelRatio, padY = 110 * devicePixelRatio;
  placed = STARS.map(s => {
    const fx = (Date.parse(s.center) - tMin) / (tMax - tMin);
    // myShare 0 -> they wrote everything (top), 1 -> I did (bottom)
    const fy = s.myShare;
    // deterministic jitter so dense regions breathe
    let h = 0; for (const ch of s.name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    const jx = ((h % 1000) / 1000 - .5) * 26 * devicePixelRatio;
    const jy = (((h >> 10) % 1000) / 1000 - .5) * 26 * devicePixelRatio;
    const r = (2 + Math.log2(s.n) * 1.6) * devicePixelRatio;
    // warmth: lifespan in years, 0..1 over ~4 years
    const warmth = Math.min(s.spanDays / 1460, 1);
    // brightness: fades over two idle years
    const fade = Math.max(0.12, 1 - s.idleDays / 730);
    const hx = padX + fx * (W - 2 * padX) + jx;
    const hy = padY + fy * (H - 2 * padY) + jy;
    return { s, r, warmth, fade,
      homeX: hx, homeY: hy,   // where the star belongs in the sky
      x: hx, y: hy,           // animated current position
      tx: hx, ty: hy,         // animation target
      dim: 1, tdim: 1,        // 1 = normal, <1 = dimmed by search
      tw: (h % 628) / 100 };
  });
  applySearch();
}

// ---- fuzzy search: matching stars cluster into a spiral, the rest dim ----
const searchEl = document.getElementById("search");
const countEl = document.getElementById("count");

// subsequence match: every query char appears in order; tighter = better score
function fuzzyScore(query, name) {
  const q = query.toLowerCase(), t = name.toLowerCase();
  let qi = 0, score = 0, last = -1;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += (last === ti - 1) ? 3 : 1;  // reward consecutive runs
      last = ti; qi++;
    }
  }
  return qi === q.length ? score + (t.startsWith(q) ? 5 : 0) : -1;
}

// full-history word lookup: every indexed word starting with the query
// (the index stores lemmas, so "работ" finds работать in every chat)
function indexMatches(q) {
  const m = new Map();  // star index -> total mentions
  if (q.length < 3) return m;
  for (const w in WORDS) {
    if (w.startsWith(q)) {
      for (const [i, c] of WORDS[w]) m.set(i, (m.get(i) || 0) + c);
    }
  }
  return m;
}

const resultsEl = document.getElementById("results");
let highlightStar = null;
const MAX_ROWS = 8;

function toneColor(tone) {
  return tone === null ? "#56526a" : tone > 0.25 ? "#5fc8a8" : tone < -0.1 ? "#d07a9a" : "#9a96a8";
}

// side-by-side comparison of the matched relationships, sorted by relevance
function renderResults(hits) {
  resultsEl.innerHTML = "";
  highlightStar = null;
  for (const [_, p] of hits.slice(0, MAX_ROWS)) {
    const s = p.s;
    const row = document.createElement("div");
    row.className = "row";
    const tone = s.tone === null ? "" :
      `<span style="color:${toneColor(s.tone)}">${s.tone > 0 ? "+" : ""}${s.tone}</span>`;
    const mentions = p.mentions ? ` · <span style="color:#e8b84b">${p.mentions}× the word</span>` : "";
    row.innerHTML =
      `<div class="rowname"><span class="nm">${s.name}</span>${tone}</div>` +
      `<div class="rowstats">${s.n.toLocaleString()} msgs · ${Math.round(s.myShare * 100)}% me · ${s.first} — ${s.last}${mentions}</div>`;
    row.appendChild(sparkline(s, 224, 26));
    row.addEventListener("mouseenter", () => { highlightStar = p; });
    row.addEventListener("mouseleave", () => { highlightStar = null; });
    resultsEl.appendChild(row);
  }
  if (hits.length > MAX_ROWS) {
    const more = document.createElement("div");
    more.className = "more";
    more.textContent = `+ ${hits.length - MAX_ROWS} more — hover the cluster`;
    resultsEl.appendChild(more);
  }
}

function applySearch() {
  const q = searchEl.value.trim();
  if (!q) {
    for (const p of placed) { p.tx = p.homeX; p.ty = p.homeY; p.tdim = 1; }
    countEl.textContent = "";
    resultsEl.innerHTML = "";
    highlightStar = null;
    return;
  }
  const ql = q.toLowerCase();
  const mentions = indexMatches(ql);
  const hits = [];
  placed.forEach((p, i) => {
    const nameSc = fuzzyScore(ql, p.s.name);
    const cnt = mentions.get(i) || 0;
    p.mentions = cnt || null;
    // people outrank words; word matches rank by how often the word was said
    const sc = nameSc >= 0 ? 1000 + nameSc : cnt > 0 ? Math.log2(cnt + 1) : -1;
    if (sc >= 0) hits.push([sc, p]); else { p.tdim = 0.08; p.tx = p.homeX; p.ty = p.homeY; }
  });
  hits.sort((a, b) => b[0] - a[0] || b[1].s.n - a[1].s.n);
  // phyllotaxis spiral at the center of the current viewport, in world coords
  const ccx = (W / 2 - view.tox) / view.tk, ccy = (H / 2 - view.toy) / view.tk;
  const spread = 13 * devicePixelRatio * Math.sqrt(Math.max(hits.length, 8)) / view.tk;
  hits.forEach(([_, p], i) => {
    const ang = i * 2.39996;  // golden angle
    const rad = spread * Math.sqrt((i + 0.5) / Math.max(hits.length, 1));
    p.tx = ccx + Math.cos(ang) * rad;
    p.ty = ccy + Math.sin(ang) * rad;
    p.tdim = 1.6;  // matched stars glow brighter than normal
  });
  countEl.textContent = hits.length
    ? `${hits.length} ${hits.length === 1 ? "match" : "matches"}`
    : "no matches";
  renderResults(hits);
}

searchEl.addEventListener("input", applySearch);
searchEl.addEventListener("keydown", e => {
  if (e.key === "Escape") { searchEl.value = ""; applySearch(); searchEl.blur(); }
});

function starColor(warmth, alpha) {
  const r = Math.round(140 + warmth * 115);
  const g = Math.round(170 + warmth *  35);
  const b = Math.round(255 - warmth * 175);
  return `rgba(${r},${g},${b},${alpha})`;
}

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const worldX = ts => PADX + (ts - tMin) / (tMax - tMin) * (W - 2 * PADX);

function draw(t) {
  const dpr = devicePixelRatio;
  // ease the camera toward its target
  view.k += (view.tk - view.k) * .14;
  view.ox += (view.tox - view.ox) * .14;
  view.oy += (view.toy - view.oy) * .14;
  cx.clearRect(0, 0, W, H);

  // time gridlines: years always; months fade in once there is room for them
  const monthPx = (worldX(Date.UTC(2020, 1, 1)) - worldX(Date.UTC(2020, 0, 1))) * view.k;
  const y0 = new Date(tMin).getFullYear(), y1 = new Date(tMax).getFullYear();
  for (let y = y0; y <= y1; y++) {
    for (let m = 0; m < 12; m++) {
      const ts = Date.UTC(y, m, 1);
      if (ts < tMin || ts > tMax) continue;
      const isYear = m === 0;
      if (!isYear && monthPx < 45 * dpr) continue;
      const px = worldX(ts) * view.k + view.ox;
      if (px < -60 || px > W + 60) continue;
      cx.strokeStyle = isYear ? "rgba(110,104,160,.5)" : "rgba(110,104,160,.16)";
      cx.setLineDash([4 * dpr, 6 * dpr]);
      cx.beginPath(); cx.moveTo(px, 0); cx.lineTo(px, H); cx.stroke();
      cx.setLineDash([]);
      if (isYear) {
        cx.fillStyle = "#aaa6c8";
        cx.font = `600 ${13 * dpr}px Helvetica Neue`;
        cx.fillText(y, px + 6 * dpr, H - 14 * dpr);
      } else if (monthPx > 70 * dpr) {
        cx.fillStyle = "#66627a";
        cx.font = `${11 * dpr}px Helvetica Neue`;
        cx.fillText(MONTHS[m], px + 5 * dpr, H - 14 * dpr);
      }
    }
  }

  const rScale = Math.sqrt(view.k);  // dots grow slower than space spreads
  for (const p of placed) {
    // ease toward search targets (cluster / home / dim)
    p.x += (p.tx - p.x) * 0.07;
    p.y += (p.ty - p.y) * 0.07;
    p.dim += (p.tdim - p.dim) * 0.09;
    const px = p.x * view.k + view.ox, py = p.y * view.k + view.oy;
    const rr = p.r * rScale;
    if (px < -rr * 4 || px > W + rr * 4 || py < -rr * 4 || py > H + rr * 4) continue;
    const twinkle = .85 + .15 * Math.sin(t / 900 + p.tw);
    const a = Math.min(1, p.fade * twinkle * p.dim);
    const glow = cx.createRadialGradient(px, py, 0, px, py, rr * 4);
    glow.addColorStop(0, starColor(p.warmth, .5 * a));
    glow.addColorStop(1, "rgba(0,0,0,0)");
    cx.fillStyle = glow;
    cx.beginPath(); cx.arc(px, py, rr * 4, 0, 7); cx.fill();
    cx.fillStyle = starColor(p.warmth, a);
    cx.beginPath(); cx.arc(px, py, rr, 0, 7); cx.fill();
  }
  // ring around the star whose result row is hovered
  if (highlightStar) {
    const p = highlightStar;
    const px = p.x * view.k + view.ox, py = p.y * view.k + view.oy;
    cx.strokeStyle = "rgba(232,228,216,.9)";
    cx.lineWidth = 1.5 * dpr;
    cx.beginPath(); cx.arc(px, py, p.r * rScale + 7 * dpr, 0, 7); cx.stroke();
  }
  requestAnimationFrame(draw);
}

// ---- zoom (wheel, anchored at cursor) and pan (drag) ----
let dragging = false, lastX = 0, lastY = 0;

cv.addEventListener("wheel", e => {
  e.preventDefault();
  const mx = e.clientX * devicePixelRatio, my = e.clientY * devicePixelRatio;
  const k2 = Math.min(Math.max(view.tk * Math.exp(-e.deltaY * 0.002), 1), 30);
  const ratio = k2 / view.tk;
  view.tox = mx - (mx - view.tox) * ratio;
  view.toy = my - (my - view.toy) * ratio;
  view.tk = k2;
  clampView();
  if (searchEl.value.trim()) applySearch();  // keep cluster centered in view
}, { passive: false });

cv.addEventListener("mousedown", e => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  cv.classList.add("dragging");
});
addEventListener("mouseup", () => { dragging = false; cv.classList.remove("dragging"); });

cv.addEventListener("dblclick", () => {
  view.tk = 1; view.tox = 0; view.toy = 0;
  if (searchEl.value.trim()) applySearch();
});

// mini histogram: monthly volume, gold = me, pink = them, stacked
function sparkline(s, w = 248, h = 54) {
  const n = s.me.length;
  const c = document.createElement("canvas");
  c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
  c.style.width = w + "px"; c.style.height = h + "px";
  const g = c.getContext("2d");
  g.scale(devicePixelRatio, devicePixelRatio);
  let max = 1;
  for (let i = 0; i < n; i++) max = Math.max(max, s.me[i] + s.them[i]);
  const bw = w / n;
  for (let i = 0; i < n; i++) {
    const hm = (s.me[i] / max) * (h - 2), ht = (s.them[i] / max) * (h - 2);
    g.fillStyle = "#e8b84b";
    g.fillRect(i * bw, h - hm, Math.max(bw - 0.5, 0.5), hm);
    g.fillStyle = "rgba(224,108,159,.85)";
    g.fillRect(i * bw, h - hm - ht, Math.max(bw - 0.5, 0.5), ht);
  }
  return c;
}

function toneHTML(tone) {
  if (tone === null) return "";
  const pct = Math.round((tone + 1) / 2 * 100);
  const word = tone > 0.25 ? "warm" : tone < -0.1 ? "tense" : "even";
  return `<div class="tonebar"><div class="tonedot" style="left:calc(${pct}% - 6px)"></div></div>` +
    `<div class="tonelabel"><span>−</span><span>tone: ${word} (${tone > 0 ? "+" : ""}${tone})</span><span>+</span></div>`;
}

let tipFor = null;
addEventListener("mousemove", e => {
  if (dragging) {
    view.tox += (e.clientX - lastX) * devicePixelRatio;
    view.toy += (e.clientY - lastY) * devicePixelRatio;
    lastX = e.clientX; lastY = e.clientY;
    clampView();
    view.ox = view.tox; view.oy = view.toy;  // pan follows the hand, no easing lag
    tip.style.opacity = 0; tipFor = null;
    return;
  }
  const mx = e.clientX * devicePixelRatio, my = e.clientY * devicePixelRatio;
  const rScale = Math.sqrt(view.k);
  let best = null, bd = 28 * devicePixelRatio;
  for (const p of placed) {
    const px = p.x * view.k + view.ox, py = p.y * view.k + view.oy;
    const d = Math.hypot(px - mx, py - my) - p.r * rScale;
    if (d < bd) { bd = d; best = p; }
  }
  if (best) {
    const s = best.s;
    if (tipFor !== s) {   // rebuild card only when the star changes
      tipFor = s;
      const words = (s.words || []).map(w => `<span class="word">${w}</span>`).join("");
      tip.innerHTML = `<b>${s.name}</b><br>` +
        `${s.n.toLocaleString()} messages · <span class="dim">${Math.round(s.myShare * 100)}% me</span><br>` +
        `<span class="dim">${s.first} — ${s.last}</span>`;
      tip.appendChild(sparkline(s));
      tip.insertAdjacentHTML("beforeend",
        `<span class="dim" style="font-size:11px">monthly volume — gold me, pink them</span>` +
        (words ? `<div class="words">${words}</div>` : "") +
        toneHTML(s.tone));
    }
    tip.style.left = Math.min(e.clientX + 16, innerWidth - 300) + "px";
    tip.style.top = Math.min(e.clientY + 16, innerHeight - tip.offsetHeight - 16) + "px";
    tip.style.opacity = 1;
  } else { tip.style.opacity = 0; tipFor = null; }
});

addEventListener("resize", resize);
resize();
requestAnimationFrame(draw);
</script>
</body>
</html>
"""


def main() -> None:
    stars, meta, word_index = build_stars()
    subtitle = (f"{meta['totalChats']} relationships · "
                f"{meta['totalMessages']:,} messages · "
                f"{meta['tMin'][:4]}–{meta['tMax'][:4]}")
    html = (
        HTML.replace("__STARS__", json.dumps(stars, ensure_ascii=False))
        .replace("__META__", json.dumps(meta))
        .replace("__WORDS__", json.dumps(word_index, ensure_ascii=False, separators=(",", ":")))
        .replace("__SUBTITLE__", subtitle)
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(stars)} stars, {len(word_index):,} indexed words, "
          f"{OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
