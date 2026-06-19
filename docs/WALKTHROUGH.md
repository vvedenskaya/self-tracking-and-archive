# Walkthrough — Personal Archive Pipeline

What was built, how the code is organized, and what comes next. This document
walks through the whole project: raw exports → clean tables → lexical analysis
→ meaning layer → art pieces → your written notes.

Everything below ran against your real Telegram archive (**143,318 messages ·
618 conversations · 539 people · April 2017 to June 2026**; you wrote 69,318,
48%).

---

## Architecture in one picture

```
~/dev/self data/                     <- this repo (code + gitignored data)
  raw/                               <- original exports, never modified
        |
   parsers/                         <- one script per source (Telegram done)
        |
  processed/                         <- clean Parquet tables + ML caches
        |
   +----+----+
   |         |
lexical/   meaning/                  <- two analysis layers, same messages
   |         |
   |    textfilter.py  (shared filter)
   |    embeddings.py  (vectors, once)
   |    topics.py      (topic river)
   |    sentiment.py   (emotion scores)
   |
word_histograms.py
signature_words.py
telegram_analysis.py
conflict_heatmap.py
        |
  analysis/  +  art/  +  notebooks/
        |
  visualizations/                    <- PNG charts + HTML art pieces
  notes/                             <- auto observations + your annotations
```

**Design rules:**

- **Code is committable; data is not.** `raw/`, `processed/`, `visualizations/`,
  `notes/`, and `people.yaml` are all gitignored. Nothing personal ever leaves
  your machine — all parsing, embedding, and classification run locally.
- **One config file.** Every path lives in [config.py](../config.py). If the
  archive moves, edit one file.
- **Idempotent scripts.** Each script reads from `raw/` or `processed/`, writes
  its outputs, and never touches the original export.
- **Two layers on the same messages.** Lexical passes count *words* (fast,
  deterministic). Meaning passes read *semantics* (slow, model-based). They
  share a common filter ([analysis/textfilter.py](../analysis/textfilter.py))
  so topics and sentiment see the same "human voice" subset.

---

## Repository layout

| Path | Role |
|---|---|
| [config.py](../config.py) | All filesystem paths; creates output dirs on import |
| [requirements.txt](../requirements.txt) | Python deps (Phase 1 + local ML stack for Phase 2) |
| [parsers/telegram.py](../parsers/telegram.py) | Telegram HTML export → `telegram_messages.parquet` |
| [analysis/telegram_analysis.py](../analysis/telegram_analysis.py) | Volume, rhythm, lifespans — four canonical charts |
| [analysis/word_histograms.py](../analysis/word_histograms.py) | RU/EN lemma counts → `word_frequencies.parquet` |
| [analysis/signature_words.py](../analysis/signature_words.py) | TF-IDF per chat and per year |
| [analysis/textfilter.py](../analysis/textfilter.py) | Shared human-voice filter for meaning passes |
| [analysis/embeddings.py](../analysis/embeddings.py) | Multilingual sentence embeddings (cached once) |
| [analysis/topics.py](../analysis/topics.py) | Topic clustering + nine-year topic river |
| [analysis/sentiment.py](../analysis/sentiment.py) | Per-message emotion scores (RU classifier) |
| [analysis/conflict_heatmap.py](../analysis/conflict_heatmap.py) | Tension episodes for one relationship |
| [analysis/notes_generator.py](../analysis/notes_generator.py) | Auto observations → `notes/*.md` |
| [analysis/people_registry.py](../analysis/people_registry.py) | Seeds `people.yaml` from top chats |
| [art/constellation.py](../art/constellation.py) | Relationship Constellations (interactive HTML) |
| [notebooks/01_telegram.ipynb](../notebooks/01_telegram.ipynb) | Interactive exploration + DuckDB SQL |

### Processed tables (what gets built)

| File | Produced by | Contents |
|---|---|---|
| `telegram_messages.parquet` | `parsers/telegram.py` | Every message: time, chat, sender, text, flags |
| `word_frequencies.parquet` | `word_histograms.py` | Lemma × chat × speaker × year counts |
| `signature_words.parquet` | `signature_words.py` | TF-IDF scores per chat and per year |
| `message_index.parquet` | `embeddings.py` | ~88k human-voice messages + metadata |
| `message_embeddings.npy` | `embeddings.py` | Float32 vectors [N, 384], row-aligned with index |
| `topics.parquet` | `topics.py` | Topic id + label per substantive message |
| `sentiment.parquet` | `sentiment.py` | Six emotion probabilities per message |

---

## For backend engineers

This is a **batch ETL + analytics pipeline**, not an application. No server, no
requests — scripts read Parquet, write Parquet/PNG, and exit. The mental model
that makes the repo legible: **trace data files, not import graphs.**

### Job DAG (dependency order)

```
parsers/telegram.py
    └── telegram_messages.parquet
            ├── telegram_analysis.py        → PNGs
            ├── word_histograms.py          → word_frequencies.parquet → PNGs
            │       └── signature_words.py  → signature_words.parquet → PNGs
            ├── textfilter.py               (library — not a job)
            ├── embeddings.py               → message_index.parquet + message_embeddings.npy
            │       ├── topics.py           → topics.parquet → topic river PNG
            │       └── sentiment.py        → sentiment.parquet
            ├── conflict_heatmap.py         → PNGs + notes report (reads telegram directly)
            ├── people_registry.py          → people.yaml (append-only)
            └── notes_generator.py          → notes/*.md

art/constellation.py  ← reads telegram + embeddings + sentiment + topics → HTML
```

Run jobs top-to-bottom. If an upstream artifact is missing, the downstream script
will fail or exit with a clear message.

### Parquet filenames as the service map

In a microservice backend, you navigate by API endpoints and database tables.
Here, **Parquet filenames are the API between jobs.** [config.py](../config.py)
is the registry — every artifact path in one file.

| Backend analogy | This project |
|---|---|
| `POST /users` creates a row | `parsers/telegram.py` creates `telegram_messages.parquet` |
| Service A writes table `orders` | `word_histograms.py` writes `word_frequencies.parquet` |
| Service B reads from `orders` | `signature_words.py` reads `word_frequencies.parquet` |
| OpenAPI / schema = contract | **Filename + columns = contract between scripts** |

You do not need to memorize every script. Know which `.parquet` files exist,
who writes them, who reads them — that *is* the architecture diagram.

**Example debugging:** `topics.py` fails → check the map: it needs
`message_embeddings.npy` + `message_index.parquet` from `embeddings.py`, which
needs `telegram_messages.parquet` from `parsers/telegram.py`.

### Script categories

| Type | Examples | Pattern |
|---|---|---|
| **Parser** | `parsers/telegram.py` | raw → normalized Parquet; one source, one output schema |
| **Transform** | `word_histograms.py`, `topics.py` | read Parquet → compute → write Parquet + optional viz |
| **Library** | `textfilter.py` | imported by other scripts; never run directly |
| **Long worker** | `embeddings.py`, `sentiment.py` | chunked batch job with on-disk checkpoints |

### Parquet schemas (contracts between jobs)

| Table | Grain | Key columns |
|---|---|---|
| `telegram_messages.parquet` | 1 row = 1 message | `chat_name`, `ts_utc`, `sender`, `text`, `is_me` |
| `word_frequencies.parquet` | 1 row = lemma × chat × speaker × year | `lemma`, `count` |
| `message_index.parquet` | 1 row = 1 embeddable message | `clean`, `is_code`; row *i* aligns with row *i* of `.npy` |
| `topics.parquet` | 1 row = 1 message + topic | `topic`, `topic_label` |
| `sentiment.parquet` | 1 row = 1 message + 6 emotion probs | `joy`, `sadness`, `anger`, … |

### How to read any script

Every script follows the same shape. Read in this order:

1. **Module docstring** — I/O contract (inputs, outputs, CLI flags)
2. **`import config`** — which artifacts it touches
3. **`main()` / `build()`** — orchestration only (~10–30 lines)
4. **One core function** — the actual transform (`tokens()`, `cluster()`, `detect_episodes()`)
5. **Skip chart code on first pass** — matplotlib is presentation, not logic

For [conflict_heatmap.py](../analysis/conflict_heatmap.py) specifically:
`resolve_chat_name()` → lookup · `score_message()` / `is_boundary()` → rules ·
`detect_episodes()` → clustering · `chart_*()` → rendering (read last).

### Shared infrastructure patterns

**Single config.** Every script does `sys.path.insert(…); import config`. All
paths live in one place.

**Idempotent overwrite.** Scripts overwrite their outputs (except
`people_registry.py`, which append-only merges new chats). Safe to rerun.

**Shared filter.** [textfilter.py](../analysis/textfilter.py) is middleware for
Phase 2: `human_voice(df)` adds a `clean` column and drops forwards, code stubs,
and media-only messages. Topics and sentiment must see identical input.

**Chunked workers** (`embeddings.py`, `sentiment.py`) — same pattern as a
resumable batch consumer:

```
chunk messages → process → write chunk_N to disk → exit after TIME_BUDGET (200s)
re-run → skip existing chunks → continue
all chunks done → merge → write final artifact → delete temp dir
```

Embedding 88k messages on CPU takes hours; the script yields instead of blocking.
Checkpoints live in `processed/emb_chunks/` and `processed/sent_chunks/`.

### Phase 1 vs Phase 2

| | Phase 1 (lexical) | Phase 2 (semantic) |
|---|---|---|
| Compute | deterministic, fast | ML models, slow |
| Dependencies | pandas, pymorphy3 | + torch, transformers |
| Input | all messages | filtered human voice (~88k) |
| Output | counts, charts | vectors, clusters, probabilities |
| Re-run cost | seconds | hours (embeddings) |

Phase 2 adds a **feature store** (`message_embeddings.npy`) that downstream jobs
read without re-embedding.

### Suggested reading order

1. [config.py](../config.py) — full artifact map (2 min)
2. [parsers/telegram.py](../parsers/telegram.py) — ingestion + schema (15 min)
3. [analysis/textfilter.py](../analysis/textfilter.py) — shared preprocessing (5 min)
4. [analysis/embeddings.py](../analysis/embeddings.py) — checkpoint pattern (10 min)
5. [analysis/topics.py](../analysis/topics.py) — `main()` + `cluster()` (10 min)
6. One tool script of interest — e.g. [conflict_heatmap.py](../analysis/conflict_heatmap.py)

Skip on first pass: `notes_generator.py`, chart functions, `constellation.py`
(art layer, not pipeline core).

### Inspect the data layer without reading code

```bash
# Schema
.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('processed/telegram_messages.parquet').dtypes)"

# Row counts
.venv/bin/python -c "import pandas as pd; df=pd.read_parquet('processed/telegram_messages.parquet'); print(len(df), df.chat_name.nunique())"

# SQL over parquet
.venv/bin/python -c "import duckdb; print(duckdb.sql(\"SELECT chat_name, COUNT(*) c FROM 'processed/telegram_messages.parquet' GROUP BY 1 ORDER BY 2 DESC LIMIT 10\").df())"
```

Or use [notebooks/01_telegram.ipynb](../notebooks/01_telegram.ipynb) as a REPL.

### What's intentionally not here (yet)

No orchestrator (Airflow/Prefect), Makefile, or CLI wrapper — you run scripts
manually in dependency order. No tests. No Parquet schema versioning. No API
layer; the "API" is files on disk. Fine for a personal archive; the gap to
production is deliberate.

---

## Setup

```bash
cd ~/dev/self\ data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Phase 2 adds a local ML stack (`torch`, `transformers`, `sentence-transformers`,
`scikit-learn`). Versions are pinned for Intel macOS — see comments in
`requirements.txt` before upgrading.

---

# Phase 1 — Foundation + Telegram

## Step 1 — Project scaffold

- `requirements.txt` — parsing (`beautifulsoup4`, `lxml`), tables (`pandas`,
  `pyarrow`), SQL (`duckdb`), charts (`matplotlib`), Russian lemmatization
  (`pymorphy3`), notebooks (`jupyter`).
- `.venv/` — isolated Python 3.12 environment.
- [config.py](../config.py) — every input and output path.

## Step 2 — Telegram parser

**Script:** [parsers/telegram.py](../parsers/telegram.py)
**Run:** `.venv/bin/python parsers/telegram.py` (~3.5 minutes)
**Output:** `processed/telegram_messages.parquet` — 143,318 rows

The Telegram export is 719 folders of paginated HTML (`messages.html`,
`messages2.html`, …). The parser walks every folder and pulls, per message:

| column | meaning | where it comes from |
|---|---|---|
| `chat_id` / `chat_name` | which conversation | folder name + page header |
| `msg_html_id` | stable message id | HTML `id` attribute |
| `ts_local` / `ts_utc` | when, local and UTC | `title` attr: `19.10.2021 17:31:10 UTC-05:00` |
| `sender` | who wrote it | `from_name` div; consecutive messages omit it, so the parser carries the last sender forward |
| `text` | the words | `text` div |
| `media_type` | photo / voice / sticker / call / … | CSS classes on the message |
| `is_me` | did you write it | see below |
| `is_forwarded` | forwarded content | `forwarded body` marker |

**The "who am I" trick.** Telegram's export never says whose archive it is. The
parser detects the owner statistically: the sender who speaks in the most
*distinct* chats. You ("Y") speak in 592 of 618 chats; the runner-up appears in
34. Unambiguous.

101 of the 719 folders contained no parseable messages (empty or
service-message-only chats), leaving 618 real conversations.

## Step 3 — Analysis charts

**Script:** [analysis/telegram_analysis.py](../analysis/telegram_analysis.py)
**Run:** `.venv/bin/python analysis/telegram_analysis.py`
**Output:** four PNGs in `visualizations/`

| Chart | File | What it shows |
|---|---|---|
| Nine years of conversation | `telegram_volume_monthly.png` | Messages per month, you vs everyone else |
| Top conversations | `telegram_top_chats.png` | 25 biggest relationships; color = who writes more |
| Daily rhythm | `telegram_daily_rhythm.png` | Hour × weekday heatmap, all years summed |
| Conversation lifespans | `telegram_lifespans.png` | 30 biggest relationships as timelines |

The archive is quiet until late 2019, erupts in 2021–2022 (peaking above 4,500
messages/month), then settles into a steadier 2024–2026 rhythm.

## Step 4 — Word histograms (RU + EN)

**Script:** [analysis/word_histograms.py](../analysis/word_histograms.py)
**Run:** `.venv/bin/python analysis/word_histograms.py`
**Output:** `word_frequencies.parquet` (297,280 lemma×chat×speaker×year rows)
+ two PNGs

How a word becomes a count:

1. URLs are stripped (before this fix, "https" and "com" topped your list).
2. Text is lowercased and split into Cyrillic/Latin word tokens.
3. Russian words are **lemmatized** with pymorphy3 — думаю / думала / думаешь
   all collapse to *думать*.
4. RU + EN stopwords and tokens under 3 letters are dropped.

Counts are kept per **(lemma, chat, speaker, year)** so any later analysis can
slice by relationship without re-tokenizing.

Outputs: `telegram_words_me_vs_them.png`, `telegram_words_by_year.png`.

Both sides of nine years share a #1 word: **хотеть** — to want. Your column
runs хотеть, завтра, написать, давать, сегодня, знать, делать, работать — a
vocabulary of intention and forward motion.

## Step 5 — Interactive notebook

**File:** [notebooks/01_telegram.ipynb](../notebooks/01_telegram.ipynb)

Five sections:

1. Load parquet tables, print vital signs.
2. Regenerate all four canonical charts inline.
3. **SQL over the archive** with DuckDB — find your ten most intense
   conversation days; edit the query to ask anything.
4. **Single-relationship zoom**: set `PERSON` to any chat name for monthly shape
   + that relationship's defining words.
5. Word histograms inline + one-word time machine (default *работать*).

```bash
cd ~/dev/self\ data && .venv/bin/jupyter lab notebooks/01_telegram.ipynb
```

## Step 6 — Relationship Constellations (art piece)

**Script:** [art/constellation.py](../art/constellation.py)
**Output:** `visualizations/constellation.html` — open in any browser, no server
needed (501 stars, data embedded)

Every conversation with ≥3 messages becomes a star:

- **x** — when the relationship lived (center of mass of messages)
- **y** — who carried it (they wrote more ↑, you wrote more ↓)
- **size** — total messages (log scale)
- **color** — lifespan (brief = blue-white, long = gold)
- **brightness** — recency (active relationships shine; silent ones fade)

## Step 7 — Signature words (TF-IDF)

**Script:** [analysis/signature_words.py](../analysis/signature_words.py)
**Output:** `signature_words.parquet` + two PNGs

Histograms show what you say *most*; TF-IDF shows what you say *here and nowhere
else*. Every chat with ≥300 counted words becomes a document (178 of them);
every year of your words becomes another. Words present everywhere (хотеть)
score zero — only the distinctive survives.

- `telegram_signature_words_chats.png` — 16 biggest relationships, ten words each
- `telegram_signature_words_years.png` — what made each year sound like itself

## Step 8 — Interpretation layer (`notes/`)

**Script:** [analysis/notes_generator.py](../analysis/notes_generator.py)
**Output:** one markdown file per chart in `notes/`

Each note has auto-computed observations (loudest months, year-over-year shifts,
asymmetric relationships, …) followed by **"What this actually was"** — yours
to fill in. The auto block sits between `<!-- auto:begin -->` /
`<!-- auto:end -->` markers; regenerating replaces only that block.

## Step 9 — `people.yaml`

**Script:** [analysis/people_registry.py](../analysis/people_registry.py)
**Output:** `people.yaml` at repo root (gitignored)

Registry of who's who: Telegram chat names, email addresses (empty until Gmail),
tags, notes. Seeded from top 60 chats; rerunning **appends** new big chats but
never rewrites — hand edits are always safe. Backbone for cross-source views and
tools like `conflict_heatmap.py --person sofia_dro`.

---

# Phase 2 — Meaning layer

Phase 2 answers questions word counts cannot: *what were you talking about*,
*how did it feel*, *when did tension cluster*. It runs on a filtered subset of
~88k "human voice" messages (real text, not forwards, not code-debugging stubs).

## Step 10 — Human voice filter

**Module:** [analysis/textfilter.py](../analysis/textfilter.py)

Shared by embeddings, topics, and sentiment. Drops:

- media-only stubs and empty messages
- forwards (someone else's words)
- messages under 3 words after URL stripping
- code-heavy messages (optional; on by default)

Adds a `clean` column (URLs stripped, whitespace normalized). Also flags
`is_code` so downstream passes can include or exclude programming chatter
consistently.

This directly addresses the "chat became a workbench" finding from Phase 1 —
by 2025 your top words are JSON keys and function names. The meaning layer
separates human conversation from debugging noise before clustering or scoring
emotions.

## Step 11 — Message embeddings (run once, resumable)

**Script:** [analysis/embeddings.py](../analysis/embeddings.py)
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, RU+EN, CPU)
**Output:** `message_embeddings.npy` + `message_index.parquet`

Embedding ~88k messages is the expensive step. The script:

- processes messages in chunks of 2,000
- saves each chunk to disk
- **exits after ~200 seconds** so it can finish inside sandbox/timeout limits
- skips already-done chunks on rerun

```bash
.venv/bin/python analysis/embeddings.py            # one slice, then exit
# repeat until it prints FINISHED
.venv/bin/python analysis/embeddings.py --force    # start over
```

Other scripts import `embeddings.load()` to get `(vectors, index)` in matching
row order.

## Step 12 — Topic river

**Script:** [analysis/topics.py](../analysis/topics.py)
**Output:** `topics.parquet` + `telegram_topic_river.png`

Where signature words show each year's *fingerprint*, the topic river shows the
*flow* — what themes rose and fell across 2017–2026.

Pipeline (hand-rolled on sklearn — BERTopic/UMAP/HDBSCAN are commented out in
`requirements.txt` because `llvmlite` won't build on this Intel Mac):

```
cached embeddings  →  drop register-noise (laughter, fillers, profanity)
                   →  PCA(50)  →  MiniBatchKMeans(k=40)
                   →  c-TF-IDF labels per cluster
                   →  streamgraph (% share per quarter)
```

Register-noise filtering matters: the first pass produced clusters like
"смешно/пиздец" and "ням/аминь" — *how* people talk, not *what* about. The
river is drawn in **shares** (% of that quarter's substantive messages) so it
shows composition shifts, not just volume changes.

```bash
.venv/bin/python analysis/topics.py                 # full archive, k=40
.venv/bin/python analysis/topics.py --k 30
.venv/bin/python analysis/topics.py --person "Sofia Dro"
```

## Step 13 — Sentiment / emotional weather

**Script:** [analysis/sentiment.py](../analysis/sentiment.py)
**Model:** `cointegrated/rubert-tiny2-cedr-emotion-detection` (CEDR labels)
**Output:** `sentiment.parquet`

Scores every human-voice message for six emotions: joy, sadness, surprise, fear,
anger, no_emotion. Same resumable chunk pattern as embeddings.

**Important:** per-message scores are noise. The design intent is aggregation —
(chat × month) means for emotional weather, warmth asymmetry, and eventually
correlating felt state (daily logs) against expressed emotion (Telegram).

```bash
.venv/bin/python analysis/sentiment.py            # one slice, then exit
# repeat until FINISHED
.venv/bin/python analysis/sentiment.py --force
```

Requires `message_index.parquet` from embeddings first.

## Step 14 — Conflict heatmap (per relationship)

**Script:** [analysis/conflict_heatmap.py](../analysis/conflict_heatmap.py)
**Output:** `conflict_heatmap_{person}.png`, `conflict_timeline_{person}.png`

A relationship-specific tool (not archive-wide). Scores messages for
tension/anger language (RU + EN regex patterns), clusters them into episodes,
attributes who escalated first, and draws:

- a year × month heatmap (color = conflict intensity; dots = episode initiator)
- a timeline of episodes with peak message excerpts

Uses `people.yaml` to resolve `--person sofia_dro` → chat name.

```bash
.venv/bin/python analysis/conflict_heatmap.py
.venv/bin/python analysis/conflict_heatmap.py "Sofia Dro"
.venv/bin/python analysis/conflict_heatmap.py --person sofia_dro
```

Future work: rebuild this heatmap using `sentiment.parquet` anger/fear aggregates
instead of (or alongside) keyword patterns.

## Step 15 — Constellation, upgraded (Phase 2 layers)

**Script:** [art/constellation.py](../art/constellation.py)
**Output:** `visualizations/constellation.html` — still one self-contained, offline file (~1 MB)

The Phase 1 constellation plotted *when* relationships lived and *who* carried
them. Phase 2 folds embeddings and sentiment into the same page as switchable
layers. Everything is reduced to **per-chat summaries** before it ships, so the
file stays ~1 MB and loads instantly — raw embeddings (130 MB) never reach the
browser.

- **Color → warmth.** A `lifespan / warmth` toggle. Warmth = mean(joy) −
  mean(anger + sadness + fear) per chat, from `sentiment.parquet`, on a diverging
  pink→grey→teal scale. 271/501 chats have enough scored messages; the rest stay
  neutral grey. (Most relationships read warm — median +0.13.)
- **Layout → meaning.** A `timeline / meaning` toggle. "Meaning" lays chats out
  by a t-SNE of their mean embeddings, so similar conversations cluster — family
  near family, the pikesquares/work chats in one knot. 403/501 positioned.
- **Click a star → dossier.** A large in-page panel: emotional weather by month
  (teal up = warmer, pink down = tenser), volume (me/them), signature words, and
  **similar relationships** — the nearest chats by content, each clickable to
  jump. This is the embeddings' payoff: who you talk to *like* you talk to
  someone else.
- **Topic river strip.** A toggleable bottom overview — the theme shares from
  `topics.parquet`, redrawn as a live streamgraph over 2017–2026.

Two fixes worth remembering for future you:

- **Mean-centering.** Raw chat-mean embeddings all point the same way (everyday
  chatter), so every cosine was ~0.99 and t-SNE collapsed into one blob.
  Subtracting the global mean before cosine/t-SNE is what makes neighbours and
  the galaxy meaningful.
- **Code out of the tags.** Signature words now run through `is_code_heavy` plus
  a `CODE_JUNK` denylist, so a chat that began in 2020 is no longer tagged with
  code that only entered your life in 2024+.

```bash
.venv/bin/python art/constellation.py    # needs embeddings, sentiment, topics first
```

---

## Meta-patterns — what the archive says when you step back

Computed from `word_frequencies.parquet` (word shares per 10,000 words) and
`telegram_messages.parquet`. Caveat: 2017 is only ~200 of your words — anecdote,
not statistics.

### Work turned from a noun into a verb

**работа** (work as a *thing*) peaked in 2019 at 66 per 10k, then faded to
16–18 by 2025–26. **работать** (work as an *activity*) climbed from 2021 (30)
through 2024 (50). Around 2020–2021, work stopped being something you *talked
about* and became something you *were doing*.

### The chat became a workbench

Top-8 words you wrote each year drift from names and logistics (2018) → pure
intention (2021–2023) → code tokens (2024–2026: bridge, stats, null, lat, lon).
Telegram absorbed your programming life. Phase 2's `textfilter.is_code_heavy`
is the first structural response to this.

### Wanting is quieting down

**хотеть** is #1 on both sides — but falling: ~74–86 per 10k through 2020–2023,
then 72 (2024), 53 (2025), 40 (2026). Half its former strength.

### 2021–2022 was a big bang of people

New conversations per year: 7, 17, 48, 37, then **146 (2021) and 188 (2022)**,
then ~35. Two years brought in more new people than the other eight combined.

### Half of all relationships are sparks

Of 618 conversations: **286 lasted under a month**, 149 between a month and a
year, 142 over a year, 33 over four years.

### You used to listen; now you lead

Your share of messages written: 39% (2017) → ~48–49% through 2023 → **52%
(2025) and 54% (2026)** — first time you consistently out-write everyone.

### 2023 was the tired year

**устать**, **спать**, and **любить** all peak in 2023 simultaneously. Fatigue,
sleep, and love cresting together — the kind of pattern **daily logs** (mood,
energy, sleep) will eventually confirm or complicate.

### The vocabulary of intention, refined

Dominant words are verbs of near-future action — хотеть, написать, давать,
делать — anchored by *tomorrow* (завтра). Feeling-words barely register. Your
Telegram voice plans and builds; it rarely declares. Whether feelings lived
elsewhere is exactly what daily logs, diaries, and the meaning layer exist to
answer.

---

## Where things stand

### Done

| Area | Status |
|---|---|
| Project scaffold + config | ✓ |
| Telegram parser (143k messages) | ✓ |
| Canonical charts (volume, rhythm, lifespans) | ✓ |
| Word histograms + signature words (TF-IDF) | ✓ |
| Interactive notebook + DuckDB | ✓ |
| Constellation art piece (HTML) | ✓ |
| Notes generator + `people.yaml` registry | ✓ |
| Human voice filter (`textfilter.py`) | ✓ |
| Message embeddings (resumable cache) | ✓ |
| Topic river (sklearn clustering + streamgraph) | ✓ |
| Sentiment scoring (resumable, RU classifier) | ✓ |
| Conflict heatmap (keyword-based, per person) | ✓ |
| Constellation Phase 2 layers (warmth · galaxy · dossier · river) | ✓ |

### Not yet built (in planned order)

| Pass | What it adds |
|---|---|
| **Sentiment aggregates** | Per-relationship emotional weather now lives in the constellation dossier; still to do: me-vs-them warmth asymmetry and a sentiment-based conflict view |
| **Gmail** | Streaming parser for the 7.6 GB mbox (metadata for all mail, bodies for sent mail); same word pipeline for email voice |
| **Calendar + Maps** | `.ics` and JSON parsers into a unified timeline |
| **Daily logs** | Schema + intake for Google Forms (mood, energy, sleep, exercise, reflection fields); join to Telegram/sentiment on date |
| **Weekly data portraits** | Semi-automated Dear Data pages from the combined timeline |
| **Health / money** | Parsers as data arrives (`raw/blood tests/` already present) |
| **Code vs human voice** | Refine word histograms to exclude programming tokens globally (partially done in Phase 2 filter) |

---

## Rerunning everything

Run in dependency order. Steps marked *(repeat)* exit partway through and must
be rerun until they print `FINISHED`.

```bash
cd ~/dev/self\ data

# --- Phase 1: parse + lexical ---
.venv/bin/python parsers/telegram.py              # after a fresh Telegram export
.venv/bin/python analysis/telegram_analysis.py
.venv/bin/python analysis/word_histograms.py
.venv/bin/python analysis/signature_words.py
.venv/bin/python analysis/notes_generator.py      # your annotations survive
.venv/bin/python analysis/people_registry.py      # only appends, never edits

# --- Phase 2: meaning layer ---
.venv/bin/python analysis/embeddings.py           # *(repeat)* until FINISHED
.venv/bin/python analysis/topics.py
.venv/bin/python analysis/sentiment.py            # *(repeat)* until FINISHED
.venv/bin/python analysis/conflict_heatmap.py     # optional; pick a person

# --- art: constellation reads embeddings + sentiment + topics, so it runs last ---
.venv/bin/python art/constellation.py
```

Each script is idempotent: it reads from `raw/` or `processed/`, overwrites its
own outputs, and never touches the original export.

---

## How the layers connect (for future you)

When daily logs arrive, the join key is **date**:

```
daily_logs.parquet.date  ←→  telegram_messages.ts_local.date
                           ←→  sentiment (aggregated per day)
                           ←→  topics (dominant topic per day)
                           ←→  calendar events
```

When Gmail arrives, `people.yaml` links chat names to email addresses so the
same person appears once across Telegram, mail, and calendar.

The notes in `notes/` are the human interpretation layer on top of all of this —
auto observations give you the *what*; your "What this actually was" sections
become the narrative spine of the final artifact.
