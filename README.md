# Self Data — Personal Archive Pipeline

Code for a living autobiographical archive. Raw exports (Telegram, Gmail,
Calendar, Maps, daily logs, health records) are parsed into a unified,
queryable timeline; on top of that sit analysis scripts (counts, timelines,
correlations) and data-art renderers (constellations, weather, portraits).

The goal is not a productivity dashboard — it is a system for understanding
patterns in life, relationships, energy, attention, health, money, and memory,
in both **analytical** form (charts, counts, correlations) and **poetic** form
(personal atlas, relationship constellations, weekly portraits).

For the full step-by-step story of what has been built so far, with every
chart shown, read **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**.

## Status

| Source | State |
|---|---|
| Telegram (143k messages, 618 chats, 2017–2026) | Parsed, lexical analysis, constellation art |
| Meaning layer (embeddings, topics, sentiment) | Built — resumable local models |
| Conflict heatmap (per relationship) | Built — keyword-based |
| Gmail (7.6 GB mbox) | Planned |
| Calendar / Maps | Planned |
| Daily logs | Schema designed; intake not built |
| Health / money | Planned as data arrives |

## How it's organized

Everything lives in `~/dev/self data`. Code is committable; personal data
dirs are gitignored and never leave your machine.

```
~/dev/self data/
  (committed)     parsers/, analysis/, art/, notebooks/, config.py, docs/
  (gitignored)    raw/, processed/, visualizations/, notes/, people.yaml
```

Key paths: [config.py](config.py) (all filesystem paths),
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) (full architecture walkthrough).

## Setup

```bash
cd ~/dev/self\ data
python3 -m venv .venv               # one isolated Python environment
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

Phase 1 (parse + lexical analysis):

```bash
python parsers/telegram.py
python analysis/telegram_analysis.py
python analysis/word_histograms.py
python analysis/signature_words.py
python art/constellation.py
```

Phase 2 (meaning layer — embeddings and sentiment run in slices; repeat until
`FINISHED`):

```bash
python analysis/embeddings.py
python analysis/topics.py
python analysis/sentiment.py
```

Full command list and dependency order: [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

Each script is idempotent: it reads from `raw/` or `processed/`, overwrites its
outputs, and never touches the original exports.

## What the tools are (plain language)

This project leans on a few pieces of technology. Here is what each one is and
why it's here, no prior knowledge assumed.

### Parquet tables — the core one

A **Parquet table** is a file format for storing a table of data — rows and
columns, like a spreadsheet — but designed for large amounts of data and fast
analysis. A file like `telegram_messages.parquet` holds all 143,318 messages,
one row per message, with columns for timestamp, chat name, sender, text, and
so on.

Why Parquet instead of a plain CSV or an Excel file?

- **Columnar storage.** A spreadsheet stores data row by row. Parquet stores it
  *column by column*. So when you ask "how many messages per month?", the
  computer reads only the timestamp column and ignores the text — much faster
  when a table has hundreds of thousands of rows.
- **Compressed and small.** Parquet squeezes the data down. Those 143k messages
  fit in a few megabytes, and reading them back is near-instant.
- **Remembers its types.** It knows a timestamp is a date, a count is a number,
  `is_me` is true/false. A CSV forgets all of that — everything is just text —
  so dates and numbers have to be re-guessed every time you open it.
- **It's the standard.** Pandas, DuckDB, and almost every data tool read and
  write Parquet directly, so the same file works everywhere without conversion.

Think of `processed/` as the clean, queryable heart of the archive: the messy
HTML/email exports go *in*, and tidy Parquet tables come *out*, ready for any
chart or query. You don't open these files by hand — the scripts and notebook
read them for you.

### The supporting cast

- **pandas** — Python's spreadsheet-in-code. It loads a Parquet table into a
  `DataFrame` (a table you can filter, group, and summarize in a line or two)
  and is what the analysis scripts use to build every chart.
- **DuckDB** — a tiny database that runs **SQL** queries directly on Parquet
  files, with zero setup. It's how the notebook answers questions like "my ten
  most intense conversation days ever" in one short query, even over millions
  of rows.
- **BeautifulSoup / lxml** — read and pull data out of HTML pages. The Telegram
  export is hundreds of HTML files; these turn that tangle into clean rows.
- **matplotlib** — draws the charts (the PNGs in `visualizations/`).
- **pymorphy3** — understands Russian grammar. It reduces *думаю / думала /
  думаешь* to a single base word *думать*, so word counts measure ideas rather
  than grammatical endings.
- **Jupyter notebook** — an interactive document mixing code, results, and
  notes, so you can poke at the data and re-run pieces without editing scripts.
- **Parquet vs. PNG vs. HTML outputs** — Parquet is the *data*, PNGs are static
  *charts*, and the constellation HTML is an *interactive* piece you open in a
  browser.

## Next

Sentiment aggregate charts, Gmail, Calendar/Maps, daily-log intake, and weekly
data portraits — see **Where things stand** in
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).
