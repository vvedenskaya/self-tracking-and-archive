"""Interpretation layer: one markdown note per visualization.

Each note gets auto-generated observations (anomalies, shifts, asymmetries)
computed fresh from the parquet tables, plus a "What this actually was"
section that belongs to you. Reruns replace ONLY the block between the
auto markers — everything you write by hand survives.

Usage: python analysis/notes_generator.py
"""
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

AUTO_BEGIN = "<!-- auto:begin -->"
AUTO_END = "<!-- auto:end -->"

TEMPLATE = """# {title}

Chart: `visualizations/{chart}`

{begin}
{body}
{end}

## What this actually was

_(your words — this section is never touched by the generator)_
"""


def month_str(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


# ---------------------------------------------------------------- observations

def obs_volume(df, freq, sig):
    monthly = df.groupby("month").size()
    mine = df[df.is_me].groupby("month").size().reindex(monthly.index, fill_value=0)
    share = mine / monthly
    z = (monthly - monthly.mean()) / monthly.std()
    loud = z.nlargest(5)
    yearly = df.groupby(df.ts_local.dt.year).size()
    if df.ts_local.max().month < 12:   # current year is incomplete
        yearly = yearly.iloc[:-1]
    yoy = yearly.diff().dropna()
    big = share[monthly >= 200]
    out = ["Loudest months (z-score vs the whole archive):"]
    out += [f"  - {month_str(m)}: {monthly[m]:,} messages (z={zv:.1f})"
            for m, zv in loud.items()]
    out += [
        f"Biggest year-over-year jump: {int(yoy.idxmax())} "
        f"({yoy.max():+,.0f} messages vs {int(yoy.idxmax())-1}); "
        f"biggest drop: {int(yoy.idxmin())} ({yoy.min():+,.0f}).",
        f"Month I dominated most: {month_str(big.idxmax())} "
        f"({big.max():.0%} of messages were mine); most listening: "
        f"{month_str(big.idxmin())} ({big.min():.0%} mine). "
        f"(months with ≥200 messages)",
    ]
    return out


def obs_top_chats(df, freq, sig):
    sizes = df.groupby("chat_name").size().sort_values(ascending=False)
    top5_share = sizes.head(5).sum() / sizes.sum()
    top25 = sizes.head(25)
    share_me = df[df.chat_name.isin(top25.index)].groupby("chat_name").is_me.mean()
    they = share_me.nsmallest(3)
    me = share_me.nlargest(3)
    out = [
        f"Concentration: the top 5 chats hold {top5_share:.0%} of all "
        f"{sizes.sum():,} messages; the top 25 hold "
        f"{top25.sum() / sizes.sum():.0%}.",
        "Most asymmetric of the top 25 — they carry it: "
        + "; ".join(f"{n} ({1 - v:.0%} them)" for n, v in they.items()),
        "Most asymmetric of the top 25 — I carry it: "
        + "; ".join(f"{n} ({v:.0%} me)" for n, v in me.items()),
    ]
    return out


def obs_rhythm(df, freq, sig):
    hours = df.ts_local.dt.hour
    wd = df.ts_local.dt.dayofweek
    cell = df.groupby([hours, wd]).size()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hot = cell.nlargest(3)
    weekend = (wd >= 5).mean()
    night = ((hours >= 1) & (hours < 5)).mean()
    out = ["Hottest cells: " + "; ".join(
        f"{names[w]} {h:02d}:00 ({v:,})" for (h, w), v in hot.items())]
    out += [
        f"Weekend share of all messages: {weekend:.0%} "
        f"(a flat week would be 29%).",
        f"Deep-night messages (01:00–05:00): {night:.1%} of everything.",
        f"Single biggest hour overall: "
        f"{df.groupby(hours).size().idxmax():02d}:00.",
    ]
    return out


def obs_lifespans(df, freq, sig):
    g = df.groupby("chat_name").ts_local
    span = (g.max() - g.min()).dt.days
    sizes = df.groupby("chat_name").size()
    long_lived = span[sizes >= 100].nlargest(5)
    # burstiness: biggest single month as share of the chat's whole life
    monthly_max = df.groupby(["chat_name", "month"]).size().groupby("chat_name").max()
    burst = (monthly_max / sizes)[sizes >= 300].nlargest(3)
    out = ["Longest-lived relationships (≥100 messages):"]
    out += [f"  - {n}: {d / 365:.1f} years" for n, d in long_lived.items()]
    out += ["Burstiest big chats (share of all messages in one month): "
            + "; ".join(f"{n} ({v:.0%})" for n, v in burst.items())]
    return out


def obs_words_me_vs_them(df, freq, sig):
    by = freq.groupby(["lemma", "is_me"])["count"].sum().unstack(fill_value=0)
    by.columns = ["them", "me"]
    totals = by.sum()
    shares = by / totals  # per-word share of each voice's corpus
    big = by[(by.me + by.them) >= 80]
    ratio = (shares.me + 1e-9) / (shares.them + 1e-9)
    big_ratio = ratio[big.index]
    mine = big_ratio.nlargest(8)
    theirs = big_ratio.nsmallest(8)
    out = [
        "Words that are mine, almost never said to me: "
        + ", ".join(mine.index),
        "Words said to me that I almost never use: "
        + ", ".join(theirs.index),
        f"(threshold: ≥80 total uses; ratios computed per 10k words of each "
        f"voice — corpora are {totals.me:,} of my words vs {totals.them:,} "
        f"of theirs)",
    ]
    return out


def obs_words_by_year(df, freq, sig):
    mine = freq[freq.is_me]
    yearly = mine.groupby("year")["count"].sum()
    per10k = (
        mine.groupby(["lemma", "year"])["count"].sum().unstack(fill_value=0)
        / yearly * 10_000
    )
    years = sorted(c for c in per10k.columns if yearly[c] >= 5_000)
    if len(years) < 2:
        return ["Not enough full years to compare."]
    a, b = years[-2], years[-1]
    active = per10k[(per10k[a] >= 3) | (per10k[b] >= 3)]
    delta = (active[b] - active[a]).sort_values()
    out = [
        f"Biggest risers {a} → {b} (per 10k of my words): "
        + ", ".join(f"{w} ({active.loc[w, a]:.0f}→{active.loc[w, b]:.0f})"
                    for w in delta.tail(6).index[::-1]),
        f"Biggest fallers {a} → {b}: "
        + ", ".join(f"{w} ({active.loc[w, a]:.0f}→{active.loc[w, b]:.0f})"
                    for w in delta.head(6).index),
    ]
    newcomers = per10k[(per10k[b] >= 5) & (per10k[years[:-1]].max(axis=1) < 1)]
    if len(newcomers):
        out.append(f"Words that basically didn't exist before {b}: "
                   + ", ".join(newcomers[b].nlargest(8).index))
    return out


def obs_signature_chats(df, freq, sig):
    chats = sig[sig.scope == "chat"]
    sizes = df.groupby("chat_name").size()
    top = [c for c in sizes.nlargest(10).index if c in set(chats.key)]
    out = ["One-line signature of each top relationship (top TF-IDF words):"]
    for c in top:
        words = chats[chats.key == c].nsmallest(5, "rank").lemma
        out.append(f"  - {c}: {', '.join(words)}")
    return out


def obs_signature_years(df, freq, sig):
    years = sig[sig.scope == "year"]
    out = ["The word that owned each year (top TF-IDF, my voice):"]
    for y in sorted(years.key.unique()):
        words = years[years.key == y].nsmallest(3, "rank").lemma
        out.append(f"  - {y}: {', '.join(words)}")
    return out


def obs_constellation(df, freq, sig):
    g = df.groupby("chat_name").ts_local
    span = (g.max() - g.min()).dt.days
    last = g.max()
    cutoff = df.ts_local.max() - pd.Timedelta(days=90)
    active_companions = span[(span >= 4 * 365) & (last >= cutoff)]
    sparks = (span < 30).sum()
    out = [
        f"{sparks} of {len(span)} conversations lasted under a month "
        f"(the blue-white stars).",
        f"{len(active_companions)} relationships span 4+ years AND are still "
        f"active in the last 90 days: "
        + ", ".join(active_companions.sort_values(ascending=False).index[:8]),
    ]
    return out


NOTES = [
    ("telegram_volume_monthly.png", "Nine years of conversation", obs_volume),
    ("telegram_top_chats.png", "Top conversations", obs_top_chats),
    ("telegram_daily_rhythm.png", "Daily rhythm", obs_rhythm),
    ("telegram_lifespans.png", "Conversation lifespans", obs_lifespans),
    ("telegram_words_me_vs_them.png", "My words vs words said to me",
     obs_words_me_vs_them),
    ("telegram_words_by_year.png", "Vocabulary year by year",
     obs_words_by_year),
    ("telegram_signature_words_chats.png", "Signature words per relationship",
     obs_signature_chats),
    ("telegram_signature_words_years.png", "Signature words per year",
     obs_signature_years),
    ("constellation.html", "Relationship constellations", obs_constellation),
]


def render_body(observations) -> str:
    lines = [f"_Generated {date.today().isoformat()} from the processed "
             "parquet tables. Regenerating replaces only this block._", "",
             "### Observations", ""]
    for o in observations:
        if o.startswith("  - "):
            lines.append(o.replace("  - ", "    - ", 1))
        else:
            lines.append(f"- {o}")
    return "\n".join(lines)


def write_note(chart: str, title: str, body: str) -> str:
    path = config.NOTES / (Path(chart).stem + ".md")
    block = f"{AUTO_BEGIN}\n{body}\n{AUTO_END}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if AUTO_BEGIN in text and AUTO_END in text:
            new = re.sub(
                re.escape(AUTO_BEGIN) + r".*?" + re.escape(AUTO_END),
                block.replace("\\", "\\\\"), text, flags=re.S)
            path.write_text(new, encoding="utf-8")
            return "updated"
        # file exists but has no markers — don't risk someone's writing
        path.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
        return "appended"
    path.write_text(
        TEMPLATE.format(title=title, chart=chart, begin=AUTO_BEGIN,
                        body=body, end=AUTO_END),
        encoding="utf-8")
    return "created"


def main() -> None:
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    df["month"] = df.ts_local.dt.to_period("M").dt.to_timestamp()
    freq = pd.read_parquet(config.WORD_FREQ_PARQUET)
    sig = (pd.read_parquet(config.SIGNATURE_WORDS_PARQUET)
           if config.SIGNATURE_WORDS_PARQUET.exists() else pd.DataFrame())

    for chart, title, fn in NOTES:
        if "signature" in chart and sig.empty:
            print(f"  skip {chart} (run signature_words.py first)")
            continue
        status = write_note(chart, title, render_body(fn(df, freq, sig)))
        print(f"  {status}  notes/{Path(chart).stem}.md")
    print(f"\nnotes live in {config.NOTES} — fill in 'What this actually was'")


if __name__ == "__main__":
    main()
