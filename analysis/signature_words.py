"""TF-IDF signature words: what distinguishes each relationship and each year.

Two views over processed/word_frequencies.parquet (lemma, chat_name, is_me,
year, count):

- per relationship: each chat with enough words is a document; TF-IDF finds
  the words that belong to *that* conversation and almost nowhere else
- per year: each year of my own words is a document; TF-IDF finds what made
  that year sound like itself

Writes processed/signature_words.parquet (scope, key, lemma, count, tfidf,
rank) and two chart PNGs.

Usage: python analysis/signature_words.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

plt.rcParams.update(
    {
        "figure.facecolor": "#101018",
        "axes.facecolor": "#101018",
        "savefig.facecolor": "#101018",
        "text.color": "#e8e4d8",
        "axes.labelcolor": "#e8e4d8",
        "xtick.color": "#9a96a8",
        "ytick.color": "#9a96a8",
        "axes.edgecolor": "#3a3650",
        "font.family": "Helvetica Neue",
        "figure.dpi": 150,
    }
)

MIN_DOC_WORDS = 300   # a chat needs this many counted words to be a document
MIN_TERM_COUNT = 5    # a word needs this many uses inside a doc to qualify
TOP_K = 15            # signature words kept per document


def tfidf(docs: pd.DataFrame) -> pd.DataFrame:
    """docs: columns [doc, lemma, count]. Returns top-K signature words per doc.

    tf is the word's share of its document; idf is smoothed log over the
    document set WITHOUT the sklearn '+1' addend, so a word present in every
    document (хотеть, знать) scores exactly zero — only what distinguishes
    survives.
    """
    totals = docs.groupby("doc")["count"].sum()
    n_docs = docs.doc.nunique()
    doc_freq = docs.groupby("lemma")["doc"].nunique()

    d = docs[docs["count"] >= MIN_TERM_COUNT].copy()
    d["tf"] = d["count"] / d.doc.map(totals)
    d["idf"] = np.log((1 + n_docs) / (1 + d.lemma.map(doc_freq)))
    d["tfidf"] = d.tf * d.idf
    d = d[d.tfidf > 0]

    d = d.sort_values("tfidf", ascending=False)
    d["rank"] = d.groupby("doc").cumcount() + 1
    return d[d["rank"] <= TOP_K].reset_index(drop=True)


def relationship_signatures(freq: pd.DataFrame) -> pd.DataFrame:
    """One document per chat (both voices: the chat is the relationship)."""
    by_chat = (
        freq.groupby(["chat_name", "lemma"], as_index=False)["count"].sum()
        .rename(columns={"chat_name": "doc"})
    )
    big_enough = by_chat.groupby("doc")["count"].sum()
    keep = big_enough[big_enough >= MIN_DOC_WORDS].index
    return tfidf(by_chat[by_chat.doc.isin(keep)])


def year_signatures(freq: pd.DataFrame) -> pd.DataFrame:
    """One document per year, my words only: the sound of each year."""
    by_year = (
        freq[freq.is_me]
        .groupby(["year", "lemma"], as_index=False)["count"].sum()
        .assign(doc=lambda d: d.year.astype(str))[["doc", "lemma", "count"]]
    )
    return tfidf(by_year)


def chart_grid(sig: pd.DataFrame, docs: list, fname: str, title: str,
               color: str, n_words: int = 10) -> None:
    cols = 4
    rows = -(-len(docs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.4 * rows))
    for ax, doc in zip(axes.flat, docs):
        top = sig[sig.doc == doc].nsmallest(n_words, "rank").iloc[::-1]
        ax.barh(top.lemma, top.tfidf, color=color)
        label = doc if len(doc) <= 22 else doc[:21] + "…"
        ax.set_title(label, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xticks([])
    for ax in axes.flat[len(docs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16, y=0.998)
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / fname)
    plt.close(fig)


def main() -> None:
    freq = pd.read_parquet(config.WORD_FREQ_PARQUET)
    if "chat_name" not in freq.columns:
        sys.exit("word_frequencies.parquet has no chat_name column — "
                 "rerun analysis/word_histograms.py first")

    chat_sig = relationship_signatures(freq)
    year_sig = year_signatures(freq)

    out = pd.concat([
        chat_sig.assign(scope="chat"),
        year_sig.assign(scope="year"),
    ]).rename(columns={"doc": "key"})
    out = out[["scope", "key", "lemma", "count", "tfidf", "rank"]]
    out.to_parquet(config.SIGNATURE_WORDS_PARQUET, index=False)
    print(f"wrote {len(out):,} signature rows "
          f"({chat_sig.doc.nunique()} chats, {year_sig.doc.nunique()} years) "
          f"to {config.SIGNATURE_WORDS_PARQUET}")

    # chart the 16 biggest relationships and every year
    chat_sizes = freq.groupby("chat_name")["count"].sum()
    top_chats = [c for c in chat_sizes.nlargest(16).index
                 if c in set(chat_sig.doc)]
    chart_grid(chat_sig, top_chats, "telegram_signature_words_chats.png",
               "Signature words — what each relationship talks about\n"
               "(TF-IDF: high = this conversation's word, rare elsewhere)",
               "#e8b84b")

    years = sorted(year_sig.doc.unique())
    chart_grid(year_sig, years, "telegram_signature_words_years.png",
               "Signature words of each year (my words only)\n"
               "(TF-IDF against the other years)",
               "#5fc8c8")
    print(f"wrote 2 signature charts to {config.VISUALIZATIONS}")

    for doc in top_chats[:5]:
        words = chat_sig[chat_sig.doc == doc].nsmallest(8, "rank").lemma
        print(f"  {doc}: {', '.join(words)}")


if __name__ == "__main__":
    main()
