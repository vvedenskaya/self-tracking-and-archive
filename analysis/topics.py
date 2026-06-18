"""Topic modelling over Telegram messages — the nine-year topic river.

Where signature_words shows each year's *fingerprint*, this shows the *flow*:
what you talked about, and how those themes rose and fell across 2017–2026.

Pipeline (hand-rolled on sklearn, no BERTopic/UMAP/HDBSCAN — those pull
numba->llvmlite, which won't build on this Intel mac):

    cached embeddings  ->  keep substantive msgs  ->  PCA(50)
                       ->  MiniBatchKMeans(k)      ->  c-TF-IDF labels

"Substantive" drops register-noise (laughter, interjections, profanity,
transliterated function words) so clusters land on *topics*, not *tone* —
the first pass produced bands like "смешно/пиздец" and "ням/аминь", which
are how people *talk*, not what they talk *about*. The river is drawn in
shares (% of that quarter) so it shows shifts in composition, not just volume.

c-TF-IDF reuses the idf-without-+1 trick from signature_words and the RU
lemmatizer from word_histograms, so labels read as ideas, not grammar.

    python analysis/topics.py                 # k=40 topics
    python analysis/topics.py --k 30
    python analysis/topics.py --person "Sofia Dro"
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import embeddings as emb_cache
from word_histograms import tokens  # lemmatizing RU+EN tokenizer + stopwords

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

PCA_DIMS = 50
MIN_MEANINGFUL = 3   # substantive message needs this many non-junk lemmas
TOP_WORDS = 7        # label words per topic
RIVER_TOPICS = 12    # topics drawn in the streamgraph
MIN_Q_VOLUME = 40    # quarters with fewer substantive msgs are dropped (sparse)

# register-noise to drop before clustering and exclude from labels: laughter,
# interjections, fillers, profanity, and common transliterated RU function words
# (Latin-charset but actually Russian — рут-noise the histograms never saw).
JUNK = set(
    """ха хах хаха хахах ахах ахаха хех хе хихи лол ор ору ржу ахаха
    ой ай эх ого ого-го вау воу упс мда хм гм эээ ааа ооо ммм нуу неа агась
    ням мур бла блаблабла ладненько окей оке ока окей-окей угум
    блин блить бля блядь пиздец пизда нахуй нахер хуй хуйня охуеть охуенно
    охренеть херня хрень фигня говно дерьмо жопа сука сучка ебать заебать
    ебаный долбоёб мудак придурок чёрт черт капец жесть
    lol lmao lmfao rofl haha hahaha hehe omg wtf btw idk imo tbh
    yeah yep yup nope ok okay okey oh ah hmm yes yay nah uh huh
    eto etot kak kakoy chto shto chtoto tebya menya tebe mne tvoy moy
    seychas seichas teper rabotaet rabota esli tak vot privet poka spasibo
    pozhaluysta gde kogda pochemu zachem budet bylo byla nado mozhno nelzya
    ochen prosto dlya eshe esche uzhe tozhe tolko ili potomu konechno davay
    normalno horosho ploho da net nu ty vy oni ona ono kotoryy""".split()
)


def meaningful(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in JUNK]


def cluster(emb: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    reduced = PCA(n_components=PCA_DIMS, random_state=seed).fit_transform(emb)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=2048, max_iter=200)
    return km.fit_predict(reduced)


def label_topics(topics: np.ndarray, tok_lists: list[list[str]]) -> dict[int, str]:
    """c-TF-IDF: each cluster is one document; return its top lemmas as a label."""
    counts: Counter = Counter()
    for topic, toks in zip(topics, tok_lists):
        for lemma in toks:
            counts[(int(topic), lemma)] += 1
    df = pd.DataFrame(
        [(t, l, n) for (t, l), n in counts.items()],
        columns=["doc", "lemma", "count"],
    )
    totals = df.groupby("doc")["count"].sum()
    n_docs = df.doc.nunique()
    doc_freq = df.groupby("lemma")["doc"].nunique()
    df = df[df["count"] >= 3].copy()
    df["tf"] = df["count"] / df.doc.map(totals)
    df["idf"] = np.log((1 + n_docs) / (1 + df.lemma.map(doc_freq)))
    df["tfidf"] = df.tf * df.idf
    df = df[df.tfidf > 0].sort_values("tfidf", ascending=False)

    labels: dict[int, str] = {}
    for topic, g in df.groupby("doc"):
        words = g.nlargest(TOP_WORDS, "tfidf").lemma.tolist()
        labels[int(topic)] = ", ".join(words) if words else f"topic {topic}"
    return labels


def chart_river(idx: pd.DataFrame, labels: dict[int, str], title: str,
                fname: str) -> Path:
    """Streamgraph of topic *share* per quarter (top RIVER_TOPICS)."""
    idx = idx.copy()
    idx["q"] = pd.PeriodIndex(idx.ts_local, freq="Q")
    qvol = idx.groupby("q").size()
    keep_q = qvol[qvol >= MIN_Q_VOLUME].index
    idx = idx[idx.q.isin(keep_q)]

    quarters = pd.period_range(idx.q.min(), idx.q.max(), freq="Q")
    total_per_q = idx.groupby("q").size().reindex(quarters, fill_value=0)
    top = idx.topic.value_counts().nlargest(RIVER_TOPICS).index.tolist()

    mat = np.zeros((len(top), len(quarters)))
    for ti, topic in enumerate(top):
        per_q = idx[idx.topic == topic].groupby("q").size()
        for qi, q in enumerate(quarters):
            denom = total_per_q.iloc[qi] or 1
            mat[ti, qi] = per_q.get(q, 0) / denom

    colors = plt.cm.turbo(np.linspace(0.04, 0.96, len(top)))
    fig, ax = plt.subplots(figsize=(16, 8.5))
    x = np.arange(len(quarters))
    ax.stackplot(x, mat, colors=colors,
                 labels=[labels.get(t, str(t))[:40] for t in top])

    year_starts = [i for i, q in enumerate(quarters) if q.quarter == 1]
    ax.set_xticks(year_starts)
    ax.set_xticklabels([str(quarters[i].year) for i in year_starts], fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_ylabel("share of substantive messages")
    ax.set_title(title, fontsize=17, pad=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=10,
              frameon=False, labelcolor="#e8e4d8")
    ax.margins(x=0, y=0)
    ax.set_ylim(0, min(1.0, mat.sum(0).max() * 1.02))

    out = config.VISUALIZATIONS / fname
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Topic river over Telegram")
    ap.add_argument("--k", type=int, default=40, help="number of topics")
    ap.add_argument("--person", help="restrict the river to one chat_name")
    args = ap.parse_args()

    emb, idx = emb_cache.load()
    idx = idx.reset_index(drop=True)
    print(f"loaded {len(idx):,} embeddings — tokenizing for content filter...")

    tok_lists = [meaningful(t) for t in idx["clean"]]
    substantive = np.array([len(t) >= MIN_MEANINGFUL for t in tok_lists])
    print(f"substantive: {substantive.sum():,} of {len(idx):,} "
          f"({100 * substantive.mean():.0f}%) — dropping register-noise")

    sub_idx = idx[substantive].reset_index(drop=True)
    sub_emb = emb[substantive]
    sub_toks = [t for t, k in zip(tok_lists, substantive) if k]

    sub_idx["topic"] = cluster(sub_emb, args.k)
    labels = label_topics(sub_idx["topic"].values, sub_toks)
    sub_idx["topic_label"] = sub_idx.topic.map(labels)

    sub_idx[["chat_id", "chat_name", "msg_html_id", "ts_utc", "is_me",
             "topic", "topic_label"]].to_parquet(config.TOPICS_PARQUET, index=False)
    print(f"wrote {len(sub_idx):,} message-topic rows to {config.TOPICS_PARQUET.name}")

    if args.person:
        view = sub_idx[sub_idx.chat_name == args.person]
        if view.empty:
            sys.exit(f"no messages for {args.person!r}")
        title = f"Topic river — {args.person}"
        fname = f"topic_river_{slug(args.person)}.png"
    else:
        view = sub_idx
        title = "Topic river — nine years of Telegram"
        fname = "telegram_topic_river.png"

    river = chart_river(view, labels, title, fname)
    print(f"river -> {river}")

    sizes = sub_idx.topic.value_counts()
    print(f"\ntop topics ({args.k} total):")
    for topic in sizes.nlargest(14).index:
        print(f"  [{sizes[topic]:>5}]  {labels.get(topic, '')[:66]}")


if __name__ == "__main__":
    main()
