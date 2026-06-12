"""Most-used words across the Telegram history (RU + EN).

- tokenizes all message text, lowercased
- lemmatizes Russian words with pymorphy3 so думаю/думала/думаешь count once
- drops RU + EN stopwords and one/two-letter tokens
- writes processed/word_frequencies.parquet (lemma, is_me, year, count)
  for reuse by later art pieces, plus histogram PNGs

Usage: python analysis/word_histograms.py
"""
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pymorphy3

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

TOKEN_RE = re.compile(r"[а-яёa-z]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|org|ru|net|io)\S*")

RU_STOP = set(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за
    бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну
    вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
    потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была
    сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому
    этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
    куда зачем всех никогда можно при наконец два об другой хоть после над больше
    тот через эти нас про всего них какая много разве три эту моя впрочем хорошо
    свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
    между это весь свой мочь это-то ну-ка ага ах ой эх ок окей ладно типа короче
    просто очень вообще кстати блин пока привет спасибо пожалуйста давай давайте
    тебе тобой мной нам вами ими нему ней ничто никто весь вся всё мои твои твой
    твоя наш ваш который которая которое которые она оно мы-то да-да угу ща щас
    ещё еще
    """.split()
)

EN_STOP = set(
    """the a an and or but if then else when at by for with about against between
    into through during before after above below to from up down in out on off
    over under again further once here there all any both each few more most
    other some such no nor not only own same so than too very s t can will just
    don should now i me my myself we our ours ourselves you your yours yourself
    yourselves he him his himself she her hers herself it its itself they them
    their theirs themselves what which who whom this that these those am is are
    was were be been being have has had having do does did doing would could
    might must shall ok okay yeah yes hi hello hey thanks thank please like get
    got im dont its lol haha
    """.split()
)


@lru_cache(maxsize=200_000)
def lemmatize_ru(word: str, _morph=pymorphy3.MorphAnalyzer()) -> str:
    return _morph.parse(word)[0].normal_form


def tokens(text: str):
    for w in TOKEN_RE.findall(URL_RE.sub(" ", text.lower())):
        if len(w) < 3:
            continue
        if re.match(r"[а-яё]", w):
            lemma = lemmatize_ru(w)
            if lemma not in RU_STOP and len(lemma) >= 3:
                yield lemma
        elif w not in EN_STOP:
            yield w


def count_words(df: pd.DataFrame) -> pd.DataFrame:
    counts: Counter = Counter()
    for is_me, year, text in zip(df.is_me, df.ts_local.dt.year, df.text):
        if not text:
            continue
        for lemma in tokens(text):
            counts[(lemma, is_me, year)] += 1
    out = pd.DataFrame(
        [(k[0], k[1], k[2], v) for k, v in counts.items()],
        columns=["lemma", "is_me", "year", "count"],
    )
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def chart_me_vs_them(freq: pd.DataFrame, n: int = 30) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 10))
    for ax, mine, color, title in (
        (axes[0], True, "#e8b84b", "words I use most"),
        (axes[1], False, "#e06c9f", "words said to me"),
    ):
        top = (
            freq[freq.is_me == mine].groupby("lemma")["count"].sum().nlargest(n).iloc[::-1]
        )
        ax.barh(top.index, top.values, color=color)
        ax.set_title(title, fontsize=14, pad=12)
        ax.tick_params(labelsize=10)
    fig.suptitle("Most used words — nine years of Telegram", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / "telegram_words_me_vs_them.png")
    plt.close(fig)


def chart_by_year(freq: pd.DataFrame, n: int = 12) -> None:
    mine = freq[freq.is_me]
    years = sorted(mine.year.unique())
    cols = 4
    rows = -(-len(years) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows))
    for ax, year in zip(axes.flat, years):
        top = mine[mine.year == year].groupby("lemma")["count"].sum().nlargest(n).iloc[::-1]
        ax.barh(top.index, top.values, color="#5fc8c8")
        ax.set_title(str(year), fontsize=13)
        ax.tick_params(labelsize=9)
    for ax in axes.flat[len(years):]:
        ax.axis("off")
    fig.suptitle("My vocabulary, year by year", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / "telegram_words_by_year.png")
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    print(f"counting words in {len(df):,} messages (lemmatizing RU)...")
    freq = count_words(df)
    freq.to_parquet(config.WORD_FREQ_PARQUET, index=False)
    print(f"wrote {len(freq):,} (lemma, speaker, year) rows to {config.WORD_FREQ_PARQUET}")

    chart_me_vs_them(freq)
    chart_by_year(freq)
    print(f"wrote 2 histogram charts to {config.VISUALIZATIONS}")

    top10 = freq[freq.is_me].groupby("lemma")["count"].sum().nlargest(10)
    print("\nmy top 10 words:", ", ".join(f"{w} ({c})" for w, c in top10.items()))


if __name__ == "__main__":
    main()
