"""Conflict heatmap for a Telegram relationship.

Scores messages for tension/anger language (RU + EN), clusters them into
episodes, attributes who escalated first, and writes charts to visualizations/.

Usage:
    python analysis/conflict_heatmap.py                  # Sofia Dro (default)
    python analysis/conflict_heatmap.py "Sofia Dro"
    python analysis/conflict_heatmap.py --person sofia_dro
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

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

GOLD = "#e8b84b"
ROSE = "#e06c9f"
TEAL = "#5fc8c8"
AMBER = "#f0a030"
DEEP = "#1a1428"

STRONG_PATTERNS = [
    r"обидно", r"обидел", r"обидела", r"обиделась", r"обиделся", r"мне обидно",
    r"злюсь", r"злишься", r"разозл", r"злой\b", r"злая\b", r"зла\b",
    r"бесишь", r"раздражаешь", r"надоел\b", r"надоела\b",
    r"достал меня", r"достала меня", r"ты меня достал", r"ты меня достала",
    r"грубо", r"грубишь", r"грубость", r"грубил", r"грубила", r"груба\b", r"груб\b",
    r"несправедлив", r"unfair", r"не права", r"не прав\b", r"не права\b",
    r"манипул", r"границ", r"boundary", r"boundaries",
    r"не понимаешь", r"не слушаешь", r"не слышишь",
    r"виноват", r"виновата", r"обвиня", r"неуваж", r"предал", r"предала",
    r"поссор", r"ссор", r"конфликт", r"ругаем", r"ругает", r"руга",
    r"отстань", r"отвали", r"заткнись", r"leave me alone",
    r"не хочу с тобой", r"надоело", r"нечестно", r"перестань", r"хватит уже",
    r"кричал", r"кричала", r"кричишь", r"кричала на", r"кричала на тебя",
    r"агрессив", r"токсич", r"триггер",
    r"почему ты так", r"зачем ты так",
    r"извини но", r"прости но", r"sorry but",
    r"не могу так", r"не буду терпеть",
    r"you always", r"you never", r"ты всегда", r"ты никогда",
    r"respect", r"уважай",
    r"расстроил", r"расстроила", r"расстроен", r"расстроена",
    r"hurt\b", r"upset", r"angry",
    r"не могу больше", r"устала от тебя", r"устал от тебя",
    r"не нравится как", r"не нравится что ты",
    r"давишь", r"давление", r"игнорир", r"молчишь специально",
    r"разочаров", r"disappoint", r"некомфорт", r"uncomfortable",
    r"переступил", r"переступила", r"потрескались границы",
    r"не ok\b", r"not ok", r"корябает", r"копится",
    r"процессир", r"не остановить",
    r"не подходит", r"продвигаешь",
]

MEDIUM_PATTERNS = [
    r"обид", r"зл\b", r"бесит", r"раздраж", r"груб", r"неправ",
    r"прости", r"извини", r"sorry", r"apolog",
    r"тяжело", r"плохо\b", r"ужас",
    r"не понимаю тебя", r"зачем", r"почему",
    r"хватит", r"не могу", r"can't", r"stop\b", r"стоп\b",
    r"не хочу", r"don't want", r"проблем", r"problem",
    r"annoy", r"доста", r"wtf", r"блин\b", r"фак\b",
    r"pressure", r"weird",
]

BOUNDARY_PATTERNS = [
    r"границ", r"boundary", r"boundaries", r"переступил", r"переступила",
    r"потрескались границы", r"неуваж", r"давишь", r"давление",
    r"не подходит", r"продвигаешь", r"не остановить", r"манипул",
    r"несправедлив", r"unfair", r"нечестно", r"не права", r"не прав\b",
    r"токсич", r"crossed",
]


@dataclass
class Episode:
    start: pd.Timestamp
    end: pd.Timestamp
    score: float
    n_msgs: int
    initiated_by_me: bool
    peak_text: str


def resolve_chat_name(person_arg: str | None, chat_arg: str | None) -> str:
    if chat_arg:
        return chat_arg
    if not person_arg:
        return "Sofia Dro"
    if person_arg in ("sofia_dro", "sofia"):
        return "Sofia Dro"
    registry = yaml.safe_load(config.PEOPLE_YAML.read_text(encoding="utf-8")) or []
    for entry in registry:
        if entry["id"] == person_arg:
            tg = entry.get("telegram") or []
            if tg:
                return tg[0]
            return entry["display"]
    return person_arg


def score_message(text: str) -> float:
    if not text or len(text.strip()) < 3:
        return 0.0
    t = text.lower()
    score = 0.0
    for p in STRONG_PATTERNS:
        if re.search(p, t, re.I):
            score += 3.0
    for p in MEDIUM_PATTERNS:
        if re.search(p, t, re.I):
            score += 1.0
    if text.count("!") >= 3:
        score += 1.0
    if re.search(r"[A-ZА-Я]{4,}", text):
        score += 1.0
    return score


def is_boundary(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.I) for p in BOUNDARY_PATTERNS)


def load_chat(chat_name: str) -> pd.DataFrame:
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    chat = df[df.chat_name == chat_name].sort_values("ts_utc").reset_index(drop=True)
    if chat.empty:
        raise SystemExit(f"no messages for chat {chat_name!r}")
    chat["score"] = chat.text.map(score_message)
    chat["boundary"] = chat.text.map(is_boundary)
    return chat


def detect_episodes(chat: pd.DataFrame) -> list[Episode]:
    """Cluster tense exchanges; attribute initiator to first score>=2 message."""
    gap = timedelta(hours=8)
    thresh = 2.0
    min_total = 4.0

    episodes: list[Episode] = []
    i = 0
    n = len(chat)
    while i < n:
        if chat.loc[i, "score"] < thresh:
            i += 1
            continue
        start = i
        end = i
        total = chat.loc[i, "score"]
        while end + 1 < n:
            gap_t = chat.loc[end + 1, "ts_utc"] - chat.loc[end, "ts_utc"]
            nxt = chat.loc[end + 1, "score"]
            same_burst = gap_t <= gap and (
                nxt >= 1.0 or chat.loc[end + 1, "is_me"] != chat.loc[end, "is_me"]
            )
            late_escalation = nxt >= thresh and gap_t <= timedelta(hours=24)
            if same_burst or late_escalation:
                end += 1
                total += chat.loc[end, "score"]
            else:
                break
        if total >= min_total:
            chunk = chat.loc[start:end]
            escalators = chunk[chunk.score >= thresh]
            initiator_me = bool(
                escalators.iloc[0]["is_me"] if len(escalators) else chunk.iloc[0]["is_me"]
            )
            peak = chunk.loc[chunk.score.idxmax()]
            episodes.append(
                Episode(
                    start=chunk.iloc[0]["ts_local"],
                    end=chunk.iloc[-1]["ts_local"],
                    score=total,
                    n_msgs=len(chunk),
                    initiated_by_me=initiator_me,
                    peak_text=peak["text"][:160],
                )
            )
        i = end + 1
    return episodes


def monthly_matrix(chat: pd.DataFrame, episodes: list[Episode]) -> pd.DataFrame:
    chat = chat.copy()
    chat["month"] = chat.ts_local.dt.to_period("M")
    monthly_score = chat.groupby("month")["score"].sum()
    monthly_msgs = chat.groupby("month").apply(lambda g: (g.score >= 2).sum())

    ep_rows = []
    for ep in episodes:
        month = ep.start.to_period("M")
        ep_rows.append(
            {
                "month": month,
                "score": ep.score,
                "her_initiated": not ep.initiated_by_me,
            }
        )
    ep_df = pd.DataFrame(ep_rows) if ep_rows else pd.DataFrame(columns=["month", "score", "her_initiated"])

    months = pd.period_range(chat.ts_local.min(), chat.ts_local.max(), freq="M")
    rows = []
    for m in months:
        her_init = int(ep_df[(ep_df.month == m) & ep_df.her_initiated].shape[0]) if len(ep_df) else 0
        me_init = int(ep_df[(ep_df.month == m) & ~ep_df.her_initiated].shape[0]) if len(ep_df) else 0
        rows.append(
            {
                "month": m,
                "conflict_score": float(monthly_score.get(m, 0)),
                "tense_msgs": int(monthly_msgs.get(m, 0)),
                "episodes": her_init + me_init,
                "her_initiated": her_init,
                "me_initiated": me_init,
            }
        )
    return pd.DataFrame(rows)


def slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_")


def chart_heatmap(chat_name: str, chat: pd.DataFrame, monthly: pd.DataFrame, episodes: list[Episode]) -> Path:
    """Year × month grid: color = conflict intensity; dots = episodes by initiator."""
    out = config.VISUALIZATIONS / f"conflict_heatmap_{slug(chat_name)}.png"

    years = sorted(monthly.month.dt.year.unique())
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    grid = np.zeros((len(years), 12))
    her_dots: list[tuple[int, int]] = []
    me_dots: list[tuple[int, int]] = []

    for _, row in monthly.iterrows():
        y_idx = years.index(row.month.year)
        m_idx = row.month.month - 1
        grid[y_idx, m_idx] = row.conflict_score
        if row.her_initiated:
            her_dots.append((y_idx, m_idx))
        if row.me_initiated:
            me_dots.append((y_idx, m_idx))

    vmax = max(grid.max(), 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(years)), years)
    ax.set_xticks(range(12), month_labels)
    ax.set_title(
        f"Conflict heatmap — {chat_name}\n"
        f"{len(chat):,} messages · {chat.ts_local.min():%b %Y} – {chat.ts_local.max():%b %Y}",
        fontsize=15,
        pad=14,
    )
    for yi, mi in her_dots:
        ax.scatter(mi, yi, s=80, marker="v", c=ROSE, edgecolors="white", linewidths=0.6, zorder=5)
    for yi, mi in me_dots:
        ax.scatter(mi, yi, s=80, marker="^", c=GOLD, edgecolors="white", linewidths=0.6, zorder=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("tension score (keyword-weighted sum)", color="#9a96a8")

    her_patch = mpatches.Patch(color=ROSE, label="episode started by her")
    me_patch = mpatches.Patch(color=GOLD, label="episode started by me")
    ax.legend(handles=[her_patch, me_patch], loc="upper left", frameon=False, fontsize=9)

    ax2 = axes[1]
    her_n = sum(not ep.initiated_by_me for ep in episodes)
    me_n = sum(ep.initiated_by_me for ep in episodes)
    total = her_n + me_n or 1
    bars = ax2.barh(["Her initiated", "I initiated"], [her_n, me_n], color=[ROSE, GOLD], height=0.5)
    ax2.set_xlim(0, max(her_n, me_n) * 1.25 + 1)
    ax2.set_title(
        f"Who escalated first — {len(episodes)} detected episodes "
        f"(her {her_n}/{total} = {100 * her_n / total:.0f}%)",
        fontsize=12,
        pad=10,
    )
    for bar, n, pct in zip(bars, [her_n, me_n], [100 * her_n / total, 100 * me_n / total]):
        ax2.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                 f"{n} ({pct:.0f}%)", va="center", fontsize=11, color="#e8e4d8")
    ax2.grid(axis="x", color="#262236", linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_timeline(chat_name: str, chat: pd.DataFrame, episodes: list[Episode]) -> Path:
    """Rolling monthly tension + episode markers."""
    out = config.VISUALIZATIONS / f"conflict_timeline_{slug(chat_name)}.png"

    monthly = chat.set_index("ts_local").resample("ME")["score"].sum()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(monthly.index, monthly.values, alpha=0.35, color=AMBER)
    ax.plot(monthly.index, monthly.values, color=AMBER, lw=1.5)

    for ep in episodes:
        color = ROSE if not ep.initiated_by_me else GOLD
        ax.axvline(ep.start, color=color, alpha=0.35, lw=0.8)

    ax.set_title(f"Tension over time — {chat_name}", fontsize=14, pad=12)
    ax.set_ylabel("monthly tension score")
    ax.grid(True, color="#262236", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def write_report(chat_name: str, chat: pd.DataFrame, episodes: list[Episode], paths: list[Path]) -> Path:
    out = config.NOTES / f"conflict_report_{slug(chat_name)}.md"
    her_n = sum(not ep.initiated_by_me for ep in episodes)
    me_n = sum(ep.initiated_by_me for ep in episodes)
    total = her_n + me_n or 1
    boundary_hits = chat[chat.boundary & ~chat.is_me]
    boundary_me = chat[chat.boundary & chat.is_me]

    lines = [
        f"# Conflict report — {chat_name}",
        "",
        f"- Messages: {len(chat):,} ({chat.is_me.sum():,} mine, {(~chat.is_me).sum():,} hers)",
        f"- Span: {chat.ts_local.min():%Y-%m-%d} to {chat.ts_local.max():%Y-%m-%d}",
        f"- Detected episodes: {len(episodes)}",
        f"- Initiated by her: **{her_n}** ({100 * her_n / total:.0f}%)",
        f"- Initiated by me: **{me_n}** ({100 * me_n / total:.0f}%)",
        f"- Boundary/unfair-tagged messages (hers): {len(boundary_hits)}",
        f"- Boundary/unfair-tagged messages (mine): {len(boundary_me)}",
        "",
        "## Episodes (newest first)",
        "",
    ]
    for ep in sorted(episodes, key=lambda e: e.start, reverse=True):
        who = "me" if ep.initiated_by_me else "her"
        lines.append(
            f"- **{ep.start:%Y-%m-%d %H:%M}** — started by {who}, "
            f"score {ep.score:.0f}, {ep.n_msgs} msgs"
        )
        snippet = ep.peak_text.replace("\n", " ")
        lines.append(f"  - peak: _{snippet}_")
        lines.append("")

    lines.extend(["## Charts", ""] + [f"- `{p.name}`" for p in paths])
    lines.extend(
        [
            "",
            "_Method: keyword-weighted tension scoring (RU/EN anger, hurt, boundaries, "
            "unfairness). Episodes = clusters of score≥2 messages within 8h, min total "
            "score 4. Initiator = first score≥2 message in cluster._",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Conflict heatmap for a Telegram chat")
    parser.add_argument("chat", nargs="?", help="Telegram chat name (default: Sofia Dro)")
    parser.add_argument("--person", help="people.yaml id (e.g. sofia_dro)")
    args = parser.parse_args()

    chat_name = resolve_chat_name(args.person, args.chat)
    chat = load_chat(chat_name)
    episodes = detect_episodes(chat)
    monthly = monthly_matrix(chat, episodes)

    p1 = chart_heatmap(chat_name, chat, monthly, episodes)
    p2 = chart_timeline(chat_name, chat, episodes)
    report = write_report(chat_name, chat, episodes, [p1, p2])

    her_n = sum(not ep.initiated_by_me for ep in episodes)
    me_n = len(episodes) - her_n
    print(f"{chat_name}: {len(chat):,} messages, {len(episodes)} conflict episodes")
    print(f"  initiated by her: {her_n}  |  by you: {me_n}")
    print(f"  heatmap -> {p1}")
    print(f"  timeline -> {p2}")
    print(f"  report -> {report}")


if __name__ == "__main__":
    main()
