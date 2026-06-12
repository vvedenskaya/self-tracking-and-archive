"""Telegram analysis charts: volume over time, relationships, daily rhythms.

Reads processed/telegram_messages.parquet, writes PNGs to visualizations/.

Usage: python analysis/telegram_analysis.py
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
        "axes.grid": True,
        "grid.color": "#262236",
        "grid.linewidth": 0.6,
        "font.family": "Helvetica Neue",
        "figure.dpi": 150,
    }
)

GOLD = "#e8b84b"
ROSE = "#e06c9f"
TEAL = "#5fc8c8"


def load() -> pd.DataFrame:
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    df["month"] = df.ts_local.dt.to_period("M").dt.to_timestamp()
    return df


def chart_volume(df: pd.DataFrame) -> None:
    """Messages per month, mine vs. theirs, across the whole archive."""
    monthly = df.groupby(["month", "is_me"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly.index, monthly[True], width=25, color=GOLD, label="my messages")
    ax.bar(monthly.index, monthly[False], width=25, bottom=monthly[True],
           color=ROSE, alpha=0.75, label="messages to me")
    ax.set_title("Nine years of conversation — messages per month", fontsize=15, pad=14)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / "telegram_volume_monthly.png")
    plt.close(fig)


def chart_top_chats(df: pd.DataFrame, n: int = 25) -> None:
    """The n biggest relationships by total message count."""
    top = df.groupby("chat_name").size().nlargest(n).iloc[::-1]
    share_me = df[df.chat_name.isin(top.index)].groupby("chat_name").is_me.mean()
    fig, ax = plt.subplots(figsize=(10, 11))
    bars = ax.barh(top.index, top.values, color=GOLD)
    for bar, name in zip(bars, top.index):
        bar.set_color(plt.cm.RdYlBu(share_me[name]))  # red = they talk, blue = I talk
    ax.set_title(f"Top {n} conversations (color = who speaks more:\n"
                 "red = mostly them, blue = mostly me)", fontsize=14, pad=14)
    ax.set_xlabel("messages")
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / "telegram_top_chats.png")
    plt.close(fig)


def chart_rhythm(df: pd.DataFrame) -> None:
    """Hour-of-day x weekday heatmap in local time: the shape of your days."""
    pivot = (
        df.assign(hour=df.ts_local.dt.hour, weekday=df.ts_local.dt.dayofweek)
        .pivot_table(index="hour", columns="weekday", values="text", aggfunc="size", fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(pivot, aspect="auto", cmap="magma")
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_title("Daily rhythm — when messages happen\n(local time, all years)", fontsize=14, pad=14)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="messages")
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / "telegram_daily_rhythm.png")
    plt.close(fig)


def chart_lifespans(df: pd.DataFrame, n: int = 30) -> None:
    """When each big relationship started, peaked, faded: one line per chat."""
    top_names = df.groupby("chat_name").size().nlargest(n).index
    fig, ax = plt.subplots(figsize=(13, 11))
    for y, name in enumerate(reversed(list(top_names))):
        sub = df[df.chat_name == name]
        monthly = sub.groupby("month").size()
        sizes = 4 + 40 * (monthly / monthly.max())
        ax.scatter(monthly.index, [y] * len(monthly), s=sizes, color=TEAL, alpha=0.55,
                   edgecolors="none")
        ax.plot([sub.ts_local.min(), sub.ts_local.max()], [y, y],
                color="#3a3650", lw=0.8, zorder=0)
        label = name if len(name) <= 24 else name[:23] + "…"
        ax.text(df.ts_local.min(), y, label + "  ", ha="right", va="center",
                fontsize=8, color="#c8c4d8")
    ax.set_yticks([])
    ax.set_xlim(df.ts_local.min() - pd.Timedelta(days=700), df.ts_local.max())
    ax.set_title("Conversation lifespans — when each relationship was alive\n"
                 "(dot size = messages that month)", fontsize=14, pad=14)
    fig.tight_layout()
    fig.savefig(config.VISUALIZATIONS / "telegram_lifespans.png")
    plt.close(fig)


def main() -> None:
    df = load()
    print(f"loaded {len(df):,} messages, {df.chat_id.nunique()} chats")
    chart_volume(df)
    chart_top_chats(df)
    chart_rhythm(df)
    chart_lifespans(df)
    print(f"wrote 4 charts to {config.VISUALIZATIONS}")


if __name__ == "__main__":
    main()
