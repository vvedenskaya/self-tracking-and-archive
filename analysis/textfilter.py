"""Shared 'human voice' filter for the Phase 2 meaning-layer passes.

The lexical passes (word_histograms, signature_words) count *words*; the
meaning passes (topics, sentiment) read *meaning*, so they need to drop the
noise that carries none: media stubs, forwards, bare reactions, raw URLs,
and — optionally — code-debugging messages. Defining that once here means
topics and sentiment clean their input identically instead of drifting apart.

    from textfilter import human_voice
    df = human_voice(pd.read_parquet(config.TELEGRAM_PARQUET))   # adds .clean

`drop_code` defaults on. Topic modelling can leave it on (debugging is its own
dull cluster) or off (let "code talk" surface as a real topic); sentiment
should always drop it — snippets carry no feeling and only dilute aggregates.
"""
from __future__ import annotations

import re

import pandas as pd

# Matches word_histograms.URL_RE so both layers strip links the same way.
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|org|ru|net|io)\S*")
_WS_RE = re.compile(r"\s+")

# Code signals chosen to survive Russian chat, where ")" and "(" are smileys
# and laughter — never used here. Each entry is a distinct *kind* of evidence;
# a message is "code-heavy" only when several kinds co-occur (see THRESHOLD).
_CODE_SIGNALS = [
    re.compile(r"\b(def|class|return|import|from|await|async|lambda|elif|"
               r"except|raise|yield|kwargs|argv|stderr|stdout|uwsgi)\b"),
    re.compile(r"[a-z]+_[a-z]+(_[a-z]+)*"),          # snake_case identifiers
    re.compile(r"==|!=|->|=>|:=|\+=|\|\||&&|::"),     # operators
    re.compile(r"\b[a-z_][\w.]*\([\w\s,'\"=]*\)"),    # function call: name(...)
    re.compile(r"\{[^}]*:[^}]*\}|\[[^\]]*\]"),         # json / list literals
    re.compile(r"\bself\.\w|\w+\.\w+\(|/\w+/\w+"),     # attr access / paths
]
_CODE_THRESHOLD = 2  # distinct signal kinds required to call a message code


def clean_text(text: str | None) -> str:
    """Strip URLs and collapse whitespace; what the models actually read."""
    if not text:
        return ""
    return _WS_RE.sub(" ", URL_RE.sub(" ", text)).strip()


def is_code_heavy(text: str | None) -> bool:
    if not text:
        return False
    t = text.lower()
    hits = sum(bool(p.search(t)) for p in _CODE_SIGNALS)
    return hits >= _CODE_THRESHOLD


def human_voice(
    df: pd.DataFrame,
    min_words: int = 3,
    drop_code: bool = True,
    drop_forwarded: bool = True,
) -> pd.DataFrame:
    """Filter a telegram_messages frame down to conversational human text.

    Adds a `clean` column (URL-stripped) and keeps rows that are:
      - real text (not a media-only stub),
      - not forwarded (someone else's words),
      - at least `min_words` words after cleaning,
      - not code-heavy, when `drop_code`.

    Returns a copy; the original index is preserved so callers can join back
    to the full frame on (chat_id, msg_html_id).
    """
    out = df.copy()
    out["clean"] = out["text"].map(clean_text)

    keep = out["clean"].str.len() > 0
    if drop_forwarded and "is_forwarded" in out.columns:
        keep &= ~out["is_forwarded"].fillna(False)
    keep &= out["clean"].str.split().map(len) >= min_words
    if drop_code:
        keep &= ~out["clean"].map(is_code_heavy)

    return out[keep].reset_index(drop=True)
