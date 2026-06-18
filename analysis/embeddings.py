"""Embed Telegram messages once; topics and sentiment both read the cache.

Embedding 88k messages is the expensive Phase 2 step, so it happens here once.
Output is a row-aligned pair:

    message_embeddings.npy   float32 [N, 384]   the vectors
    message_index.parquet    N rows             keys + metadata, same order

Resumable on purpose. Each run embeds messages in chunks, saves every chunk to
disk, and *exits itself* after TIME_BUDGET seconds — short enough to finish
inside the command sandbox's limits, so we never have to disable it. Re-run
until it prints "FINISHED"; already-done chunks are skipped, so a kill mid-run
loses at most one chunk.

Model: paraphrase-multilingual-MiniLM-L12-v2 — 384-dim, RU+EN, CPU-friendly.

    python analysis/embeddings.py            # do a slice, then exit
    (repeat until it says FINISHED)
    python analysis/embeddings.py --force    # start over
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from textfilter import human_voice, is_code_heavy

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_COLS = ["chat_id", "chat_name", "msg_html_id", "ts_utc", "ts_local",
              "is_me", "is_code", "clean"]

CHUNK = 2000        # messages embedded per chunk file
TIME_BUDGET = 200   # seconds, then exit cleanly so the sandbox never kills us
CHUNK_DIR = config.PROCESSED / "emb_chunks"
IDX_TMP = CHUNK_DIR / "_index.parquet"


def _prepare_index() -> pd.DataFrame:
    """Build (once) the deterministic, ordered list of messages to embed."""
    if IDX_TMP.exists():
        return pd.read_parquet(IDX_TMP)
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    hv = human_voice(df, min_words=3, drop_code=False).reset_index(drop=True)
    hv["is_code"] = hv["clean"].map(is_code_heavy)
    hv = hv[INDEX_COLS]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    hv.to_parquet(IDX_TMP, index=False)
    return hv


def _finalize(idx: pd.DataFrame, n_chunks: int) -> None:
    parts = [np.load(CHUNK_DIR / f"chunk_{i:05d}.npy") for i in range(n_chunks)]
    emb = np.vstack(parts).astype(np.float32)
    assert len(emb) == len(idx), "chunk rows != index rows"
    np.save(config.MESSAGE_EMBEDDINGS, emb)
    idx.reset_index(drop=True).to_parquet(config.MESSAGE_INDEX, index=False)
    shutil.rmtree(CHUNK_DIR, ignore_errors=True)
    print(f"FINISHED — saved {emb.shape} -> {config.MESSAGE_EMBEDDINGS.name} "
          f"+ {config.MESSAGE_INDEX.name}")


def build(force: bool = False) -> None:
    if force:
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
        config.MESSAGE_EMBEDDINGS.unlink(missing_ok=True)
        config.MESSAGE_INDEX.unlink(missing_ok=True)
    if config.MESSAGE_EMBEDDINGS.exists() and config.MESSAGE_INDEX.exists():
        print("cache present — pass --force to rebuild")
        return

    idx = _prepare_index()
    n_chunks = -(-len(idx) // CHUNK)
    done = sum((CHUNK_DIR / f"chunk_{i:05d}.npy").exists() for i in range(n_chunks))
    print(f"{len(idx):,} messages in {n_chunks} chunks · {done} already done")

    if done >= n_chunks:
        _finalize(idx, n_chunks)
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    start = time.time()
    did = 0
    for ci in range(n_chunks):
        out = CHUNK_DIR / f"chunk_{ci:05d}.npy"
        if out.exists():
            continue
        if did > 0 and time.time() - start > TIME_BUDGET:
            break
        rows = idx.iloc[ci * CHUNK:(ci + 1) * CHUNK]
        emb = model.encode(rows["clean"].tolist(), batch_size=64,
                           convert_to_numpy=True, normalize_embeddings=True)
        np.save(out, emb.astype(np.float32))
        did += 1
        print(f"  chunk {ci + 1}/{n_chunks} done ({done + did}/{n_chunks} total)",
              flush=True)

    remaining = n_chunks - (done + did)
    if remaining <= 0:
        _finalize(idx, n_chunks)
    else:
        print(f"PAUSED — {remaining} chunks left, run again to continue")


def load() -> tuple[np.ndarray, pd.DataFrame]:
    """Return (embeddings [N,384], index DataFrame) in matching row order."""
    emb = np.load(config.MESSAGE_EMBEDDINGS)
    idx = pd.read_parquet(config.MESSAGE_INDEX)
    assert len(emb) == len(idx), "embeddings and index out of sync — rebuild"
    return emb, idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the message embedding cache")
    ap.add_argument("--force", action="store_true", help="rebuild from scratch")
    build(ap.parse_args().force)


if __name__ == "__main__":
    main()
