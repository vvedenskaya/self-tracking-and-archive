"""Emotion scoring over Telegram messages — the archive's emotional weather.

Runs a small RU emotion classifier (rubert-tiny2, CEDR labels: joy, sadness,
surprise, fear, anger, no_emotion) over every substantive message, then leaves
aggregation to the reader. Per-message scores are noise; only (chat × month)
means are ever meant to be read — emotional weather, warmth asymmetry, and a
sentiment-grounded rebuild of the conflict heatmap.

Reuses message_index.parquet (the 88k human-voice messages) so it lines up with
the embedding/topic passes. Resumable + chunked + self-exiting, exactly like
embeddings.py, so it finishes inside the sandbox without disabling anything.

    python analysis/sentiment.py            # score a slice, then exit
    (repeat until it prints FINISHED)
    python analysis/sentiment.py --force    # start over

Model is RU-centric: English messages score less reliably, but aggregates wash
most of that out and the archive is mostly Russian.
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

MODEL = "cointegrated/rubert-tiny2-cedr-emotion-detection"
CHUNK = 2000
TIME_BUDGET = 200      # seconds, then exit so the sandbox never kills us
CHUNK_DIR = config.PROCESSED / "sent_chunks"
LABELS = ["no_emotion", "joy", "sadness", "surprise", "fear", "anger"]


def _load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    # trust the model's own label order if it exposes one
    id2label = getattr(model.config, "id2label", None)
    labels = ([id2label[i] for i in range(len(id2label))]
              if id2label and len(id2label) == 6 else LABELS)

    @torch.no_grad()
    def score(texts: list[str]) -> np.ndarray:
        enc = tok(texts, return_tensors="pt", truncation=True,
                  padding=True, max_length=128)
        probs = torch.sigmoid(model(**enc).logits)  # CEDR is multi-label
        return probs.numpy().astype(np.float32)

    return score, labels


def _finalize(idx: pd.DataFrame, n_chunks: int, labels: list[str]) -> None:
    parts = [np.load(CHUNK_DIR / f"chunk_{i:05d}.npy") for i in range(n_chunks)]
    probs = np.vstack(parts).astype(np.float32)
    assert len(probs) == len(idx), "chunk rows != index rows"
    out = idx[["chat_id", "chat_name", "msg_html_id", "ts_utc", "ts_local",
               "is_me", "is_code"]].reset_index(drop=True)
    for j, lab in enumerate(labels):
        out[lab] = probs[:, j]
    out.to_parquet(config.SENTIMENT_PARQUET, index=False)
    shutil.rmtree(CHUNK_DIR, ignore_errors=True)
    print(f"FINISHED — saved {probs.shape} -> {config.SENTIMENT_PARQUET.name}")


def build(force: bool = False) -> None:
    if force:
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
        config.SENTIMENT_PARQUET.unlink(missing_ok=True)
    if config.SENTIMENT_PARQUET.exists():
        print("sentiment cache present — pass --force to rebuild")
        return
    if not config.MESSAGE_INDEX.exists():
        sys.exit("message_index.parquet missing — run embeddings.py first")

    idx = pd.read_parquet(config.MESSAGE_INDEX)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = -(-len(idx) // CHUNK)
    done = sum((CHUNK_DIR / f"chunk_{i:05d}.npy").exists() for i in range(n_chunks))
    print(f"{len(idx):,} messages in {n_chunks} chunks · {done} already done")

    score, labels = (None, LABELS)
    start = time.time()
    did = 0
    for ci in range(n_chunks):
        out = CHUNK_DIR / f"chunk_{ci:05d}.npy"
        if out.exists():
            continue
        if did > 0 and time.time() - start > TIME_BUDGET:
            break
        if score is None:
            score, labels = _load_model()
        rows = idx.iloc[ci * CHUNK:(ci + 1) * CHUNK]
        np.save(out, score(rows["clean"].tolist()))
        did += 1
        print(f"  chunk {ci + 1}/{n_chunks} done ({done + did}/{n_chunks} total)",
              flush=True)

    if done + did >= n_chunks:
        _finalize(idx, n_chunks, labels)
    else:
        print(f"PAUSED — {n_chunks - (done + did)} chunks left, run again")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score message emotions")
    ap.add_argument("--force", action="store_true", help="rebuild from scratch")
    build(ap.parse_args().force)


if __name__ == "__main__":
    main()
