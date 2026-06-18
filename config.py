"""Central paths for the personal data archive.

Code lives in this repo; data lives in the Desktop archive and never
gets committed. Every script imports its paths from here so moving the
archive later means editing one file.
"""
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent

RAW = ARCHIVE / "raw"
PROCESSED = ARCHIVE / "processed"
VISUALIZATIONS = ARCHIVE / "visualizations"
NOTES = ARCHIVE / "notes"

TELEGRAM_EXPORT = RAW / "telegram" / "DataExport_2026-06-08"
TELEGRAM_CHATS = TELEGRAM_EXPORT / "chats"
GMAIL_MBOX = RAW / "gmail" / "All mail Including Spam and Trash-002.mbox"
CALENDAR_DIR = RAW / "gmail" / "Takeout" / "Calendar"
MAPS_DIR = RAW / "gmail" / "Takeout" / "Maps"

TELEGRAM_PARQUET = PROCESSED / "telegram_messages.parquet"
WORD_FREQ_PARQUET = PROCESSED / "word_frequencies.parquet"
SIGNATURE_WORDS_PARQUET = PROCESSED / "signature_words.parquet"

# Phase 2 meaning layer: embeddings computed once, read by topics + sentiment.
MESSAGE_EMBEDDINGS = PROCESSED / "message_embeddings.npy"
MESSAGE_INDEX = PROCESSED / "message_index.parquet"
SENTIMENT_PARQUET = PROCESSED / "sentiment.parquet"
TOPICS_PARQUET = PROCESSED / "topics.parquet"

PEOPLE_YAML = ARCHIVE / "people.yaml"

for _d in (PROCESSED, VISUALIZATIONS, NOTES):
    _d.mkdir(parents=True, exist_ok=True)
