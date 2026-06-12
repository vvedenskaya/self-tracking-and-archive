"""Parse a Telegram HTML export into one Parquet table.

Walks every chat_* folder, reads messages*.html in order, and emits one row
per message: timestamp (UTC + local), chat, sender, text, media type.

The export owner ("me") is not named anywhere in the HTML, so we detect it:
the sender who appears in the greatest number of distinct chats is the owner.

Usage: python parsers/telegram.py
"""
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lxml.html
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

TS_FORMAT = "%d.%m.%Y %H:%M:%S UTC%z"

MEDIA_CLASSES = {
    "media_photo": "photo",
    "photo_wrap": "photo",
    "media_video": "video",
    "video_file_wrap": "video",
    "media_voice_message": "voice",
    "media_audio_file": "audio",
    "media_file": "file",
    "sticker_wrap": "sticker",
    "media_animated": "gif",
    "media_call": "call",
    "media_location": "location",
    "media_contact": "contact",
    "media_poll": "poll",
    "media_game": "game",
    "media_invoice": "invoice",
}


def sort_key(p: Path) -> int:
    m = re.search(r"messages(\d*)\.html$", p.name)
    return int(m.group(1)) if m.group(1) else 1


def detect_media(msg_el) -> str | None:
    for el in msg_el.iter():
        classes = el.get("class") or ""
        for cls, label in MEDIA_CLASSES.items():
            if cls in classes:
                return label
    return None


def parse_chat(chat_dir: Path) -> list[dict]:
    files = sorted(chat_dir.glob("messages*.html"), key=sort_key)
    if not files:
        return []

    rows = []
    chat_name = None
    last_sender = None

    for f in files:
        tree = lxml.html.parse(str(f)).getroot()

        if chat_name is None:
            header = tree.cssselect("div.page_header div.text.bold")
            chat_name = header[0].text_content().strip() if header else chat_dir.name

        for msg in tree.cssselect("div.message.default"):
            date_el = msg.cssselect("div.pull_right.date.details")
            if not date_el:
                continue
            title = date_el[0].get("title")
            if not title:
                continue
            try:
                ts = datetime.strptime(title.strip(), TS_FORMAT)
            except ValueError:
                continue

            from_el = msg.cssselect("div.body > div.from_name")
            if from_el:
                # strip "via @bot" suffixes that live in a child span
                sender = from_el[0].text.strip() if from_el[0].text else \
                    from_el[0].text_content().strip()
                last_sender = sender
            else:
                sender = last_sender  # "joined" message: same sender as previous

            text_el = msg.cssselect("div.body div.text")
            text = text_el[0].text_content().strip() if text_el else ""

            is_forwarded = bool(msg.cssselect("div.forwarded.body"))

            rows.append(
                {
                    "chat_id": chat_dir.name,
                    "chat_name": chat_name,
                    "msg_html_id": msg.get("id", ""),
                    "ts_local": ts.replace(tzinfo=None),
                    "tz_offset": ts.strftime("%z"),
                    "ts_utc": ts.astimezone(timezone.utc).replace(tzinfo=None),
                    "sender": sender or "",
                    "text": text,
                    "media_type": detect_media(msg),
                    "is_forwarded": is_forwarded,
                }
            )
    return rows


def main() -> None:
    chat_dirs = sorted(
        d for d in config.TELEGRAM_CHATS.iterdir()
        if d.is_dir() and d.name.startswith("chat_")
    )
    print(f"parsing {len(chat_dirs)} chats from {config.TELEGRAM_CHATS}")

    all_rows: list[dict] = []
    for i, chat_dir in enumerate(chat_dirs, 1):
        all_rows.extend(parse_chat(chat_dir))
        if i % 50 == 0 or i == len(chat_dirs):
            print(f"  {i}/{len(chat_dirs)} chats, {len(all_rows)} messages")

    df = pd.DataFrame(all_rows)

    # Owner detection: the account holder speaks in more distinct chats
    # than anyone else by a wide margin.
    chats_per_sender = (
        df[df.sender != ""].groupby("sender")["chat_id"].nunique().sort_values(ascending=False)
    )
    owner = chats_per_sender.index[0]
    print(f"\ndetected export owner: {owner!r} "
          f"(speaks in {chats_per_sender.iloc[0]} of {df.chat_id.nunique()} chats; "
          f"runner-up: {chats_per_sender.index[1]!r} in {chats_per_sender.iloc[1]})")

    df["is_me"] = df.sender == owner
    df["text_len"] = df.text.str.len()
    df["n_words"] = df.text.str.split().str.len().fillna(0).astype(int)

    df = df.sort_values("ts_utc").reset_index(drop=True)
    config.TELEGRAM_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TELEGRAM_PARQUET, index=False)

    print(f"\nwrote {len(df):,} messages "
          f"({df.ts_utc.min():%Y-%m-%d} .. {df.ts_utc.max():%Y-%m-%d}) "
          f"to {config.TELEGRAM_PARQUET}")
    print(f"chats: {df.chat_id.nunique()}, senders: {df.sender.nunique()}, "
          f"my messages: {df.is_me.sum():,} ({df.is_me.mean():.0%})")


if __name__ == "__main__":
    main()
