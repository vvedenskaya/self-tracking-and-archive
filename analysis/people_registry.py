"""people.yaml — the registry of who's who across sources.

Seeds the registry from the biggest Telegram chats and, on later runs,
appends any big chats that aren't registered yet. It NEVER rewrites the
file: new entries are appended as text, so your hand edits (merged
identities, emails, tags, comments) are always preserved.

Each entry:
    - id: stable slug, never change it once other files reference it
      display: how the person appears in charts/notes
      telegram: list of Telegram chat names that are this person
      email: list of email addresses (for the Gmail pass)
      tags: free-form (friend, work, family, ...)
      notes: one line of context

Usage: python analysis/people_registry.py [N]   (default: top 60 chats)
"""
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

HEADER = """\
# People registry — maps identities across data sources.
# Hand-edit freely: merge duplicate people by moving their telegram/email
# aliases into one entry and deleting the other. The seeder only appends
# entries for unregistered chats; it never modifies existing ones.
"""


def slugify(name: str) -> str:
    s = re.sub(r"[^\wа-яё]+", "_", name.lower(), flags=re.U).strip("_")
    return s or "unnamed"


def entry_text(name: str, used_ids: set) -> str:
    base = slugify(name)
    slug, i = base, 2
    while slug in used_ids:
        slug, i = f"{base}_{i}", i + 1
    used_ids.add(slug)
    q = name.replace('"', '\\"')
    return (f'- id: {slug}\n'
            f'  display: "{q}"\n'
            f'  telegram: ["{q}"]\n'
            f'  email: []\n'
            f'  tags: []\n'
            f'  notes: ""\n')


def load_registry() -> list:
    if not config.PEOPLE_YAML.exists():
        return []
    return yaml.safe_load(config.PEOPLE_YAML.read_text(encoding="utf-8")) or []


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    df = pd.read_parquet(config.TELEGRAM_PARQUET)
    top = df.groupby("chat_name").size().nlargest(top_n)

    existing = load_registry()
    known_tg = {alias for p in existing for alias in (p.get("telegram") or [])}
    used_ids = {p["id"] for p in existing}

    new = [name for name in top.index if name not in known_tg]
    if not new:
        print(f"people.yaml already covers the top {top_n} chats "
              f"({len(existing)} entries)")
        return

    blocks = "\n".join(entry_text(n, used_ids) for n in new)
    if config.PEOPLE_YAML.exists():
        with config.PEOPLE_YAML.open("a", encoding="utf-8") as f:
            f.write("\n" + blocks)
        print(f"appended {len(new)} new entries "
              f"({len(existing)} were already registered)")
    else:
        config.PEOPLE_YAML.write_text(HEADER + "\n" + blocks, encoding="utf-8")
        print(f"created people.yaml with {len(new)} entries "
              f"from the top {top_n} chats")
    print(f"-> {config.PEOPLE_YAML} (gitignored — it's personal data)")


if __name__ == "__main__":
    main()
