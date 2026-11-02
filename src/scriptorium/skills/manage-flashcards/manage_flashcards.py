#!/usr/bin/env python3
"""Bundled flashcard skill: insert draft cards under a note's Flashcards heading.

This is the default target of ``flashcard_skill`` in ``config.toml``.  It is a
standalone script (no imports from the package) so a user can copy it, edit it,
or replace it with their own skill at another path.

Usage:
    python manage_flashcards.py add-drafts --file /path/to/note.md < cards.txt

The card text on stdin is inserted verbatim, one draft line per non-empty line,
under the note's ``## Flashcards`` heading.  Existing lines are never changed or
removed; the caller verifies that afterwards.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DRAFT_PREFIX = "<!--DRAFT:"
DRAFT_SUFFIX = "-->"
HEADING_RE = re.compile(r"^(#{1,6})\s+Flashcards\s*$", re.IGNORECASE)


def encode_line(line: str) -> str:
    return f"{DRAFT_PREFIX} {line} {DRAFT_SUFFIX}"


def encode_block(rendered: str) -> list[str]:
    lines = rendered.rstrip("\n").split("\n")
    return [encode_line(line) if line.strip() else "" for line in lines]


def insert_drafts(text: str, block: list[str]) -> str:
    lines = text.splitlines()
    heading_index = None
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line.strip())
        if match:
            heading_index = index
            break

    if heading_index is None:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            lines.append("")
        lines.append("## Flashcards")
        lines.append("")
        lines.extend(block)
    else:
        insert_at = heading_index + 1
        if insert_at < len(lines) and lines[insert_at].strip():
            lines.insert(insert_at, "")
            insert_at += 1
        insert_at += 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        for offset, line in enumerate(block):
            lines.insert(insert_at + offset, line)
    return "\n".join(lines) + "\n"


def add_drafts(path: Path) -> int:
    rendered = sys.stdin.read()
    if not rendered.strip():
        return 0
    block = encode_block(rendered)
    original = path.read_text(encoding="utf-8")
    updated = insert_drafts(original, block)
    if updated == original:
        return 0
    path.write_text(updated, encoding="utf-8")
    return sum(1 for line in block if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manage_flashcards")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add-drafts", help="Insert draft cards from stdin")
    add.add_argument("--file", required=True, type=Path, help="Note to update")
    args = parser.parse_args(argv)

    if not args.file.is_file():
        print(f"note not found: {args.file}", file=sys.stderr)
        return 1
    written = add_drafts(args.file)
    print(f"inserted {written} draft card line(s) into {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
