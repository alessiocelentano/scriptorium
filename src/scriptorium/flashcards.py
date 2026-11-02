"""Flashcard validation, rendering, and draft insertion.

Cards are always written as drafts: every physical line is wrapped in
``<!--DRAFT: ... -->``. The Obsidian Spaced Repetition plugin ignores those lines
until the user activates them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import FlashcardError
from .log import Logger
from .models import Flashcard
from .vault import Vault, added_lines, only_additions

DRAFT_PREFIX = "<!--DRAFT:"
DRAFT_SUFFIX = "-->"
_HAS_TAG_RE = re.compile(r"(?:^|\s)#(?:cs|language)(?:/[^\s#]+)?\b")
_INLINE_SEPARATOR_RE = re.compile(r"\S[ \t]+\?\?\?[ \t]+\S")
_EM_DASH_RE = re.compile(r"[\u2013\u2014]")
_CLOZE_RE = re.compile(r"\{\{c\d+::.+?\}\}")


@dataclass
class CardWriteResult:
    path: str
    requested: int = 0
    written: int = 0
    skipped: int = 0
    writer: str = "skill"

    @property
    def summary(self) -> str:
        return f"{self.path}: {self.written} draft card(s) via {self.writer}" + (
            f", {self.skipped} dropped" if self.skipped else ""
        )


def validate_flashcard(card: Flashcard) -> list[str]:
    issues: list[str] = []
    front = card.front.strip()
    back = card.back.strip()

    if not front:
        return ["empty front"]
    if len(front) > 600:
        issues.append("front is longer than 600 characters")
    if len(back) > 3000:
        issues.append("back is longer than 3000 characters")
    if DRAFT_SUFFIX in front or DRAFT_SUFFIX in back:
        issues.append("card text contains '-->' which breaks draft encoding")
    if _EM_DASH_RE.search(front) or _EM_DASH_RE.search(back):
        issues.append("card text contains an em dash or en dash")

    if card.kind == "cloze":
        if not _CLOZE_RE.search(front):
            issues.append("cloze card has no {{c1::...}} deletion")
        return issues

    if not front.endswith("?"):
        issues.append("front is not a question ending with '?'")
    if not back:
        issues.append("empty back")
    if ";;;" in front or ";;;" in back:
        issues.append("card text contains the card separator ';;;'")
    if "???" in front or "???" in back:
        issues.append("card text contains the card separator '???'")

    if card.kind in {"single", "reversed"}:
        if "\n" in front or "\n" in back:
            issues.append("single-line cards cannot contain newlines")
        if _INLINE_SEPARATOR_RE.search(front) or _INLINE_SEPARATOR_RE.search(back):
            issues.append("inline '???' separator is not parseable by the plugin")

    if card.kind == "multi":
        for line in (front + "\n" + back).splitlines():
            if line.strip() == "???":
                issues.append("multi-line card content contains a standalone '???' line")
                break
        if "\n\n" in front or "\n\n" in back:
            issues.append("multi-line card content contains a blank line, which ends the card")

    return issues


def render_cards(cards: list[Flashcard]) -> str:
    blocks: list[str] = []
    for card in cards:
        front = card.front.strip()
        back = card.back.strip()
        if card.kind == "reversed":
            blocks.append(f"{front} ;;;; {back}")
        elif card.kind == "multi":
            blocks.append(f"{front}\n???\n{back}")
        elif card.kind == "cloze":
            blocks.append(front)
        else:
            blocks.append(f"{front} ;;; {back}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def note_has_flashcard_tag(text: str, extra_tags: list[str] | None = None) -> bool:
    from .vault import parse_tags

    for tag in list(parse_tags(text)) + list(extra_tags or []):
        if tag == "cs" or tag.startswith("cs/") or tag == "language" or tag.startswith("language/"):
            return True
    return bool(_HAS_TAG_RE.search(text))


def write_drafts(
    vault: Vault,
    note_path: str,
    cards: list[Flashcard],
    config: Config,
    logger: Logger,
    *,
    extra_tags: list[str] | None = None,
) -> CardWriteResult:
    result = CardWriteResult(path=note_path, requested=len(cards))
    if not cards:
        return result

    text = vault.read(note_path)
    if not note_has_flashcard_tag(text, extra_tags):
        result.skipped = len(cards)
        logger.warn(
            f"flashcards: {note_path} has no tag under #cs or #language; "
            "cards would be invisible, so they were skipped"
        )
        return result

    if config.flashcard_max_per_note > 0:
        cards = cards[: config.flashcard_max_per_note]

    prepared = vault.ensure_flashcards_section(note_path)
    if prepared != text:
        vault.apply_change(note_path, prepared, action="add flashcards section")

    rendered = render_cards(cards)
    changed, result.writer = _write_with_skill(vault, note_path, rendered, config, logger)

    result.written = len(cards) if changed else 0
    result.skipped = len(cards) - result.written
    return result


def _write_with_skill(
    vault: Vault,
    note_path: str,
    rendered: str,
    config: Config,
    logger: Logger,
) -> tuple[int, str]:
    skill = Path(config.flashcard_skill)
    path = vault.resolve(note_path)
    original = vault.read(note_path)
    environment = dict(os.environ)
    environment["SCRIPTORIUM_NOTES_ROOT"] = str(vault.root)
    command = [sys.executable, str(skill), "add-drafts", "--file", str(path)]
    completed = subprocess.run(
        command,
        input=rendered,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    updated = vault.read(note_path)

    if completed.returncode != 0:
        if updated != original:
            vault.apply_change(
                note_path,
                original,
                expect_only_additions=False,
                action="restore after failed skill",
            )
        if _skill_lacks_add_drafts(completed.stdout, completed.stderr):
            logger.warn(
                f"flashcard skill does not support add-drafts; "
                f"inserting drafts directly into {note_path}"
            )
        else:
            reason = (completed.stderr or completed.stdout).strip()[-600:]
            logger.warn(
                f"flashcard skill failed for {note_path}; inserting drafts directly: {reason}"
            )
        return _write_internal(vault, note_path, rendered, logger), "internal"

    if not only_additions(original, updated):
        vault.apply_change(
            note_path,
            original,
            expect_only_additions=False,
            action="restore after skill",
        )
        raise FlashcardError(
            f"flashcard skill modified existing lines in {note_path}; change reverted"
        )

    inserted = added_lines(original, updated)
    non_draft = [
        line for line in inserted if line.strip() and not line.strip().startswith(DRAFT_PREFIX)
    ]
    if non_draft:
        vault.apply_change(
            note_path,
            original,
            expect_only_additions=False,
            action="restore non-draft skill output",
        )
        logger.warn(
            f"flashcard skill wrote non-draft line(s) to {note_path}; "
            "reverted and inserting drafts directly"
        )
        return _write_internal(vault, note_path, rendered, logger), "internal"

    written = sum(1 for line in inserted if line.strip().startswith(DRAFT_PREFIX))
    logger.debug(f"flashcards: skill inserted {written} draft line(s) into {note_path}")
    return written, "skill"


def _skill_lacks_add_drafts(stdout: str, stderr: str) -> bool:
    """True when the skill rejected ``add-drafts`` as an unknown command."""
    output = f"{stdout}\n{stderr}".casefold()
    if "add-drafts" not in output:
        return False
    markers = (
        "invalid choice",
        "unrecognized arguments",
        "unknown command",
        "no such command",
        "unrecognized subcommand",
    )
    return any(marker in output for marker in markers)


def _write_internal(
    vault: Vault,
    note_path: str,
    rendered: str,
    logger: Logger,
) -> int:
    """Insert drafts without a skill, for skills that do not support add-drafts."""
    lines = rendered.rstrip("\n").split("\n")
    block = "\n".join(
        f"{DRAFT_PREFIX} {line} {DRAFT_SUFFIX}" if line.strip() else "" for line in lines
    )
    original = vault.read(note_path)
    updated = vault.insert_under_heading(note_path, "Flashcards", block)
    if updated == original:
        return 0
    vault.apply_change(note_path, updated, action="add draft cards")
    written = sum(1 for line in block.splitlines() if line.strip())
    logger.debug(f"flashcards: inserted {written} draft line(s) into {note_path}")
    return written
