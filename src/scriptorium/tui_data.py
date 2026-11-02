"""Read job artifacts into display models for the TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Aside, ExtractResult, Flashcard, Plan


@dataclass
class DiffEntry:
    path: str
    action: str
    added: int = 0
    removed: int = 0
    diff: str = ""
    expanded: bool = False
    group: str = ""

    @property
    def title(self) -> str:
        sign = {"create": "+", "merge": "~", "moc": "~", "update": "~"}.get(self.action, "~")
        counts = f"+{self.added}" if self.removed == 0 else f"+{self.added} -{self.removed}"
        return f"{sign} {self.path}  ({counts})"

    @property
    def diff_lines(self) -> list[str]:
        if not self.diff:
            return ["(no diff available for this write)"]
        return self.diff.splitlines()


@dataclass
class CardGroup:
    path: str
    cards: list[Flashcard] = field(default_factory=list)
    requested: int = 0
    written: int = 0
    skipped: int = 0
    writer: str = ""
    expanded: bool = False

    @property
    def title(self) -> str:
        marker = f"{self.written}/{self.requested or len(self.cards)}"
        return f"{self.path}  ({marker} cards)"


def load_diff_entries(path: Path) -> list[DiffEntry]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries: list[DiffEntry] = []
    for group, key in (("created", "created"), ("merged", "merged"), ("moc", "moc")):
        for item in payload.get(key, []):
            entries.append(
                DiffEntry(
                    path=str(item.get("path", "")),
                    action=str(item.get("action", group)),
                    added=int(item.get("added_lines") or 0),
                    removed=int(item.get("removed_lines") or 0),
                    diff=str(item.get("diff") or ""),
                    group=group,
                )
            )
    return entries


def load_asides(plan_path: Path, extract_path: Path | None = None) -> list[Aside]:
    """Off-topic material extracted from the transcript, shown in the Extras pane.

    The extract artifact appears first, so the pane fills as soon as extraction
    finishes.  After cleanup removes it, the merged plan remains the source.
    """
    if extract_path is not None and extract_path.is_file():
        try:
            extract = ExtractResult.model_validate_json(
                extract_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            pass
        else:
            return list(extract.asides)
    if not plan_path.is_file():
        return []
    try:
        plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(plan.asides)


def load_card_groups(plan_path: Path, cards_path: Path) -> list[CardGroup]:
    if not plan_path.is_file():
        return []
    try:
        plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    stats: dict[str, dict[str, Any]] = {}
    if cards_path.is_file():
        try:
            payload = json.loads(cards_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = []
        for item in payload:
            stats[str(item.get("path", ""))] = item

    groups: list[CardGroup] = []
    for note in plan.notes:
        if not note.flashcards:
            continue
        item = stats.get(note.path, {})
        groups.append(
            CardGroup(
                path=note.path,
                cards=list(note.flashcards),
                requested=int(item.get("requested") or len(note.flashcards)),
                written=int(item.get("written") or 0),
                skipped=int(item.get("skipped") or 0),
                writer=str(item.get("writer") or ""),
                expanded=True,
            )
        )
    return groups


def load_skipped_writes(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    lines: list[str] = []
    for item in payload.get("skipped_merges", []):
        lines.append(f"skipped: {item}")
    for item in payload.get("refused", []):
        lines.append(f"refused: {item}")
    return lines
