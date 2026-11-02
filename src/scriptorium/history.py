"""Completed-run history: the generated text mapped to the box that showed it.

A run is saved when the pipeline completes.  The snapshot holds only what the
TUI displays -- the collected sources, the vault changes and their diffs, the
draft flashcards and the Extras -- so ``/history`` can restore a previous run
without replaying the pipeline or depending on the job directory.

The file is deterministic JSON under ``$SCRIPTORIUM_ARTIFACTS_DIR/history``; one file per
job id, rewritten when the same job is resumed and completed again.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Aside, Flashcard
from .tui_data import CardGroup, DiffEntry, load_asides, load_card_groups, load_diff_entries


@dataclass
class HistoryEntry:
    """One completed run, already shaped for the TUI panes."""

    id: str
    job_id: str
    course: str
    created_at: str
    finished_at: str
    sources: list[str] = field(default_factory=list)
    changes: list[DiffEntry] = field(default_factory=list)
    card_groups: list[CardGroup] = field(default_factory=list)
    extras: list[Aside] = field(default_factory=list)

    @property
    def card_count(self) -> int:
        return sum(len(group.cards) for group in self.card_groups)

    @property
    def summary(self) -> str:
        """One line for the history window."""
        pieces = [
            f"{len(self.changes)} change(s)",
            f"{self.card_count} card(s)",
            f"{len(self.extras)} extra(s)",
            f"{len(self.sources)} source(s)",
        ]
        return " · ".join(pieces)

    @property
    def label(self) -> str:
        stamp = self.finished_at or self.created_at
        course = self.course or self.job_id
        return f"{stamp} · {course}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "course": self.course,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "sources": list(self.sources),
            "changes": [_diff_to_dict(entry) for entry in self.changes],
            "flashcards": [_group_to_dict(group) for group in self.card_groups],
            "extras": [aside.model_dump() for aside in self.extras],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistoryEntry:
        job_id = str(payload.get("job_id") or payload.get("id") or "")
        return cls(
            id=str(payload.get("id") or job_id),
            job_id=job_id,
            course=str(payload.get("course") or ""),
            created_at=str(payload.get("created_at") or ""),
            finished_at=str(payload.get("finished_at") or payload.get("created_at") or ""),
            sources=[str(path) for path in payload.get("sources") or []],
            changes=[_diff_from_dict(item) for item in payload.get("changes") or []],
            card_groups=[_group_from_dict(item) for item in payload.get("flashcards") or []],
            extras=[Aside.model_validate(item) for item in payload.get("extras") or []],
        )


class HistoryStore:
    """Reads and writes completed-run snapshots in one directory."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def save(self, entry: HistoryEntry) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{_safe_id(entry.id or entry.job_id)}.json"
        _atomic_write(path, json.dumps(entry.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return path

    def entries(self, limit: int = 200) -> list[HistoryEntry]:
        """Every readable snapshot, newest finished run first."""
        if not self.directory.is_dir():
            return []
        found: list[HistoryEntry] = []
        for path in sorted(self.directory.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            try:
                found.append(HistoryEntry.from_dict(payload))
            except (TypeError, ValueError):
                continue
        found.sort(key=lambda entry: entry.finished_at or entry.created_at, reverse=True)
        return found[:limit]


def record_completed_run(
    history_dir: Path | str,
    job_dir: Path | str,
    *,
    job_id: str,
    course: str,
    sources: Sequence[str],
    created_at: str,
    finished_at: str,
) -> HistoryEntry:
    """Read a completed job's artifacts into a snapshot and save it."""
    directory = Path(job_dir)
    plan_path = directory / "05.plan.json"
    entry = HistoryEntry(
        id=job_id,
        job_id=job_id,
        course=course,
        created_at=created_at,
        finished_at=finished_at,
        sources=[str(source) for source in sources],
        changes=load_diff_entries(directory / "apply.json"),
        card_groups=load_card_groups(plan_path, directory / "cards.json"),
        extras=load_asides(plan_path, directory / "03.extract.json"),
    )
    HistoryStore(history_dir).save(entry)
    return entry


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def _diff_to_dict(entry: DiffEntry) -> dict[str, Any]:
    return {
        "path": entry.path,
        "action": entry.action,
        "added": entry.added,
        "removed": entry.removed,
        "diff": entry.diff,
        "expanded": entry.expanded,
        "group": entry.group,
    }


def _diff_from_dict(payload: dict[str, Any]) -> DiffEntry:
    return DiffEntry(
        path=str(payload.get("path") or ""),
        action=str(payload.get("action") or "update"),
        added=_as_int(payload.get("added")),
        removed=_as_int(payload.get("removed")),
        diff=str(payload.get("diff") or ""),
        expanded=bool(payload.get("expanded") or False),
        group=str(payload.get("group") or ""),
    )


def _group_to_dict(group: CardGroup) -> dict[str, Any]:
    return {
        "path": group.path,
        "cards": [card.model_dump() for card in group.cards],
        "requested": group.requested,
        "written": group.written,
        "skipped": group.skipped,
        "writer": group.writer,
        "expanded": group.expanded,
    }


def _group_from_dict(payload: dict[str, Any]) -> CardGroup:
    return CardGroup(
        path=str(payload.get("path") or ""),
        cards=[Flashcard.model_validate(card) for card in payload.get("cards") or []],
        requested=_as_int(payload.get("requested")),
        written=_as_int(payload.get("written")),
        skipped=_as_int(payload.get("skipped")),
        writer=str(payload.get("writer") or ""),
        expanded=bool(payload.get("expanded") or False),
    )


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
