"""Decide where a planned note belongs.

A note is either created or merged into an existing one. The model proposes a
title, but the vault is the source of truth: this module ranks the existing
notes that could receive the new material and returns the candidates as data.
Choosing one is a separate step, driven by ``Config.target_policy`` and,
for ``ask``, by a caller-supplied chooser.  Nothing here prints, prompts or
writes: the CLI, the TUI and the tests consume the same options.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .config import Config
from .vault import Vault

# Below this score a candidate is not worth showing at all.
TARGET_FLOOR = 0.35
# Weight of the title match against the heading match.
TITLE_WEIGHT = 0.7
HEADING_WEIGHT = 0.3

TargetKind = Literal["merge", "append", "create"]
Chooser = Callable[[list["NoteTarget"]], "NoteTarget | None"]


@dataclass(frozen=True)
class NoteTarget:
    """One place a note could go: an existing section, an existing note, or a new note."""

    kind: TargetKind
    path: str
    title: str
    anchor: str | None = None
    score: float = 0.0
    evidence: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        if self.kind == "create":
            return f"create a new note '{self.title}' ({self.path})"
        if self.anchor:
            return f"{self.kind} into {self.path} under '{self.anchor}'"
        return f"{self.kind} into {self.path}"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "title": self.title,
            "anchor": self.anchor,
            "score": self.score,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class TargetChoice:
    target: NoteTarget
    origin: Literal["auto", "policy", "fallback"]


def resolve_targets(
    vault: Vault,
    title: str,
    *,
    limit: int = 3,
) -> list[NoteTarget]:
    """Rank the vault notes that could receive ``title``; always ends with create."""
    candidates: list[NoteTarget] = []
    for note in vault.inventory():
        title_score = _similarity(title, note.title)
        anchor: str | None = None
        heading_score = 0.0
        for heading in note.headings:
            score = _similarity(title, heading)
            if score > heading_score:
                anchor, heading_score = heading, score
        if anchor is None:
            score = title_score
            kind: TargetKind = "append"
        else:
            score = TITLE_WEIGHT * title_score + HEADING_WEIGHT * heading_score
            kind = "merge"
        if score < TARGET_FLOOR:
            continue
        evidence = [f"title match {title_score:.2f}"]
        if anchor:
            evidence.append(f"heading '{anchor}' matches {heading_score:.2f}")
        if note.tags:
            evidence.append("tags " + ", ".join(note.tags[:3]))
        candidates.append(
            NoteTarget(
                kind=kind,
                path=note.path,
                title=note.title,
                anchor=anchor,
                score=round(min(score, 1.0), 4),
                evidence=tuple(evidence),
            )
        )

    candidates.sort(key=lambda target: target.score, reverse=True)
    selected = candidates[:limit]
    selected.append(
        NoteTarget(
            kind="create",
            path=f"{title}.md",
            title=title,
            evidence=("new note",),
        )
    )
    return selected


def decide(
    candidates: Sequence[NoteTarget],
    config: Config,
    *,
    chooser: Chooser | None = None,
) -> TargetChoice:
    """Pick a target according to ``Config.target_policy``."""
    create = next((item for item in candidates if item.kind == "create"), None)
    existing = [item for item in candidates if item.kind != "create"]
    if create is None:
        create = NoteTarget(kind="create", path="Note.md", title="Note")

    if config.target_policy == "create" or not existing:
        return TargetChoice(create, "policy")

    best = existing[0]
    if config.target_policy == "auto":
        second = existing[1] if len(existing) > 1 else None
        gap = best.score - second.score if second else 1.0
        if best.score >= config.target_auto_above and gap >= config.target_min_gap:
            return TargetChoice(best, "auto")
        return TargetChoice(create, "fallback")

    if chooser is not None:
        chosen = chooser(list(candidates))
        if chosen is not None and any(item is chosen for item in candidates):
            return TargetChoice(chosen, "policy")
    return TargetChoice(create, "fallback")


def merge_candidates(candidates: Sequence[NoteTarget]) -> list[NoteTarget]:
    return [item for item in candidates if item.kind != "create"]


def _similarity(left: str, right: str) -> float:
    left_norm, right_norm = _normalize(left), _normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    ratio = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    score = 0.62 * jaccard + 0.38 * ratio
    if left_norm in right_norm or right_norm in left_norm:
        score += 0.2
    if left_tokens and left_tokens <= right_tokens:
        score += 0.1
    return min(score, 1.0)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _tokens(value: str) -> set[str]:
    return set(_normalize(value).split())
