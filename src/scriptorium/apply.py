"""Deterministic application of a validated plan to the vault."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import VaultError
from .log import Logger
from .models import Plan, PlannedNote, TodoItem, heading_level_and_text
from .planning import moc_headings
from .ui import ConsoleUI
from .vault import ChangeRecord, Vault


@dataclass
class ApplyResult:
    created: list[ChangeRecord] = field(default_factory=list)
    merged: list[ChangeRecord] = field(default_factory=list)
    moc: list[ChangeRecord] = field(default_factory=list)
    applied_paths: list[str] = field(default_factory=list)
    skipped_merges: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)


def apply_plan(
    plan: Plan,
    vault: Vault,
    ui: ConsoleUI,
    logger: Logger,
    *,
    accept_merges: bool,
    update_moc: bool = True,
    conflict_dir: Path | None = None,
    merge_policy: str = "ask",
) -> ApplyResult:
    result = ApplyResult()
    pending: list[PlannedNote] = []

    for note in plan.notes:
        if note.action == "create" and not vault.exists(note.path):
            if note.path.startswith("University/"):
                result.refused.append(note.path)
                result.todos.append(
                    TodoItem(
                        kind="restructure",
                        text=f"Refused to create a note inside University/: {note.path}",
                        note_path=note.path,
                    )
                )
                continue
            try:
                record = vault.write_new(note.path, create_note_text(note))
            except VaultError as exc:
                result.refused.append(note.path)
                result.todos.append(
                    TodoItem(
                        kind="review",
                        text=f"Could not create {note.path}: {exc}",
                        note_path=note.path,
                    )
                )
                continue
            result.created.append(record)
            result.applied_paths.append(note.path)
            logger.success(f"created {note.path}")
        else:
            # Either the plan asked for a merge, or the note appeared after planning.
            pending.append(note)

    vault.refresh()
    _apply_merges(
        pending,
        vault,
        ui,
        logger,
        accept_merges,
        result,
        conflict_dir=conflict_dir,
        merge_policy=merge_policy,
    )

    if update_moc and plan.moc_path:
        _update_moc(plan, vault, logger, result, plan.moc_path)

    return result


def _apply_merges(
    pending: list[PlannedNote],
    vault: Vault,
    ui: ConsoleUI,
    logger: Logger,
    accept_merges: bool,
    result: ApplyResult,
    *,
    conflict_dir: Path | None,
    merge_policy: str,
) -> None:
    merge_all = accept_merges
    stop = False

    for note in pending:
        try:
            new_text = _merged_text(vault, note)
        except VaultError as exc:
            result.refused.append(note.path)
            result.todos.append(
                TodoItem(
                    kind="review",
                    text=f"Could not merge into {note.path}: {exc}",
                    note_path=note.path,
                )
            )
            logger.error(f"merge refused for {note.path}: {exc}")
            continue

        diff = vault.preview_change(note.path, new_text)
        if not diff:
            logger.info(f"{note.path}: nothing to merge, the note is already up to date")
            continue

        if stop:
            decision = "skip"
        elif merge_all:
            decision = "apply"
        elif merge_policy == "apply":
            decision = "apply"
        elif merge_policy == "manual":
            decision = "manual"
        else:
            decision = ui.review_merge(note, diff)
            if decision == "all":
                merge_all = True
                decision = "apply"
            elif decision == "quit":
                stop = True
                decision = "skip"
            elif decision not in {"apply", "skip", "manual"}:
                decision = "skip"

        if decision == "manual":
            conflict = _write_conflict(conflict_dir, vault, note)
            if conflict is None:
                result.skipped_merges.append(note.path)
                result.todos.append(
                    TodoItem(
                        kind="review",
                        text=f"Merge into {note.path} needs a manual review; no job "
                        "directory is available for the conflict file.",
                        note_path=note.path,
                    )
                )
                continue
            result.conflicts.append(conflict)
            result.todos.append(
                TodoItem(
                    kind="restructure",
                    text=(
                        f"Manual merge for {note.path}: edit the note yourself using the "
                        f"conflict file {conflict}; Scriptorium did not write the note."
                    ),
                    note_path=note.path,
                )
            )
            logger.info(f"wrote merge conflict file {conflict}")
            continue

        if decision != "apply":
            result.skipped_merges.append(note.path)
            result.todos.append(
                TodoItem(
                    kind="restructure",
                    text=(
                        f"Merge into {note.path}"
                        + (f" under '{note.anchor}'" if note.anchor else "")
                        + " was not applied; run the note import again to review it."
                    ),
                    note_path=note.path,
                )
            )
            continue

        try:
            record = vault.apply_change(
                note.path,
                new_text,
                action=note.action,
                expect_only_additions=note.action != "merge",
            )
        except VaultError as exc:
            result.refused.append(note.path)
            result.todos.append(
                TodoItem(
                    kind="review",
                    text=f"Could not merge into {note.path}: {exc}",
                    note_path=note.path,
                )
            )
            logger.error(f"merge refused for {note.path}: {exc}")
            continue

        result.merged.append(record)
        result.applied_paths.append(note.path)
        logger.success(
            f"merged into {note.path} (+{record.added_lines} lines, "
            f"-{record.removed_lines} lines)"
        )


def _merged_text(vault: Vault, note: PlannedNote) -> str:
    if note.action == "merge" and note.anchor:
        # A reviewed merge replaces the whole section, heading included.  If the
        # model forgot the heading, put it back: otherwise the section would lose
        # its title and the following heading would absorb its body.
        block = note.body
        first = block.splitlines()[0] if block.splitlines() else ""
        if not heading_level_and_text(first):
            block = f"## {note.anchor}\n\n{block}".strip()
        return vault.replace_section(note.path, note.anchor, block)
    if note.anchor:
        return vault.insert_under_heading(note.path, note.anchor, note.body, tight=False)
    return vault.append_at_end(note.path, "\n" + note.body)


def _conflict_path(directory: Path, note: PlannedNote) -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", note.path).strip("_") or "note.md"
    return directory / f"{name}.merge.conflict.md"


def _write_conflict(
    directory: Path | None,
    vault: Vault,
    note: PlannedNote,
) -> str | None:
    """Write a git-style conflict file outside the vault; return its path."""
    if directory is None:
        return None
    if note.anchor:
        try:
            existing = vault.section_text(note.path, note.anchor, max_chars=200_000)
        except VaultError:
            existing = ""
    else:
        try:
            existing = vault.read(note.path)
        except VaultError:
            existing = ""
    proposed = note.body
    block = "\n".join(
        [
            f"# Merge conflict: {note.path}",
            "# Scriptorium did not write the note. Edit the note by hand, choose what",
            "# belongs together, then delete the markers below.",
            f"# Anchor: {note.anchor or '(end of note)'}",
            f"# Action: {note.action}",
            "",
            f"<<<<<<< existing ({note.path})",
            existing.rstrip("\n"),
            "=======",
            proposed.rstrip("\n"),
            f">>>>>>> scriptorium ({note.action})",
            "",
        ]
    )
    path = _conflict_path(directory, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block, encoding="utf-8")
    return str(path)


def create_note_text(note: PlannedNote) -> str:
    body = note.body.strip()
    if note.tags:
        lines = ["---", "tags:"]
        for tag in note.tags:
            lines.append(f"  - {tag}")
        lines.extend(["---", "", body, ""])
        return "\n".join(lines)
    return body + "\n"


def _update_moc(
    plan: Plan,
    vault: Vault,
    logger: Logger,
    result: ApplyResult,
    moc_path: str,
) -> None:
    if not vault.exists(moc_path):
        result.todos.append(
            TodoItem(
                kind="missing",
                text=f"MOC not found, notes are not indexed: {moc_path}",
            )
        )
        return

    moc_text = vault.read(moc_path)
    blocks: list[str] = []
    for note_path in result.applied_paths:
        title = Path(note_path).stem
        if _moc_has_link(moc_text, title) or any(
            re.search(rf"\[\[{re.escape(title)}(\||#|\]\])", block, re.IGNORECASE)
            for block in blocks
        ):
            continue
        try:
            note_text = vault.read(note_path)
        except VaultError:
            continue
        blocks.append(moc_entries(title, note_text))

    if not blocks:
        return

    block = "\n".join(blocks)
    for candidate in ("Table of Contents", "Contents"):
        try:
            new_text = vault.insert_under_heading(moc_path, candidate, block, tight=True)
            break
        except VaultError:
            continue
    else:
        result.todos.append(
            TodoItem(
                kind="restructure",
                text=f"No Table of Contents heading in {moc_path}; notes are not indexed.",
            )
        )
        return

    try:
        record = vault.apply_change(moc_path, new_text, action="moc")
    except VaultError as exc:
        result.todos.append(
            TodoItem(kind="review", text=f"Could not update {moc_path}: {exc}")
        )
        return
    result.moc.append(record)
    logger.success(f"updated MOC {moc_path} (+{record.added_lines} lines)")


def moc_entries(title: str, note_text: str) -> str:
    lines = [f"- [[{title}]]"]
    for level, heading in moc_headings(note_text):
        if level == 2:
            lines.append(f"\t- {heading}")
        elif level == 3:
            lines.append(f"\t\t- {heading}")
    return "\n".join(lines)


def _moc_has_link(moc_text: str, title: str) -> bool:
    return bool(re.search(rf"\[\[{re.escape(title)}(?:\||#|\]\])", moc_text, re.IGNORECASE))
