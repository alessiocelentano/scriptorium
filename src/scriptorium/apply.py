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
from .warnings import RunWarning, WarningCode


@dataclass
class ApplyResult:
    created: list[ChangeRecord] = field(default_factory=list)
    merged: list[ChangeRecord] = field(default_factory=list)
    moc: list[ChangeRecord] = field(default_factory=list)
    applied_paths: list[str] = field(default_factory=list)
    skipped_merges: list[str] = field(default_factory=list)
    conflicts: list[ChangeRecord] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    warnings: list[RunWarning] = field(default_factory=list)


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
        elif merge_policy == "manual" and note.action == "merge":
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

        if decision == "manual" and note.action != "merge":
            decision = "skip"

        if decision == "manual":
            try:
                existing = _existing_body(vault, note)
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
            if _target_has_markers(vault, note):
                result.skipped_merges.append(note.path)
                result.warnings.append(
                    RunWarning(
                        code=WarningCode.UNRESOLVED_CONFLICT_MARKERS,
                        message=(
                            f"{note.path} already contains conflict markers; skipped the "
                            "manual merge to avoid stacking them."
                        ),
                        source=note.path,
                    )
                )
                result.todos.append(
                    TodoItem(
                        kind="restructure",
                        text=(
                            f"Manual merge for {note.path} was skipped because the note "
                            "already contains unresolved conflict markers."
                        ),
                        note_path=note.path,
                    )
                )
                logger.warn(f"{note.path}: already has conflict markers; left the note alone")
                continue
            proposed = _proposed_body(note)
            conflict_copy = _write_conflict(conflict_dir, note, existing, proposed)
            try:
                conflict_text = _manual_merge_text(vault, note, existing, proposed)
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
            try:
                record = vault.apply_change(
                    note.path,
                    conflict_text,
                    action="conflict",
                    expect_only_additions=False,
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
            record.notes = "resolve in the note"
            result.conflicts.append(record)
            result.todos.append(
                TodoItem(
                    kind="restructure",
                    text=(
                        f"Manual merge for {note.path}: resolve the conflict markers in "
                        f"the note" + (f" (see {conflict_copy})" if conflict_copy else "") + "."
                    ),
                    note_path=note.path,
                )
            )
            logger.info(
                f"wrote merge conflict markers into {note.path}"
                + (f"; copy: {conflict_copy}" if conflict_copy else "")
            )
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
            f"merged into {note.path} (+{record.added_lines} lines, -{record.removed_lines} lines)"
        )


def _merged_block(note: PlannedNote) -> str:
    block = note.body
    first = block.splitlines()[0] if block.splitlines() else ""
    if not heading_level_and_text(first):
        block = f"## {note.anchor}\n\n{block}".strip()
    return block


def _merged_text(vault: Vault, note: PlannedNote) -> str:
    if note.action == "merge" and note.anchor:
        # A reviewed merge replaces the whole section, heading included.  If the
        # model forgot the heading, put it back: otherwise the section would lose
        # its title and the following heading would absorb its body.
        return vault.replace_section(note.path, note.anchor, _merged_block(note))
    if note.anchor:
        return vault.insert_under_heading(note.path, note.anchor, note.body, tight=False)
    return vault.append_at_end(note.path, "\n" + note.body)


def _section_body(text: str) -> str:
    """Return a section without its leading heading line."""
    lines = text.splitlines()
    if lines and heading_level_and_text(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip("\n")


def _existing_body(vault: Vault, note: PlannedNote) -> str:
    """The text a manual merge would replace, without the anchor heading."""
    if not note.anchor:
        return ""
    section = vault.section_text_full(note.path, note.anchor)
    return _section_body(section)


def _proposed_body(note: PlannedNote) -> str:
    """The text the manual merge would have written into the target."""
    if note.action == "merge" and note.anchor:
        return _section_body(_merged_block(note))
    return note.body


def _conflict_block(existing: str, proposed: str) -> str:
    return "\n".join(
        [
            "<<<<<<< existing",
            existing.rstrip("\n"),
            "=======",
            proposed.rstrip("\n"),
            ">>>>>>> scriptorium",
        ]
    )


def _has_conflict_markers(text: str) -> bool:
    return any(
        line.startswith("<<<<<<<") or line.startswith(">>>>>>>") for line in text.splitlines()
    )


def _target_has_markers(vault: Vault, note: PlannedNote) -> bool:
    """True when the target note already holds an unresolved conflict block."""
    try:
        return _has_conflict_markers(vault.read(note.path))
    except VaultError:
        return False


def _manual_merge_text(
    vault: Vault,
    note: PlannedNote,
    existing: str,
    proposed: str,
) -> str:
    """The note text with conflict markers standing in for the target text."""
    block = _conflict_block(existing, proposed)
    if note.anchor:
        return vault.replace_section_body(note.path, note.anchor, block)
    return vault.append_at_end(note.path, "\n" + block)


def _conflict_path(directory: Path, note: PlannedNote) -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", note.path).strip("_") or "note.md"
    return directory / f"{name}.merge.conflict.md"


def _write_conflict(
    directory: Path | None,
    note: PlannedNote,
    existing: str,
    proposed: str,
) -> str | None:
    """Write a git-style conflict copy into the job directory for inspection."""
    if directory is None:
        return None
    block = "\n".join(
        [
            f"# Merge conflict: {note.path}",
            "# Scriptorium wrote these markers into the note itself. Edit the note by",
            "# hand, choose what belongs together, then delete the markers below.",
            f"# Anchor: {note.anchor or '(end of note)'}",
            f"# Action: {note.action}",
            "",
            _conflict_block(existing, proposed),
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
        result.todos.append(TodoItem(kind="review", text=f"Could not update {moc_path}: {exc}"))
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
