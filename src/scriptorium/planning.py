"""LLM phases: extract knowledge units, plan placement, compose notes."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import Config
from .context import ContextBundle
from .errors import LLMError, PlanValidationError
from .llm.client import DeepSeekClient
from .log import Logger
from .models import (
    Aside,
    ComposedNote,
    ExtractResult,
    Flashcard,
    Grouping,
    NoteOutline,
    Outline,
    Plan,
    PlannedNote,
    SourceGroup,
    TodoItem,
    Transcript,
    Unit,
    heading_level_and_text,
)
from .pdf_images import ExtractedImage, images_for_pages, page_numbers
from .prompting import load_prompt, render
from .sources import TextSource
from .targeting import Chooser, NoteTarget, decide, merge_candidates, resolve_targets
from .vault import Vault

FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)
FLASHCARD_BLOCK_RE = re.compile(r"^#{1,6}\s+Flashcards\s*$", re.IGNORECASE | re.MULTILINE)
EXCLUDED_MOC_HEADINGS = {
    "flashcards",
    "table of contents",
    "useful links",
    "references",
    "notes",
    "further reading",
    "see also",
}


# --------------------------------------------------------------------------- #
# Source chunking
# --------------------------------------------------------------------------- #


def split_text(
    text: str,
    *,
    max_chars: int = 22000,
    overlap_chars: int = 1800,
) -> list[str]:
    """Split text at line boundaries into overlapping chunks."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) + 1 > overlap_chars:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            current = tail + [line]
            size = tail_size + len(line) + 1
        else:
            current.append(line)
            size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def split_transcript(
    transcript: Transcript,
    *,
    max_chars: int = 22000,
    overlap_chars: int = 1800,
) -> list[str]:
    return split_text(transcript.timed_text(), max_chars=max_chars, overlap_chars=overlap_chars)


# --------------------------------------------------------------------------- #
# Map phase
# --------------------------------------------------------------------------- #


def extract_units(
    transcript: Transcript,
    client: DeepSeekClient,
    logger: Logger,
    *,
    course: str,
    max_units_per_chunk: int = 25,
) -> ExtractResult:
    """Extract units from one transcript (compatibility wrapper)."""
    return extract_sources(
        [
            TextSource(
                id="s01",
                name="lecture transcript",
                kind="media",
                path=Path("transcript.txt"),
                text=transcript.timed_text(),
            )
        ],
        client,
        logger,
        course=course,
        max_units_per_chunk=max_units_per_chunk,
    )


def extract_sources(
    sources: Sequence[TextSource],
    client: DeepSeekClient,
    logger: Logger,
    *,
    course: str,
    max_units_per_chunk: int = 25,
) -> ExtractResult:
    """Extract knowledge units from every readable source, in input order."""
    usable = [source for source in sources if source.text.strip()]
    if not usable:
        raise PlanValidationError("no source produced usable text")

    template = load_prompt("extract.md")
    system = (
        "You are a meticulous lecture-note extractor. "
        "You only return valid JSON matching the requested shape."
    )

    units: list[Unit] = []
    asides: list[Aside] = []
    topics: list[str] = []
    for source in usable:
        chunks = split_text(source.text)
        used_images: set[str] = set()
        source_units: list[Unit] = []
        for index, chunk in enumerate(chunks, start=1):
            offered = [
                image
                for image in images_for_pages(source.images, page_numbers(chunk))
                if image.id not in used_images
            ]
            logger.step(f"extract: {source.name} chunk {index}/{len(chunks)}")
            prompt = render(
                template,
                {
                    "course": course,
                    "source_name": source.name,
                    "source_kind": source.kind,
                    "topics": ", ".join(topics) or "none yet",
                    "transcript": chunk,
                    "images": _images_block(offered) or "(none)",
                    "max_units": max_units_per_chunk,
                },
            )
            payload = client.chat_json(system, prompt, phase="extract")
            result = _validate_extract(payload, logger, chunk_index=index, source_name=source.name)
            allowed_images = {image.name for image in offered}
            for unit in result.units:
                unit.images = [name for name in unit.images if name in allowed_images]
                unit.source = f"{source.name} {unit.source}".strip()
                unit.sources = [source.name]
            used_images.update(image.id for image in offered)
            for aside in result.asides:
                aside.source = f"{source.name} {aside.source}".strip()
            source_units.extend(result.units)
            asides.extend(result.asides)
            for topic in result.topics:
                if topic and topic not in topics:
                    topics.append(topic)
        _claim_orphan_images(source_units, source.images, used_images)
        units.extend(source_units)

    if not units:
        raise PlanValidationError("the extraction phase produced no knowledge units")

    deduped = _dedupe_units(units)
    for number, unit in enumerate(deduped, start=1):
        unit.id = f"u{number}"
    deduped_asides = _dedupe_asides(asides)
    logger.success(
        f"extract: {len(deduped)} knowledge units and {len(deduped_asides)} asides "
        f"from {len(usable)} sources"
    )
    return ExtractResult(units=deduped, asides=deduped_asides, topics=topics)


def _validate_extract(
    payload: dict[str, Any],
    logger: Logger,
    *,
    chunk_index: int,
    source_name: str = "",
) -> ExtractResult:
    raw_units = payload.get("units")
    if raw_units is None:
        raise LLMError("extract: response has no 'units' key")
    if not isinstance(raw_units, list):
        raise LLMError("extract: 'units' must be a list")

    units: list[Unit] = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            continue
        try:
            unit = Unit.model_validate(raw)
        except ValidationError as exc:
            logger.warn(
                f"extract: skipped invalid unit in {source_name} chunk {chunk_index}: "
                f"{exc.errors()[0]}"
            )
            continue
        if unit.content.strip():
            units.append(unit)

    asides: list[Aside] = []
    raw_asides = payload.get("asides") or []
    if isinstance(raw_asides, list):
        for raw in raw_asides:
            if not isinstance(raw, dict):
                continue
            try:
                aside = Aside.model_validate(raw)
            except ValidationError as exc:
                logger.warn(
                    f"extract: skipped invalid aside in {source_name} chunk {chunk_index}: "
                    f"{exc.errors()[0]}"
                )
                continue
            if aside.text:
                asides.append(aside)

    raw_topics = payload.get("topics") or []
    topics = [str(topic).strip() for topic in raw_topics if str(topic).strip()]
    return ExtractResult(units=units, asides=asides, topics=topics)


def _dedupe_units(units: list[Unit]) -> list[Unit]:
    result: list[Unit] = []
    index: dict[tuple[str, str], int] = {}
    for unit in units:
        key = (unit.kind, _normalize(unit.title))
        if key in index:
            existing = result[index[key]]
            if unit.confidence > existing.confidence:
                existing.title = unit.title
                existing.content = unit.content
                existing.source = unit.source or existing.source
                existing.confidence = unit.confidence
            for term in unit.terms:
                if term not in existing.terms:
                    existing.terms.append(term)
            for name in [*unit.sources, *([unit.source.split(" ", 1)[0]] if unit.source else [])]:
                if name and name not in existing.sources:
                    existing.sources.append(name)
        else:
            index[key] = len(result)
            result.append(unit.model_copy(deep=True))
    return result


def _dedupe_asides(asides: list[Aside]) -> list[Aside]:
    result: list[Aside] = []
    seen: set[str] = set()
    for aside in asides:
        key = _normalize(aside.text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(aside.model_copy(deep=True))
    return result


def units_block(units: list[Unit], *, content_chars: int) -> str:
    lines: list[str] = []
    for unit in units:
        header = f"[{unit.id}] {unit.kind} | {unit.title} | {unit.source or '?'}"
        if unit.terms:
            header += " | terms: " + ", ".join(unit.terms[:6])
        if unit.images:
            header += " | images: " + ", ".join(unit.images)
        lines.append(header)
        content = unit.content.strip().replace("\n", " ")
        if content_chars and len(content) > content_chars:
            content = content[:content_chars].rstrip() + " [...]"
        lines.append(f"  {content}")
    return "\n".join(lines)


def _images_block(images: list[ExtractedImage]) -> str:
    lines: list[str] = []
    for image in images:
        parts = [f"- {image.name} | page {image.page}"]
        if image.caption:
            parts.append(f"caption: {image.caption}")
        if image.nearby_text:
            parts.append(f"nearby text: {image.nearby_text}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _claim_orphan_images(
    units: list[Unit], images: tuple[ExtractedImage, ...], used: set[str]
) -> None:
    """Attach figures no chunk claimed to the closest unit by page number."""
    if not units:
        return
    for image in images:
        if image.id in used:
            continue
        target = min(units, key=lambda unit: abs(_unit_page(unit) - image.page))
        if image.name not in target.images:
            target.images.append(image.name)


def _unit_page(unit: Unit) -> int:
    match = re.search(r"page (\d+)", unit.source, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10**6


# --------------------------------------------------------------------------- #
# Grouping phase
# --------------------------------------------------------------------------- #


def units_for_sources(extract: ExtractResult, names: set[str]) -> list[Unit]:
    """Units that came from at least one of the given source names."""
    selected: list[Unit] = []
    for unit in extract.units:
        if any(name in unit.sources for name in names):
            selected.append(unit)
        elif any(unit.source == name or unit.source.startswith(f"{name} ") for name in names):
            selected.append(unit)
    return selected


def build_groups(
    sources: Sequence[TextSource],
    extract: ExtractResult,
    client: DeepSeekClient,
    logger: Logger,
    *,
    course: str,
) -> Grouping:
    """Ask the model which sources cover which topics, and in what order."""
    usable = [source for source in sources if source.text.strip()]
    if not usable:
        raise PlanValidationError("no readable source can be grouped")
    if len(usable) == 1:
        return Grouping(
            groups=[
                SourceGroup(
                    id="g1",
                    topic=usable[0].path.stem,
                    source_ids=[usable[0].id],
                    rationale="one source",
                )
            ]
        )

    logger.step("group: deciding which sources cover which topics")
    prompt = render(
        load_prompt("group.md"),
        {
            "course": course,
            "sources": _sources_block(usable),
            "units": _units_by_source_block(usable, extract),
        },
    )
    system = (
        "You organize study material into ordered topic groups. "
        "You only return valid JSON matching the requested shape."
    )
    payload = client.chat_json(system, prompt, phase="group")
    try:
        grouping = Grouping.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"group: invalid response shape: {exc.errors()[0]}") from exc

    grouping = normalize_grouping(grouping, usable, logger)
    logger.success(f"group: {len(grouping.groups)} group(s) from {len(usable)} sources")
    return grouping


def normalize_grouping(
    grouping: Grouping, sources: Sequence[TextSource], logger: Logger
) -> Grouping:
    """Keep every source exactly once and keep the proposed order."""
    valid = [source.id for source in sources]
    names = {source.id: source.name for source in sources}
    seen: set[str] = set()
    groups: list[SourceGroup] = []
    for group in grouping.groups:
        ids = [sid for sid in group.source_ids if sid in names and sid not in seen]
        if not ids:
            continue
        seen.update(ids)
        topic = group.topic.strip() or f"Group {len(groups) + 1}"
        groups.append(
            SourceGroup(
                id=f"g{len(groups) + 1}",
                topic=topic,
                source_ids=ids,
                rationale=group.rationale.strip(),
                confidence=group.confidence,
            )
        )

    missing = [sid for sid in valid if sid not in seen]
    if missing:
        labels = ", ".join(names[sid] for sid in missing)
        if groups:
            groups[-1].source_ids.extend(missing)
            grouping.todos.append(
                TodoItem(
                    kind="review",
                    text=(
                        "These sources were not assigned to a group and were added to the "
                        f"last one: {labels}"
                    ),
                )
            )
        else:
            groups = [
                SourceGroup(
                    id="g1",
                    topic=sources[0].path.stem,
                    source_ids=missing,
                    rationale="fallback group",
                )
            ]
        logger.warn(f"group: {len(missing)} source(s) were unassigned; added to a group")

    if not groups:
        groups = [
            SourceGroup(
                id="g1",
                topic=sources[0].path.stem,
                source_ids=valid,
                rationale="fallback group",
            )
        ]
    return Grouping(groups=groups, todos=grouping.todos)


def _sources_block(sources: Sequence[TextSource]) -> str:
    lines: list[str] = []
    for source in sources:
        text = " ".join(source.text.split())
        excerpt = text[:240] + (" [...]" if len(text) > 240 else "")
        lines.append(f"- {source.id} | {source.name} | {source.kind} | {excerpt}")
    return "\n".join(lines)


def _units_by_source_block(sources: Sequence[TextSource], extract: ExtractResult) -> str:
    lines: list[str] = []
    for source in sources:
        units = units_for_sources(extract, {source.name})
        titles = [unit.title for unit in units[:30]]
        detail = "; ".join(titles) if titles else "(no units extracted)"
        lines.append(f"- {source.id} ({source.name}): {detail}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reduce phase: outline
# --------------------------------------------------------------------------- #


def build_outline(
    context: ContextBundle,
    extract: ExtractResult,
    client: DeepSeekClient,
    logger: Logger,
    vault: Vault,
    *,
    config: Config | None = None,
    chooser: Chooser | None = None,
) -> Outline:
    logger.step("outline: assigning knowledge units to notes")
    units = list(extract.units)
    overflow_todos: list[TodoItem] = []
    if len(units) > 160:
        units = sorted(units, key=lambda unit: unit.confidence, reverse=True)[:160]
        overflow_todos.append(
            TodoItem(
                kind="review",
                text=(
                    "Some low-confidence units were not planned because the lecture produced "
                    "more than 160 units; review the transcript for uncovered topics."
                ),
            )
        )

    prompt = render(
        load_prompt("outline.md"),
        {
            "course": context.course,
            "moc_path": context.moc_path,
            "course_tags": ", ".join(context.tags) or "none inferred",
            "inventory": context.inventory_block() or "(empty vault)",
            "related": context.related_block() or "(none)",
            "units": units_block(units, content_chars=420),
        },
    )
    system = (
        "You are a careful curriculum planner. You only return valid JSON matching the "
        "requested shape and you never invent note paths."
    )
    payload = client.chat_json(system, prompt, phase="outline")

    try:
        outline = Outline.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"outline: invalid response shape: {exc.errors()[0]}") from exc

    outline = normalize_outline(
        outline, vault, context, logger, config=config, chooser=chooser
    )
    outline.todos.extend(overflow_todos)
    logger.success(f"outline: {len(outline.notes)} notes, {len(outline.todos)} todos")
    return outline


def normalize_outline(
    outline: Outline,
    vault: Vault,
    context: ContextBundle,
    logger: Logger,
    *,
    config: Config | None = None,
    chooser: Chooser | None = None,
) -> Outline:
    notes: list[NoteOutline] = []
    seen_paths: set[str] = set()
    by_title = {note.title.casefold(): note.path for note in vault.inventory()}

    for note in outline.notes:
        if note.path in seen_paths:
            continue
        seen_paths.add(note.path)

        if note.action == "create" and note.path.startswith("University/"):
            note.path = note.path.split("/", 1)[1]
        if note.action == "create":
            if config is None:
                existing_path = by_title.get(note.title.casefold())
                if existing_path:
                    note.action = "append"
                    note.path = existing_path
            else:
                candidates = resolve_targets(vault, note.title)
                choice = decide(candidates, config, chooser=chooser)
                target: NoteTarget = choice.target
                note.title = target.title
                note.path = target.path
                if target.kind == "create":
                    note.action = "create"
                    others = merge_candidates(candidates)
                    if others:
                        outline.todos.append(
                            TodoItem(
                                kind="review",
                                text=(
                                    f"'{note.title}' was planned as a new note; existing "
                                    "candidates were: "
                                    + "; ".join(item.label for item in others)
                                    + "."
                                ),
                                note_path=note.path,
                            )
                        )
                else:
                    note.action = target.kind
                    note.anchor = target.anchor
        if note.action == "create" and vault.exists(note.path):
            note.action = "append"

        if note.action in {"append", "merge"} and not vault.exists(note.path):
            note.action = "create"
            note.anchor = None

        if note.action == "create":
            if not note.tags:
                note.tags = list(context.tags)
            if not note.tags:
                outline.todos.append(
                    TodoItem(
                        kind="tag",
                        text=(
                            f"Note '{note.title}' was created without a course tag; add one "
                            "under #cs or #language so flashcards are visible."
                        ),
                        note_path=note.path,
                    )
                )
        else:
            if note.anchor:
                existing = vault.headings(note.path)
                if not any(_normalize(value) == _normalize(note.anchor) for value in existing):
                    outline.todos.append(
                        TodoItem(
                            kind="restructure",
                            text=(
                                f"Anchor '{note.anchor}' was not found in {note.path}; "
                                "the merge will append at the end of the note."
                            ),
                            note_path=note.path,
                        )
                    )
                    note.anchor = None
                    if note.action == "merge":
                        note.action = "append"

        if not note.unit_ids:
            logger.warn(f"outline: skipping note without units: {note.path}")
            continue
        notes.append(note)

    if not notes:
        notes = [_fallback_note(context, vault)]

    outline.notes = notes
    return outline


def _fallback_note(context: ContextBundle, vault: Vault) -> NoteOutline:
    title = (context.keywords[0] if context.keywords else "Lecture Notes").title()
    path = f"{title}.md"
    if vault.exists(path):
        return NoteOutline(action="append", path=path, title=title, unit_ids=["u1"])
    return NoteOutline(
        action="create", path=path, title=title, tags=list(context.tags), unit_ids=["u1"]
    )


# --------------------------------------------------------------------------- #
# Compose phase
# --------------------------------------------------------------------------- #


def compose_notes(
    outline: Outline,
    extract: ExtractResult,
    context: ContextBundle,
    client: DeepSeekClient,
    logger: Logger,
    vault: Vault,
    config: Config,
) -> Plan:
    template = load_prompt("compose.md")
    system = (
        "You write Obsidian Markdown and spaced-repetition cards. "
        "You only return valid JSON matching the requested shape."
    )
    units_by_id = {unit.id: unit for unit in extract.units}
    allowed_titles = set(context.titles) | {note.title for note in outline.notes}
    todos: list[TodoItem] = list(outline.todos)
    planned: list[PlannedNote] = []

    for index, note in enumerate(outline.notes, start=1):
        assigned = [units_by_id[unit_id] for unit_id in note.unit_ids if unit_id in units_by_id]
        if not assigned:
            todos.append(
                TodoItem(
                    kind="review",
                    text=f"No extracted units could be assigned to {note.path}.",
                    note_path=note.path,
                )
            )
            continue

        assigned_images: list[str] = []
        for unit in assigned:
            for name in unit.images:
                if name not in assigned_images:
                    assigned_images.append(name)

        existing = ""
        if note.action in {"append", "merge"}:
            if note.anchor:
                existing = vault.section_text(note.path, note.anchor, 6000)
            else:
                existing = vault.read(note.path)[-2200:]

        logger.step(f"compose: {index}/{len(outline.notes)} -> {note.path}")
        prompt = render(
            template,
            {
                "course": context.course,
                "note_path": note.path,
                "action": note.action,
                "anchor": note.anchor or "(none)",
                "tags": ", ".join(note.tags) or "(none)",
                "allowed_links": ", ".join(sorted(allowed_titles)) or "(none)",
                "existing": existing or "(empty)",
                "images": "\n".join(f"- {name}" for name in assigned_images) or "(none)",
                "units": units_block(assigned, content_chars=0),
            },
        )
        try:
            payload = client.chat_json(system, prompt, phase="compose", temperature=0.35)
        except LLMError as exc:
            logger.warn(f"compose failed for {note.path}: {exc}")
            todos.append(
                TodoItem(
                    kind="review",
                    text=f"Composition failed for {note.path}: {exc}",
                    note_path=note.path,
                )
            )
            continue

        composed = _validate_composed(payload, note.path)
        body = _clean_body(composed.body, note.title)
        if not body:
            todos.append(
                TodoItem(
                    kind="review",
                    text=f"The model returned an empty body for {note.path}.",
                    note_path=note.path,
                )
            )
            continue

        body, dropped = vault.sanitize_links(
            body, allowed_titles, allowed_images=set(assigned_images)
        )
        for target in dropped:
            todos.append(
                TodoItem(
                    kind="link",
                    text=f"Unresolved wikilink or embed '{target}' in {note.path} was flattened.",
                    note_path=note.path,
                )
            )

        flashcards = _valid_cards(composed.flashcards, note.path, logger)
        if not flashcards:
            todos.append(
                TodoItem(
                    kind="review",
                    text=f"No valid flashcards were produced for {note.path}.",
                    note_path=note.path,
                )
            )

        planned.append(
            PlannedNote(
                action=note.action,
                path=note.path,
                title=note.title,
                anchor=note.anchor,
                tags=note.tags,
                unit_ids=note.unit_ids,
                body=body,
                flashcards=flashcards,
                confidence=note.confidence,
            )
        )

    if not planned:
        raise PlanValidationError("no note could be composed from the outline")

    plan = Plan(
        course=context.course,
        moc_path=context.moc_path,
        notes=planned,
        todos=todos,
    )
    logger.success(
        f"compose: {len(planned)} notes, "
        f"{sum(len(note.flashcards) for note in planned)} candidate cards"
    )
    return plan


def _validate_composed(payload: dict[str, Any], path: str) -> ComposedNote:
    data = dict(payload)
    data["path"] = path
    try:
        return ComposedNote.model_validate(data)
    except ValidationError as exc:
        raise LLMError(f"compose: invalid response for {path}: {exc.errors()[0]}") from exc


def _valid_cards(cards: list[Flashcard], path: str, logger: Logger) -> list[Flashcard]:
    from .flashcards import validate_flashcard

    result: list[Flashcard] = []
    seen: set[str] = set()
    for card in cards:
        issues = validate_flashcard(card)
        if issues:
            logger.warn(f"compose: dropped card for {path}: {issues[0]}")
            continue
        key = _normalize(card.front)
        if key in seen:
            continue
        seen.add(key)
        card.note_path = path
        result.append(card)
    return result


def _clean_body(body: str, title: str) -> str:
    body = FRONTMATTER_RE.sub("", body.strip()).strip()
    match = FLASHCARD_BLOCK_RE.search(body)
    if match:
        body = body[: match.start()].rstrip()
    body = re.sub(
        rf"\A#\s+{re.escape(title)}\s*\n+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    return body.strip()


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def moc_headings(body: str) -> list[tuple[int, str]]:
    """Headings from a composed body, for MOC nesting."""
    result: list[tuple[int, str]] = []
    in_fence = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        parsed = heading_level_and_text(line)
        if not parsed:
            continue
        level, text = parsed
        if text.strip().casefold() in EXCLUDED_MOC_HEADINGS:
            continue
        if level <= 3:
            result.append((level, text))
    return result
