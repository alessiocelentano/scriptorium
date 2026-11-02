"""Shared data contracts.

The language model produces JSON matching :class:`ExtractResult`, :class:`Outline`,
and :class:`ComposedNote`. Everything is validated here before it reaches a writer.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    """Base model that tolerates extra keys produced by the language model."""

    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #


def format_timestamp(seconds: float, *, srt: bool = False) -> str:
    """Format seconds as ``HH:MM:SS`` or ``HH:MM:SS,mmm`` (SRT)."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    whole = int(secs)
    if srt:
        millis = int(round((secs - whole) * 1000))
        if millis == 1000:
            whole += 1
            millis = 0
        return f"{int(hours):02d}:{int(minutes):02d}:{whole:02d},{millis:03d}"
    return f"{int(hours):02d}:{int(minutes):02d}:{whole:02d}"


class Segment(Contract):
    start: float = 0.0
    end: float = 0.0
    text: str = ""

    @field_validator("text")
    @classmethod
    def _clean(cls, value: str) -> str:
        return " ".join(value.split())


class Transcript(Contract):
    language: str = "en"
    duration: float = 0.0
    segments: list[Segment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text)

    def timed_text(self) -> str:
        """Transcript with a timestamp per segment, for source references."""
        return "\n".join(
            f"[{format_timestamp(segment.start)}] {segment.text}"
            for segment in self.segments
            if segment.text
        )

    def to_srt(self) -> str:
        blocks: list[str] = []
        for index, segment in enumerate(self.segments, start=1):
            if not segment.text:
                continue
            end = segment.end if segment.end > segment.start else segment.start
            blocks.append(
                f"{index}\n"
                f"{format_timestamp(segment.start, srt=True)} --> "
                f"{format_timestamp(end, srt=True)}\n"
                f"{segment.text}\n"
            )
        return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Map phase: knowledge units
# --------------------------------------------------------------------------- #


class Unit(Contract):
    id: str = ""
    kind: str = "concept"
    title: str
    content: str
    source: str = ""
    sources: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    confidence: float = 0.8

    @field_validator("sources")
    @classmethod
    def _sources(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            name = str(value).strip()
            if name and name not in result:
                result.append(name)
        return result

    @field_validator("images")
    @classmethod
    def _images(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            name = str(value).strip().strip("'\"")
            name = name.removeprefix("![[")
            name = name.removesuffix("]]").strip()
            if name and name not in result:
                result.append(name)
        return result

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        value = (value or "concept").strip().lower()
        allowed = {
            "definition",
            "example",
            "formula",
            "exercise",
            "remark",
            "term",
            "procedure",
            "comparison",
            "concept",
        }
        return value if value in allowed else "concept"


class Aside(Contract):
    """Off-topic material kept out of the notes and shown only in the TUI.

    Lecture recordings contain comments about exams, the university, useful
    links and curiosities that do not belong in a study note.  The extract
    phase separates them here instead of dropping them.
    """

    kind: str = "remark"
    text: str
    source: str = ""

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        value = (value or "remark").strip().lower()
        allowed = {
            "exam",
            "university",
            "lesson",
            "link",
            "curiosity",
            "advice",
            "logistics",
            "remark",
        }
        return value if value in allowed else "remark"

    @field_validator("text", "source")
    @classmethod
    def _clean(cls, value: str) -> str:
        return " ".join(value.split())


class ExtractResult(Contract):
    units: list[Unit] = Field(default_factory=list)
    asides: list[Aside] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Grouping phase: ordered topic groups
# --------------------------------------------------------------------------- #


class SourceGroup(Contract):
    id: str = ""
    topic: str = ""
    source_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.8

    @field_validator("source_ids")
    @classmethod
    def _source_ids(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            value = str(value).strip()
            if value and value not in result:
                result.append(value)
        return result


class Grouping(Contract):
    groups: list[SourceGroup] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reduce phase: outline
# --------------------------------------------------------------------------- #


def _clean_note_path(value: str) -> str:
    value = (value or "").strip().strip("\"'").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.lstrip("/")
    return value


class NoteOutline(Contract):
    action: Literal["create", "append", "merge"] = "create"
    path: str
    title: str = ""
    anchor: str | None = None
    tags: list[str] = Field(default_factory=list)
    unit_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.8

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        value = _clean_note_path(value)
        if not value.lower().endswith(".md"):
            value += ".md"
        parts = value.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"invalid note path: {value!r}")
        return value

    @field_validator("tags")
    @classmethod
    def _tags(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            value = value.strip().lstrip("#").strip()
            if value and value not in result:
                result.append(value)
        return result

    @model_validator(mode="after")
    def _title_and_anchor(self) -> NoteOutline:
        if not self.title:
            self.title = self.path.rsplit("/", 1)[-1].removesuffix(".md")
        if self.anchor is not None:
            self.anchor = self.anchor.strip() or None
        return self


class TodoItem(Contract):
    kind: str = "unclear"
    text: str
    note_path: str | None = None

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        value = (value or "unclear").strip().lower()
        allowed = {"image", "restructure", "unclear", "missing", "link", "tag", "review"}
        return value if value in allowed else "unclear"


class Outline(Contract):
    notes: list[NoteOutline] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Compose phase
# --------------------------------------------------------------------------- #


FlashcardKind = Literal["single", "reversed", "multi", "cloze"]


class Flashcard(Contract):
    front: str
    back: str = ""
    kind: str = "single"
    note_path: str | None = None

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str) -> str:
        value = (value or "single").strip().lower()
        return value if value in {"single", "reversed", "multi", "cloze"} else "single"


class ComposedNote(Contract):
    path: str
    body: str = ""
    flashcards: list[Flashcard] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        value = _clean_note_path(value)
        return value if value.lower().endswith(".md") else value + ".md"

    @field_validator("body")
    @classmethod
    def _body(cls, value: str) -> str:
        return value.replace("\r\n", "\n").strip("\n")


# --------------------------------------------------------------------------- #
# Assembled plan
# --------------------------------------------------------------------------- #


class PlannedNote(Contract):
    action: Literal["create", "append", "merge"]
    path: str
    title: str
    anchor: str | None = None
    tags: list[str] = Field(default_factory=list)
    unit_ids: list[str] = Field(default_factory=list)
    body: str = ""
    flashcards: list[Flashcard] = Field(default_factory=list)
    confidence: float = 0.8


class Plan(Contract):
    course: str
    moc_path: str | None = None
    notes: list[PlannedNote] = Field(default_factory=list)
    asides: list[Aside] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    stats: dict[str, int | float | str] = Field(default_factory=dict)

    def note(self, path: str) -> PlannedNote | None:
        for note in self.notes:
            if note.path == path:
                return note
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def heading_level_and_text(line: str) -> tuple[int, str] | None:
    """Return ``(level, text)`` for an ATX heading line, else ``None``."""
    match = _HEADING_RE.match(line.rstrip("\n"))
    if not match:
        return None
    text = match.group(2).strip()
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", text)
    return len(match.group(1)), text
