"""Parse the TUI input line into pipeline arguments.

The input line accepts a space separated list of source paths: lecture videos,
slides, notes, PDFs, spreadsheets.  Nothing else is typed there.  A Markdown
file that looks like a MOC is kept as a hint for the MOC phase instead of being
treated as source material, but it never makes the input invalid.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .sources import clean_course_name

MEDIA_SUFFIXES = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

SOURCE_SUFFIXES = {
    ".csv",
    ".docx",
    ".htm",
    ".html",
    ".json",
    ".markdown",
    ".md",
    ".org",
    ".pdf",
    ".pptx",
    ".rst",
    ".tex",
    ".txt",
    ".xlsx",
}


@dataclass
class InputPlan:
    sources: list[Path] = field(default_factory=list)
    moc_hints: list[Path] = field(default_factory=list)
    course: str = ""
    error: str = ""

    @property
    def moc(self) -> Path | None:
        return self.moc_hints[0] if self.moc_hints else None

    @property
    def media(self) -> list[Path]:
        return [path for path in self.sources if path.suffix.lower() in MEDIA_SUFFIXES]

    @property
    def documents(self) -> list[Path]:
        return [path for path in self.sources if path.suffix.lower() not in MEDIA_SUFFIXES]

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.sources)


def _roots(root: Path | Sequence[Path]) -> list[Path]:
    if isinstance(root, (str, Path)):
        return [Path(root)]
    return [Path(candidate) for candidate in root]


def _resolve(token: str, root: Path | Sequence[Path]) -> Path:
    path = Path(token).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    roots = _roots(root)
    for candidate in roots:
        resolved = candidate / path
        if resolved.exists():
            return resolved
    return roots[0] / path if roots else path


def _is_moc(path: Path) -> bool:
    if path.suffix.lower() not in {".md", ".markdown"}:
        return False
    if "moc" in path.stem.casefold():
        return True
    return any(part.casefold() == "university" for part in path.parts)


def parse_input(text: str, root: Path | Sequence[Path]) -> InputPlan:
    """Split the input line into sources and MOC hints.

    ``root`` may be a single completion root or a list of them; relative tokens
    are resolved against the first root that contains the file.
    """
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        return InputPlan(error=f"unterminated quote: {exc}")

    plan = InputPlan()
    seen: set[Path] = set()
    for token in tokens:
        path = _resolve(token, root)
        if path in seen:
            continue
        seen.add(path)
        if _is_moc(path):
            plan.moc_hints.append(path)
        else:
            plan.sources.append(path)

    if not plan.sources:
        return InputPlan(
            moc_hints=plan.moc_hints,
            error="add at least one source path (video, slides, notes, PDF)",
        )

    first = plan.sources[0]
    plan.course = clean_course_name(Path(plan.moc).stem if plan.moc else first.stem)
    return plan


def missing_paths(plan: InputPlan) -> list[Path]:
    paths = [*plan.sources, *plan.moc_hints]
    return [path for path in paths if not path.exists()]
