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


def path_exists(path: Path) -> bool:
    """``Path.exists`` that treats an over-long or unreadable name as missing.

    A pasted list can contain one very long line; probing it with ``exists()``
    raises ``OSError(ENAMETOOLONG)`` on some platforms, which would abort the
    paste instead of reporting the path as not found.
    """
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _resolve(token: str, root: Path | Sequence[Path]) -> Path:
    path = Path(token).expanduser()
    if path.is_absolute():
        return path
    if path_exists(path):
        return path.resolve()
    roots = _roots(root)
    for candidate in roots:
        resolved = candidate / path
        if path_exists(resolved):
            return resolved
    return roots[0] / path if roots else path


def _is_moc(path: Path) -> bool:
    if path.suffix.lower() not in {".md", ".markdown"}:
        return False
    if "moc" in path.stem.casefold():
        return True
    return any(part.casefold() == "university" for part in path.parts)


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _plan_from_tokens(tokens: Sequence[str], root: Path | Sequence[Path]) -> InputPlan:
    plan = InputPlan()
    seen: set[Path] = set()
    for token in tokens:
        if not token:
            continue
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


def parse_input(text: str, root: Path | Sequence[Path]) -> InputPlan:
    """Split the input line into sources and MOC hints.

    ``root`` may be a single completion root or a list of them; relative tokens
    are resolved against the first root that contains the file.
    """
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        return InputPlan(error=f"unterminated quote: {exc}")
    return _plan_from_tokens(tokens, root)


def parse_pasted_input(text: str, root: Path | Sequence[Path]) -> InputPlan:
    """Parse a clipboard paste as a list of source paths.

    A multi-line paste is one path per line, so a path containing spaces is
    kept whole.  A single-line paste is split with the same shell-like rules as
    the input line, unless the entire line already names an existing file.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in normalized:
        tokens = [_strip_quotes(line.strip()) for line in normalized.split("\n") if line.strip()]
    else:
        stripped = normalized.strip()
        if not stripped:
            return InputPlan(error="add at least one source path (video, slides, notes, PDF)")
        if path_exists(_resolve(stripped, root)):
            tokens = [_strip_quotes(stripped)]
        else:
            try:
                tokens = shlex.split(stripped, posix=True)
            except ValueError as exc:
                return InputPlan(error=f"unterminated quote: {exc}")
    return _plan_from_tokens(tokens, root)


def missing_paths(plan: InputPlan) -> list[Path]:
    paths = [*plan.sources, *plan.moc_hints]
    return [path for path in paths if not path_exists(path)]
