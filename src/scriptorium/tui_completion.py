"""Recursive, filename-based tab completion.

Completion searches every file below one or more configured root directories.
Directories are never offered as candidates: typing ``lec`` finds
``lectures/lecture-01.mp4`` even though the intermediate directory is not shown.
When more than one root is configured, results carry the root name in
``display`` so the same relative path in different roots stays distinguishable.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SKIP_COMPLETION_DIRECTORIES = {
    ".cache",
    ".git",
    ".hg",
    ".svn",
    ".Trash",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "Library",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass(frozen=True)
class FileMatch:
    path: Path
    relative: str
    root: Path | None = None
    group: str = ""

    @property
    def display(self) -> str:
        if self.group:
            return f"{self.group}/{self.relative}"
        return self.relative


def _normalise_roots(roots: Path | Sequence[Path]) -> list[Path]:
    if isinstance(roots, (str, Path)):
        candidates = [roots]
    else:
        candidates = list(roots)
    unique: list[Path] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path not in unique:
            unique.append(path)
    return unique


def complete_files(
    roots: Path | Sequence[Path],
    prefix: str,
    *,
    limit: int = 200,
    max_scanned: int = 250_000,
) -> list[FileMatch]:
    """Return files below ``roots`` whose name starts with ``prefix``.

    The search is recursive and matches on the basename only, so a prefix never
    selects a directory.  Results are sorted by relative path and deduplicated
    by absolute path.
    """
    root_list = [root for root in _normalise_roots(roots) if root.is_dir()]
    if not root_list:
        return []

    needle = prefix.strip().strip("'\"")
    if "/" in needle:
        needle = needle.rsplit("/", 1)[-1]
    needle = needle.casefold()

    matches: list[FileMatch] = []
    seen: set[Path] = set()
    scanned = 0
    show_group = len(root_list) > 1
    for root in root_list:
        group = root.name if show_group else ""
        for directory, subdirs, files in os.walk(root, followlinks=False):
            subdirs[:] = [
                name
                for name in subdirs
                if not name.startswith(".") and name not in SKIP_COMPLETION_DIRECTORIES
            ]
            for name in files:
                if name.startswith("."):
                    continue
                scanned += 1
                if scanned > max_scanned:
                    break
                if not name.casefold().startswith(needle):
                    continue
                path = Path(directory) / name
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    relative = path.relative_to(root).as_posix()
                except ValueError:
                    relative = path.as_posix()
                matches.append(FileMatch(path=path, relative=relative, root=root, group=group))
            if scanned > max_scanned:
                break
        if scanned > max_scanned:
            break

    matches.sort(key=lambda match: (match.relative.casefold(), match.relative, match.group))
    return matches[:limit]


def complete_paths(prefix: str, *, include_files: bool = True, limit: int = 200) -> list[FileMatch]:
    """Return filesystem entries that continue a typed path.

    A field that holds one path keeps the whole value as the prefix, so spaces
    are part of a name instead of a token separator.  Directories carry a
    trailing slash, which makes it easy to continue completing inside them;
    files are offered when ``include_files`` is true.  The display keeps the
    form the user typed (``~/Downloads``).
    """
    raw = prefix.strip().strip("'\"")
    if not raw or raw == "~":
        raw = "~/"
    if raw.endswith(("/", os.sep)):
        typed_dir = raw
        base = ""
        directory = Path(raw).expanduser()
    else:
        cut = raw.rfind("/")
        if cut >= 0:
            typed_dir = raw[: cut + 1]
            base = raw[cut + 1 :]
            directory = Path(typed_dir).expanduser()
        else:
            typed_dir = ""
            base = raw
            directory = Path.home()
    if not directory.is_dir():
        return []
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name.casefold())
    except OSError:
        return []
    matches: list[FileMatch] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            suffix = "/"
        elif include_files and entry.is_file():
            suffix = ""
        else:
            continue
        if not entry.name.casefold().startswith(base.casefold()):
            continue
        matches.append(
            FileMatch(path=entry, relative=f"{typed_dir}{entry.name}{suffix}", root=None, group="")
        )
        if len(matches) >= limit:
            break
    return matches


def complete_directories(prefix: str, *, limit: int = 200) -> list[FileMatch]:
    """Return directories that continue a typed filesystem path."""
    return complete_paths(prefix, include_files=False, limit=limit)


def complete_note_paths(
    paths: Sequence[str],
    prefix: str,
    *,
    limit: int = 200,
) -> list[FileMatch]:
    """Return vault-relative note paths that continue ``prefix``.

    The candidates already live inside the vault, so matching them directly
    preserves the vault-relative path.  A bare file stem is accepted too, which
    lets the user type ``Foundations`` or ``University/Foundations``.
    """
    needle = prefix.strip().strip("'\"").replace("\\", "/").casefold()
    matches: list[FileMatch] = []
    for raw in paths:
        relative = str(raw).replace("\\", "/")
        stem = Path(relative).stem.casefold()
        if not (relative.casefold().startswith(needle) or stem.startswith(needle)):
            continue
        matches.append(FileMatch(path=Path(relative), relative=relative))
        if len(matches) >= limit:
            break
    return matches


def token_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for shell-like tokens, honouring quotes."""
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        start = index
        quote = ""
        while index < length:
            char = text[index]
            if quote:
                if char == quote:
                    quote = ""
                elif char == "\\":
                    index += 1
            else:
                if char in {"'", '"'}:
                    quote = char
                elif char.isspace():
                    break
            index += 1
        spans.append((start, min(index, length)))
    return spans


def token_at(text: str, cursor: int) -> tuple[int, int, str]:
    """Return the span and raw text of the token containing ``cursor``."""
    if cursor > 0 and cursor <= len(text) and text[cursor - 1].isspace():
        return cursor, cursor, ""
    spans = token_spans(text)
    for start, end in spans:
        if start <= cursor <= end:
            return start, end, text[start:end]
    return cursor, cursor, ""


def token_prefix(raw: str) -> str:
    """Extract the unquoted value of a partial token."""
    if not raw:
        return ""
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        value = raw.lstrip("'\"")
        return value.rsplit("/", 1)[-1]
    if not parts:
        return ""
    return str(parts[-1]).rsplit("/", 1)[-1]


def replace_token(text: str, start: int, end: int, value: str) -> tuple[str, int]:
    """Replace a token span and return the new text and cursor position."""
    rendered = shlex.quote(value)
    new_text = text[:start] + rendered + text[end:]
    return new_text, start + len(rendered)
