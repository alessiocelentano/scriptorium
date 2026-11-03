"""Recursive, filename-based tab completion.

Completion searches every file below one or more configured root directories.
Directories are never offered as candidates, but a directory name is still
useful: typing ``Downloads`` first lists files whose own name starts with it,
then everything inside directories whose name starts with it, newest first.
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
    limit: int = 50,
    max_scanned: int = 250_000,
) -> list[FileMatch]:
    """Return files below ``roots`` that belong to ``prefix``.

    The first group holds files whose own name starts with ``prefix``; the
    second holds files inside directories whose name starts with it, newest
    first.  Builds a fresh index and queries it.  The TUI reuses a
    :class:`CompletionIndex` across keystrokes instead; this helper keeps the
    one-shot behaviour for callers that do not need a cache.
    """
    return build_completion_index(roots, max_scanned=max_scanned).complete(prefix, limit=limit)


@dataclass(frozen=True, slots=True)
class _IndexedDirectory:
    """One directory in the index, linked to its parent by position."""

    name: str
    parent: int


@dataclass(slots=True)
class _IndexedFile:
    """One file in the index; ``modified`` is read lazily on first query."""

    path: str
    name: str
    relative: str
    root: Path
    group: str
    directory: int
    modified: float | None = None


@dataclass(frozen=True, slots=True)
class CompletionIndex:
    """A reusable snapshot of the files below the completion roots.

    Building the snapshot is the expensive part; filtering it for a new prefix
    only walks memory.  The snapshot has no time-based invalidation: the TUI
    drops it when the input line is cleared or when the roots change.
    """

    roots: tuple[Path, ...]
    directories: tuple[_IndexedDirectory, ...]
    files: tuple[_IndexedFile, ...]

    def complete(self, prefix: str, *, limit: int = 50) -> list[FileMatch]:
        """Return the two groups of matches for ``prefix``.

        1. files whose own name starts with ``prefix``, sorted by relative path;
        2. files contained in directories whose name starts with ``prefix``,
           sorted from the most recently modified file to the oldest.
        """
        needle = _needle(prefix)
        direct: list[FileMatch] = []
        contained: list[tuple[float, FileMatch]] = []
        directory_memo: dict[int, bool] = {}
        for entry in self.files:
            if entry.name.startswith(needle):
                direct.append(_file_match(entry))
                continue
            if not _directory_matches(self.directories, entry.directory, needle, directory_memo):
                continue
            if entry.modified is None:
                try:
                    entry.modified = os.stat(entry.path).st_mtime
                except OSError:
                    continue
            contained.append((entry.modified, _file_match(entry)))

        direct.sort(key=lambda match: (match.relative.casefold(), match.relative, match.group))
        contained.sort(
            key=lambda item: (
                -item[0],
                item[1].relative.casefold(),
                item[1].relative,
                item[1].group,
            )
        )
        matches = direct + [match for _, match in contained]
        return matches[:limit]


def build_completion_index(
    roots: Path | Sequence[Path],
    *,
    max_scanned: int = 250_000,
) -> CompletionIndex:
    """Scan ``roots`` once and return a reusable :class:`CompletionIndex`.

    Directory-match state is propagated during the walk, so the per-file cost
    is just a name check instead of constructing and splitting a path.  The
    index stores strings and a directory id per file; ``Path`` objects are only
    created for the matches actually returned.
    """
    requested = _normalise_roots(roots)
    root_list = [root for root in requested if root.is_dir()]
    directories: list[_IndexedDirectory] = []
    files: list[_IndexedFile] = []
    dedupe = len(root_list) > 1
    seen: set[str] = set()
    scanned = 0
    show_group = len(root_list) > 1

    for root in root_list:
        group = root.name if show_group else ""
        root_key = os.fspath(root)
        directories.append(_IndexedDirectory(root.name.casefold(), -1))
        root_directory = len(directories) - 1
        directory_ids = {root_key: root_directory}
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [
                name
                for name in subdirs
                if not name.startswith(".") and name not in SKIP_COMPLETION_DIRECTORIES
            ]
            if directory == root_key:
                directory_id = root_directory
            else:
                directory_id = directory_ids.get(directory)
                if directory_id is None:
                    parent = directory_ids.get(os.path.dirname(directory), root_directory)
                    directories.append(
                        _IndexedDirectory(os.path.basename(directory).casefold(), parent)
                    )
                    directory_id = len(directories) - 1
                    directory_ids[directory] = directory_id
            directory_abs = os.path.abspath(directory) if dedupe else ""
            for name in names:
                if name.startswith("."):
                    continue
                scanned += 1
                if scanned > max_scanned:
                    break
                path = os.path.join(directory, name)
                if dedupe:
                    identity = os.path.normcase(os.path.join(directory_abs, name))
                    if identity in seen:
                        continue
                    seen.add(identity)
                relative = os.path.relpath(path, root)
                if os.sep != "/":
                    relative = relative.replace(os.sep, "/")
                files.append(
                    _IndexedFile(
                        path=path,
                        name=name.casefold(),
                        relative=relative,
                        root=root,
                        group=group,
                        directory=directory_id,
                    )
                )
            if scanned > max_scanned:
                break
        if scanned > max_scanned:
            break

    return CompletionIndex(
        roots=tuple(requested),
        directories=tuple(directories),
        files=tuple(files),
    )


def _file_match(entry: _IndexedFile) -> FileMatch:
    return FileMatch(
        path=Path(entry.path),
        relative=entry.relative,
        root=entry.root,
        group=entry.group,
    )


def _needle(prefix: str) -> str:
    """The case-folded name fragment a prefix completes."""
    needle = prefix.strip().strip("'\"")
    if "/" in needle:
        needle = needle.rsplit("/", 1)[-1]
    return needle.casefold()


def _directory_matches(
    directories: tuple[_IndexedDirectory, ...],
    index: int,
    needle: str,
    memo: dict[int, bool],
) -> bool:
    """True when this directory or one of its ancestors starts with ``needle``."""
    chain: list[int] = []
    current = index
    while current != -1 and current not in memo:
        chain.append(current)
        current = directories[current].parent
    matched = memo[current] if current != -1 else False
    for node in reversed(chain):
        matched = matched or directories[node].name.startswith(needle)
        memo[node] = matched
    return memo[index]


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
