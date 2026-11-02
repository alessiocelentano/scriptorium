"""Vault access: inventory, MOC matching, and byte-preserving insert-only writes."""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .errors import VaultError
from .models import heading_level_and_text

IGNORED_DIRECTORIES = {
    ".git",
    ".obsidian",
    ".piwork",
    ".smart-env",
    ".trash",
    ".work",
    "Attachments",
    "Templates",
    "node_modules",
}

SKIP_SCAN_DIRECTORIES = IGNORED_DIRECTORIES | {"Archive"}

_FLASHCARD_HEADING_RE = re.compile(r"^#{1,6}\s+Flashcards\s*$", re.IGNORECASE)
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"(!?)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")


@dataclass(frozen=True)
class NoteInfo:
    path: str
    title: str
    tags: tuple[str, ...] = ()
    headings: tuple[str, ...] = ()

    @property
    def has_flashcard_tag(self) -> bool:
        return any(
            tag == "cs" or tag.startswith("cs/") or tag == "language" or tag.startswith("language/")
            for tag in self.tags
        )


@dataclass(frozen=True)
class MocMatch:
    path: str
    score: float


@dataclass
class ChangeRecord:
    path: str
    action: str
    added_lines: int = 0
    removed_lines: int = 0
    notes: str = ""
    diff: str = ""


@dataclass
class Vault:
    root: Path
    _inventory: list[NoteInfo] | None = field(default=None, init=False, repr=False)

    # -- paths --------------------------------------------------------------- #

    def resolve(self, relative: str) -> Path:
        relative = relative.replace("\\", "/").lstrip("/")
        candidate = (self.root / relative).resolve()
        root = self.root.resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise VaultError(f"path escapes the vault: {relative}")
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    def exists(self, relative: str) -> bool:
        return self.resolve(relative).is_file()

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise VaultError(f"note not found: {relative}")
        return _read_text(path)

    def read_or_empty(self, relative: str) -> str:
        path = self.resolve(relative)
        return _read_text(path) if path.is_file() else ""

    # -- inventory ----------------------------------------------------------- #

    def refresh(self) -> None:
        self._inventory = None

    def inventory(self) -> list[NoteInfo]:
        if self._inventory is not None:
            return self._inventory
        notes: list[NoteInfo] = []
        for directory, subdirs, files in os.walk(self.root):
            subdirs[:] = [
                name
                for name in subdirs
                if name not in SKIP_SCAN_DIRECTORIES and not name.startswith(".")
            ]
            for name in files:
                if not name.endswith(".md") or name.startswith("."):
                    continue
                path = Path(directory) / name
                try:
                    text = _read_text(path)
                except (OSError, UnicodeDecodeError):
                    continue
                notes.append(
                    NoteInfo(
                        path=self.relative(path),
                        title=path.stem,
                        tags=tuple(parse_tags(text)),
                        headings=tuple(headings(text)),
                    )
                )
        notes.sort(key=lambda note: note.path.lower())
        self._inventory = notes
        return notes

    def titles(self) -> list[str]:
        return [note.title for note in self.inventory()]

    def note_info(self, relative: str) -> NoteInfo | None:
        for note in self.inventory():
            if note.path == relative:
                return note
        return None

    def headings(self, relative: str) -> list[str]:
        return headings(self.read(relative))

    # -- MOC discovery ------------------------------------------------------- #

    def find_mocs(self, roots: Sequence[Path] | None = None) -> list[str]:
        """Return MOC notes under ``roots`` (default: the whole vault), recursively.

        Roots outside the vault are ignored: a MOC can only be linked and updated
        in place when it lives inside the vault.  A file counts as a MOC when its
        name contains ``moc`` or when it is the vault's ``University Home``.
        """
        vault_root = self.root.resolve()
        bases: list[Path] = []
        if roots:
            for raw in roots:
                candidate = Path(raw).expanduser()
                resolved = (
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (vault_root / candidate).resolve()
                )
                if resolved != vault_root and not resolved.is_relative_to(vault_root):
                    continue
                if resolved.is_dir() and resolved not in bases:
                    bases.append(resolved)
        else:
            bases.append(vault_root)

        paths: set[str] = set()
        for base in bases:
            for path in base.rglob("*.md"):
                try:
                    relative = path.relative_to(vault_root)
                except ValueError:
                    continue
                if any(part in SKIP_SCAN_DIRECTORIES for part in relative.parts):
                    continue
                if "moc" in path.stem.casefold() or path.stem == "University Home":
                    paths.add(relative.as_posix())
        return sorted(paths)

    def match_moc(
        self,
        query: str,
        limit: int = 5,
        roots: Sequence[Path] | None = None,
    ) -> list[MocMatch]:
        """Rank MOCs against a query built from the course name and source text.

        The title match keeps working when only a course name is available; the
        content overlap rewards a MOC whose own body mentions the source terms.
        """
        query_norm = _normalize(query)
        query_tokens = _tokens(query)
        matches: list[MocMatch] = []
        for path in self.find_mocs(roots):
            title = Path(path).stem
            title_norm = _normalize(title)
            title_tokens = _tokens(title)
            union = query_tokens | title_tokens
            jaccard = len(query_tokens & title_tokens) / len(union) if union else 0.0
            ratio = difflib.SequenceMatcher(None, query_norm, title_norm).ratio()
            title_score = 0.62 * jaccard + 0.38 * ratio
            if query_norm and query_norm in title_norm:
                title_score += 0.2
            if query_tokens and query_tokens <= title_tokens:
                title_score += 0.1
            title_score = min(title_score, 1.0)

            content_tokens = _tokens(self.read_or_empty(path))
            overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            content_score = min(1.0, overlap * 2.5)
            score = 0.65 * title_score + 0.35 * content_score
            matches.append(MocMatch(path=path, score=round(min(score, 1.0), 4)))
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    # -- link validation ----------------------------------------------------- #

    def sanitize_links(
        self,
        body: str,
        allowed_titles: set[str],
        *,
        allowed_images: set[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Drop wikilinks whose target does not exist. Returns (body, dropped).

        Embeds (``![[...]]``) are kept only for attachments this run created:
        ``allowed_images`` holds vault-relative paths, and a bare file name is
        accepted as well because Obsidian resolves either form.
        """
        allowed = {title.casefold() for title in allowed_titles}
        image_targets: set[str] = set()
        for image in allowed_images or ():
            path = image.replace("\\", "/").strip().lstrip("/")
            image_targets.add(path.casefold())
            image_targets.add(path.rsplit("/", 1)[-1].casefold())
        dropped: list[str] = []

        def replace(match: re.Match[str]) -> str:
            bang, target, alias = match.group(1), match.group(2).strip(), match.group(3)
            display = alias or target
            if bang and (
                target.replace("\\", "/").lstrip("/").casefold() in image_targets
                or target.rsplit("/", 1)[-1].casefold() in image_targets
            ):
                return match.group(0)
            if target.casefold() in allowed:
                return match.group(0)
            dropped.append(target)
            return (bang + display) if bang else display

        return _WIKILINK_RE.sub(replace, body), dropped

    # -- section utilities --------------------------------------------------- #

    def section_text(self, relative: str, heading: str, max_chars: int = 4000) -> str:
        text = self.read(relative)
        lines = text.splitlines()
        target = _normalize_heading(heading)
        start = None
        level = 0
        for index, line in enumerate(lines):
            parsed = heading_level_and_text(line)
            if parsed and _normalize_heading(parsed[1]) == target:
                start = index
                level = parsed[0]
                break
        if start is None:
            return ""
        end = len(lines)
        for index in range(start + 1, len(lines)):
            parsed = heading_level_and_text(lines[index])
            if parsed and parsed[0] <= level:
                end = index
                break
        body = "\n".join(lines[start:end]).strip()
        return body[:max_chars]

    def find_flashcards_heading(self, text: str) -> tuple[int, int] | None:
        """Return (line index, heading level) for a Flashcards heading."""
        for index, line in enumerate(text.splitlines()):
            if _FLASHCARD_HEADING_RE.match(line.strip()):
                parsed = heading_level_and_text(line)
                return index, parsed[0] if parsed else 2
        return None

    # -- writes -------------------------------------------------------------- #

    def write_new(self, relative: str, text: str) -> ChangeRecord:
        path = self.resolve(relative)
        if path.exists():
            raise VaultError(f"refusing to overwrite existing note: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes(path, text.encode("utf-8"))
        lines = text.count("\n")
        return ChangeRecord(
            path=relative,
            action="create",
            added_lines=lines,
            diff=_unified_diff("", text, relative),
        )

    def apply_change(
        self,
        relative: str,
        new_text: str,
        *,
        expect_only_additions: bool = True,
        action: str = "update",
    ) -> ChangeRecord:
        """Write ``new_text``, refusing to change pre-existing lines unless allowed.

        Insertions are the safe default: a diff that removes or changes a line
        is rolled back and reported.  ``expect_only_additions=False`` is only for
        a merge the user reviewed (see ``apply.py``), and it still returns the
        removed-line count so the report shows the size of the rewrite.
        """
        path = self.resolve(relative)
        original = _read_text(path) if path.is_file() else None
        if original == new_text:
            return ChangeRecord(path=relative, action="unchanged")

        _write_bytes(path, new_text.encode("utf-8"))

        if expect_only_additions and original is not None:
            updated = _read_text(path)
            removed = _removed_lines(original, updated)
            if removed:
                _write_bytes(path, original.encode("utf-8"))
                raise VaultError(
                    f"refusing non-insertion change to {relative}; removed or changed "
                    f"{len(removed)} pre-existing line(s)"
                )
            added = _added_lines(original, updated)
            return ChangeRecord(
                path=relative,
                action=action,
                added_lines=added,
                removed_lines=0,
                diff=_unified_diff(original, updated, relative),
            )

        added = _added_lines(original or "", new_text)
        removed = _removed_lines(original or "", new_text)
        return ChangeRecord(
            path=relative,
            action=action,
            added_lines=added,
            removed_lines=len(removed),
            diff=_unified_diff(original or "", new_text, relative),
        )

    def preview_change(self, relative: str, new_text: str) -> str:
        """Unified diff between the current note and ``new_text``."""
        try:
            original = self.read(relative)
        except VaultError:
            original = ""
        return _unified_diff(original, new_text, relative)

    def insert_under_heading(
        self,
        relative: str,
        heading: str,
        block: str,
        *,
        tight: bool = False,
        missing_ok: bool = False,
    ) -> str:
        """Return the note text with ``block`` inserted at the end of a section."""
        text = self.read(relative)
        lines = text.splitlines(keepends=True)
        target = _normalize_heading(heading)
        start = None
        level = 0
        for index, line in enumerate(lines):
            parsed = heading_level_and_text(line)
            if parsed and _normalize_heading(parsed[1]) == target:
                start = index
                level = parsed[0]
                break
        if start is None:
            if missing_ok:
                block_text = block.strip("\n")
                prefix = "" if text.endswith("\n") or not text else "\n"
                return text + prefix + block_text + "\n"
            raise VaultError(f"heading not found in {relative}: {heading!r}")

        end = len(lines)
        for index in range(start + 1, len(lines)):
            parsed = heading_level_and_text(lines[index])
            if parsed and parsed[0] <= level:
                end = index
                break

        insert_at = end
        while insert_at > start + 1 and lines[insert_at - 1].strip() == "":
            insert_at -= 1

        block_lines = [line + "\n" for line in block.strip("\n").split("\n")]
        if not tight and insert_at > 0 and lines[insert_at - 1].strip() != "":
            block_lines.insert(0, "\n")
        needs_blank_after = insert_at < len(lines) and lines[insert_at].strip() != ""

        new_lines = (
            lines[:insert_at]
            + block_lines
            + (["\n"] if needs_blank_after else [])
            + lines[insert_at:]
        )
        return "".join(new_lines)

    def replace_section(
        self,
        relative: str,
        heading: str,
        block: str,
        *,
        missing_ok: bool = False,
    ) -> str:
        """Return the note text with the whole section replaced by ``block``.

        The section includes its heading line; ``block`` must therefore start
        with the heading that replaces it.  Used by a reviewed merge, which may
        edit or delete lines the append path is not allowed to touch.
        """
        text = self.read(relative)
        lines = text.splitlines(keepends=True)
        target = _normalize_heading(heading)
        start = None
        level = 0
        for index, line in enumerate(lines):
            parsed = heading_level_and_text(line)
            if parsed and _normalize_heading(parsed[1]) == target:
                start = index
                level = parsed[0]
                break
        if start is None:
            if missing_ok:
                return self.append_at_end(relative, block)
            raise VaultError(f"heading not found in {relative}: {heading!r}")

        end = len(lines)
        for index in range(start + 1, len(lines)):
            parsed = heading_level_and_text(lines[index])
            if parsed and parsed[0] <= level:
                end = index
                break

        block_lines = [line + "\n" for line in block.strip("\n").split("\n")]
        replacement = block_lines
        if end < len(lines) and replacement and replacement[-1].strip() != "":
            replacement = [*replacement, "\n"]
        return "".join(lines[:start] + replacement + lines[end:])

    def append_at_end(self, relative: str, block: str) -> str:
        text = self.read(relative)
        suffix = "" if text.endswith("\n") else "\n"
        return text + suffix + block.rstrip("\n") + "\n"

    def ensure_flashcards_section(self, relative: str) -> str:
        text = self.read(relative)
        if self.find_flashcards_heading(text) is not None:
            return text
        return self.append_at_end(relative, "\n---\n\n## Flashcards")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def parse_tags(text: str) -> list[str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return []
    tags: list[str] = []
    in_tags = False
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if re.match(r"^tags\s*:", stripped):
            in_tags = True
            inline = stripped.split(":", 1)[1].strip()
            if inline.startswith("["):
                tags.extend(
                    tag.strip().strip("\"'").lstrip("#")
                    for tag in inline.strip("[]").split(",")
                    if tag.strip()
                )
                in_tags = False
            continue
        if in_tags:
            item = re.match(r"^-\s*(.+)$", stripped)
            if item:
                tags.append(item.group(1).strip().strip("\"'").lstrip("#"))
                continue
            if stripped:
                in_tags = False
    return [tag for tag in tags if tag]


def headings(text: str) -> list[str]:
    result: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        parsed = heading_level_and_text(line)
        if parsed:
            result.append(parsed[1])
    return result


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _tokens(value: str) -> set[str]:
    stop = {"moc", "university", "and", "of", "the", "for", "a", "an", "course"}
    return set(re.findall(r"[a-z0-9]+", value.lower())) - stop


def _normalize_heading(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^#{1,6}\s+", "", value)
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", value)
    return " ".join(value.lower().split())


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _diff_lines(original: str, updated: str):
    return difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        lineterm="",
        n=0,
    )


def _removed_lines(original: str, updated: str) -> list[str]:
    removed: list[str] = []
    for line in _diff_lines(original, updated):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
    return removed


def _unified_diff(original: str, updated: str, path: str, context: int = 4) -> str:
    if original == updated:
        return ""
    lines = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=context,
    )
    return "\n".join(lines)


def added_lines(original: str, updated: str) -> list[str]:
    """Return the lines the diff adds, without the leading ``+``."""
    lines: list[str] = []
    for line in _diff_lines(original, updated):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return lines


def _added_lines(original: str, updated: str) -> int:
    return len(added_lines(original, updated))


def only_additions(original: str, updated: str) -> bool:
    """True when the diff removes nothing and preserves final-newline state."""
    if original.endswith("\n") != updated.endswith("\n"):
        return False
    return not _removed_lines(original, updated)
