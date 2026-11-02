"""Copy-on-write vault used by ``--dry-run``.

A dry run must show the real change set without touching the vault, so the
pipeline runs against a sandbox that reads through to the vault and writes into
a throwaway directory under the system temp directory.  The directory is deleted
when the run ends; only the job artifacts and the displayed diffs survive.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .errors import VaultError
from .vault import (
    ChangeRecord,
    NoteInfo,
    Vault,
    _read_text,
    _removed_lines,
    _unified_diff,
    _write_bytes,
    headings,
    parse_tags,
)


@dataclass
class SandboxVault(Vault):
    """Vault overlay: reads fall through, writes stay in ``directory``."""

    background: Vault = field(default_factory=lambda: Vault(Path.cwd()))
    directory: Path = field(default_factory=Path)

    @classmethod
    def create(cls, background: Vault, *, prefix: str = "scriptorium-dry-run-") -> SandboxVault:
        directory = Path(tempfile.mkdtemp(prefix=prefix))
        return cls(root=background.root, background=background, directory=directory)

    # -- reads --------------------------------------------------------------- #

    def exists(self, relative: str) -> bool:
        overlay = self._overlay(relative)
        return overlay.is_file() or self.background.exists(relative)

    def read(self, relative: str) -> str:
        overlay = self._overlay(relative)
        if overlay.is_file():
            return _read_text(overlay)
        return self.background.read(relative)

    def read_or_empty(self, relative: str) -> str:
        overlay = self._overlay(relative)
        if overlay.is_file():
            return _read_text(overlay)
        return self.background.read_or_empty(relative)

    def inventory(self) -> list[NoteInfo]:
        if self._inventory is not None:
            return self._inventory
        notes = {note.path: note for note in self.background.inventory()}
        if self.directory.is_dir():
            for path in self.directory.rglob("*.md"):
                relative = path.relative_to(self.directory).as_posix()
                try:
                    text = _read_text(path)
                except (OSError, UnicodeDecodeError):
                    continue
                notes[relative] = NoteInfo(
                    path=relative,
                    title=path.stem,
                    tags=tuple(parse_tags(text)),
                    headings=tuple(headings(text)),
                )
        self._inventory = sorted(notes.values(), key=lambda note: note.path.lower())
        return self._inventory

    def refresh(self) -> None:
        self._inventory = None

    # -- writes -------------------------------------------------------------- #

    def write_new(self, relative: str, text: str) -> ChangeRecord:
        if self.exists(relative):
            raise VaultError(f"refusing to overwrite existing note: {relative}")
        self._write(relative, text)
        self._inventory = None
        return ChangeRecord(
            path=relative,
            action="create",
            added_lines=text.count("\n"),
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
        original = self.read(relative) if self.exists(relative) else None
        if original == new_text:
            return ChangeRecord(path=relative, action="unchanged")
        self._write(relative, new_text)
        if expect_only_additions and original is not None:
            removed = _removed_lines(original, new_text)
            if removed:
                self._write(relative, original)
                raise VaultError(
                    f"refusing non-insertion change to {relative}; removed or changed "
                    f"{len(removed)} pre-existing line(s)"
                )
        self._inventory = None
        return ChangeRecord(
            path=relative,
            action=action,
            added_lines=new_text.count("\n") - (original or "").count("\n"),
            diff=_unified_diff(original or "", new_text, relative),
        )

    # -- lifecycle ----------------------------------------------------------- #

    def discard(self) -> None:
        try:
            shutil.rmtree(self.directory, ignore_errors=True)
        except OSError:
            return

    def _overlay(self, relative: str) -> Path:
        return self.directory / relative.replace("\\", "/").lstrip("/")

    def _write(self, relative: str, text: str) -> None:
        path = self._overlay(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes(path, text.encode("utf-8"))
