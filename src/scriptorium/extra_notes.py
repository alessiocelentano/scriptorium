"""Render and store the off-topic material shown in the TUI Extras pane.

The Extras pane is read-only inside the vault: these helpers only format the
asides for the clipboard and write them to a path the user explicitly chose.
An existing file is always appended to, never overwritten.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .models import Aside

DEFAULT_EXPORT_NAME = "extras.md"


def format_extra_notes(asides: Sequence[Aside]) -> str:
    """Return one Markdown list item per aside, preserving multi-line text."""
    blocks: list[str] = []
    for aside in asides:
        text_lines = str(aside.text).splitlines() or [""]
        first = text_lines[0].strip()
        source = f" _{aside.source}_" if aside.source else ""
        block = [f"- **[{aside.kind}]** {first}{source}".rstrip()]
        for line in text_lines[1:]:
            block.append(f"  {line}".rstrip())
        blocks.append("\n".join(block))
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def write_extra_notes(path: Path, asides: Sequence[Aside]) -> bool:
    """Append the formatted asides to ``path``; return True when it existed.

    The parent directory must already exist: a typo in a newly typed path is
    reported instead of silently creating a tree of directories.
    """
    text = format_extra_notes(asides)
    if not text:
        raise ValueError("there are no extras to write")
    existed = path.exists()
    prefix = ""
    if existed:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            prefix = "\n"
        prefix += "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix)
        handle.write(text)
    return existed
