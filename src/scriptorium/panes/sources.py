"""The Sources box: the files the user collected before starting a job.

One box, one file.  The pane groups the collected paths under the directory
they came from so a long absolute path is read once, as a heading, instead of
being repeated below every file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..pane_api import Pane, PaneHost
from ..tui_render import STYLE_DIM, STYLE_HEADING, STYLE_SELECTED

#: Suffix drawn after a source while the pipeline transcribes it.
_STATUS_MARKS = {"running": "·", "done": "✓", "failed": "✗"}


def _status_mark(host: PaneHost, path: Path) -> str:
    """The transcription mark for one source, or nothing outside a run."""
    state = host.source_status.get(str(path.expanduser()))
    mark = _STATUS_MARKS.get(state or "")
    return f"  {mark}" if mark else ""


class SourcesPane(Pane):
    """Collected sources, grouped by parent directory."""

    name = "sources"

    def title(self, host: PaneHost) -> str:
        return f"Sources · {len(host.source_paths)}"

    def footer(self, host: PaneHost) -> str:
        return "i add · d remove" if host.vim_movements else "d remove"

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        rows: list[tuple[str, str]] = []
        selected_row = 0
        paths: list[Path] = list(host.source_paths)
        selected = (
            paths[host.source_index] if paths and 0 <= host.source_index < len(paths) else None
        )
        groups: dict[str, list[Path]] = {}
        order: list[str] = []
        for path in paths:
            parent = str(path.parent)
            if parent not in groups:
                groups[parent] = []
                order.append(parent)
            groups[parent].append(path)
        for parent in order:
            rows.append((parent, STYLE_DIM))
            for path in groups[parent]:
                is_selected = path is selected
                if is_selected:
                    selected_row = len(rows)
                marker = "▸ " if is_selected else "  "
                rows.append(
                    (
                        marker + path.name + _status_mark(host, path),
                        STYLE_SELECTED if is_selected else "normal",
                    )
                )
        if host.moc_hint is not None:
            rows.append(("MOC hint: " + host.moc_hint.name, STYLE_HEADING))
        if not paths:
            rows.append(("No sources yet. Type a path in the input and press Enter.", STYLE_DIM))
        return rows, selected_row

    # -- selection ------------------------------------------------------------ #

    def item_count(self, host: PaneHost) -> int:
        return len(host.source_paths)

    def selection(self, host: PaneHost) -> int:
        return host.source_index

    def set_selection(self, host: PaneHost, index: int) -> None:
        if not host.source_paths:
            host.source_index = 0
            return
        host.source_index = max(0, min(len(host.source_paths) - 1, index))

    def scroll(self, host: PaneHost) -> int:
        return host.source_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.source_scroll = max(0, value)

    # -- actions -------------------------------------------------------------- #

    def _remove(self, host: PaneHost) -> None:
        """Drop the selected source from the shared list."""
        if not host.source_paths:
            return
        index = max(0, min(host.source_index, len(host.source_paths) - 1))
        removed = host.source_paths.pop(index)
        host.source_index = min(index, max(0, len(host.source_paths) - 1))
        host.source_scroll = min(host.source_scroll, max(0, len(host.source_paths) - 1))
        host.status = f"removed {removed.name}"

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        if key in {"d", "D"}:
            self._remove(host)
            return True
        return super().handle_key(host, key, active=active)
