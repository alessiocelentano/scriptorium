"""The Extras box: off-topic material kept out of the notes.

One box, one file.  Extras are read-only inside the vault: they are displayed
here, copied to the clipboard, or written to a path the user explicitly chose.
It is a navigable box in the layout like any other.
"""

from __future__ import annotations

import curses
from typing import Any

from ..extra_notes import format_extra_notes
from ..pane_api import PAGE_DOWN_KEYS, PAGE_UP_KEYS, Pane, PaneHost
from ..tui_render import STYLE_DIM, wrap


class ExtrasPane(Pane):
    """Asides from the extract stage: exam notices, links and curiosities."""

    name = "extras"

    def title(self, host: PaneHost) -> str:
        return f"Extras · {len(host.asides)}"

    def footer(self, host: PaneHost) -> str:
        return "c copy · w write"

    def has_selection(self) -> bool:
        """Extras has no selection: it scrolls a read-only list."""
        return False

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        rows: list[tuple[str, str]] = []
        for aside in host.asides:
            label = f"[{aside.kind}]"
            if aside.source:
                label += f" {aside.source}"
            rows.append((label, STYLE_DIM))
            for line in str(aside.text).splitlines() or [""]:
                rows.append((line, "normal"))
            rows.append(("", "normal"))
        if not rows:
            rows.append(
                (
                    "No extras yet. Off-topic comments, links and curiosities from the "
                    "transcript appear here after the extract stage.",
                    STYLE_DIM,
                )
            )
        return rows, -1

    # -- scroll --------------------------------------------------------------- #

    def display_lines(self, host: PaneHost) -> list[tuple[str, str]]:
        """Rows soft-wrapped to the current width of the box."""
        rect = host.layout_rects().get(self.name)
        width = max(1, (rect.w - 2) if rect is not None else 40)
        lines: list[tuple[str, str]] = []
        for text, style in self.rows(host)[0]:
            for piece in wrap(text, width):
                lines.append((piece, style))
        return lines

    def scroll_limit(self, host: PaneHost) -> int:
        rect = host.layout_rects().get(self.name)
        inner_h = max(1, (rect.h - 2) if rect is not None else 1)
        return max(0, len(self.display_lines(host)) - inner_h)

    def scroll(self, host: PaneHost) -> int:
        return host.extras_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.extras_scroll = max(0, min(self.scroll_limit(host), value))

    def scroll_by(self, host: PaneHost, delta: int) -> None:
        self.set_scroll(host, self.scroll(host) + delta)

    def on_asides(self, host: PaneHost) -> None:
        """Keep the scroll inside the content after the host reloads asides."""
        self.set_scroll(host, self.scroll(host))

    # -- actions -------------------------------------------------------------- #

    def _copy(self, host: PaneHost) -> None:
        """Copy every aside to the system clipboard as Markdown text."""
        if not host.asides:
            host.status = "no extras to copy"
            return
        if host.copy_text(format_extra_notes(host.asides)):
            host.status = f"copied {len(host.asides)} extras to the clipboard"
        else:
            host.status = "could not copy: no clipboard backend available"

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        if key in {"c", "C"}:
            self._copy(host)
            return True
        if key in {"w", "W"}:
            host.open_extras_export()
            return True
        if not active:
            return False
        if key in {"q", "Q"}:
            host.request_quit()
            return True
        if key == "f":
            host.toggle_fullscreen()
            return True
        vim = host.vim_movements
        if key in {curses.KEY_DOWN} or (vim and key == "j"):
            self.scroll_by(host, 1)
            return True
        if key in {curses.KEY_UP} or (vim and key == "k"):
            self.scroll_by(host, -1)
            return True
        if key in PAGE_DOWN_KEYS:
            self.scroll_by(host, self.page(host, host.layout_rects().get(self.name)))
            return True
        if key in PAGE_UP_KEYS:
            self.scroll_by(host, -self.page(host, host.layout_rects().get(self.name)))
            return True
        if key in {"g", curses.KEY_HOME}:
            self.set_scroll(host, 0)
            return True
        if key in {"G", curses.KEY_END}:
            self.set_scroll(host, self.scroll_limit(host))
            return True
        return False
