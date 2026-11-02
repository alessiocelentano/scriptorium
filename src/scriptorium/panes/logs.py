"""The Logs box: process output, warnings and failures.

One box, one file.  ``Ctrl-L`` toggles it (the key comes from the pane's
``spawn_key`` or the ``key`` configured under ``[tui_panes.logs]``).  While it
is hidden its cells are redistributed to the other panes; when shown it is a
navigable pane like the others.  Its lines soft-wrap instead of being clipped
at the right edge.
"""

from __future__ import annotations

import curses
from typing import Any

from ..pane_api import ENTER_KEYS, PAGE_DOWN_KEYS, PAGE_UP_KEYS, Pane, PaneHost
from ..tui_layout import PaneRect
from ..tui_render import STYLE_DIM, wrap


class LogsPane(Pane):
    """The inline log viewer."""

    name = "logs"
    spawn_key = "C-l"

    def has_selection(self) -> bool:
        """The log viewer scrolls a read-only buffer."""
        return False

    def title(self, host: PaneHost) -> str:
        return f"Logs · {len(host.log_lines)} lines"

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        return [], -1

    # -- scroll --------------------------------------------------------------- #

    def display_lines(self, host: PaneHost) -> list[str]:
        """The log text soft-wrapped to the width of the box."""
        rect = host.layout_rects().get(self.name)
        width = max(1, (rect.w - 4) if rect is not None else 40)
        lines: list[str] = []
        for line in host.log_lines or ["(no logs yet)"]:
            lines.extend(wrap(line, width))
        return lines

    def scroll_limit(self, host: PaneHost) -> int:
        return max(0, len(self.display_lines(host)) - 1)

    def scroll(self, host: PaneHost) -> int:
        return host.log_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.log_scroll = max(0, min(self.scroll_limit(host), value))

    def scroll_by(self, host: PaneHost, delta: int) -> None:
        self.set_scroll(host, self.scroll(host) + delta)

    def on_shown(self, host: PaneHost) -> None:
        """Open the viewer at its last page."""
        self.set_scroll(host, self.scroll_limit(host))

    def on_log(self, host: PaneHost) -> None:
        """Keep the scroll inside the content after new log lines arrive."""
        if host.pane_shown(self.name):
            self.set_scroll(host, self.scroll(host))

    # -- drawing -------------------------------------------------------------- #

    def draw(self, host: PaneHost, rect: PaneRect) -> None:
        if rect.h < 3 or rect.w < 4:
            return
        painter = host.painter
        painter.draw_box(
            rect.y,
            rect.x,
            rect.h,
            rect.w,
            self.title(host),
            self.footer(host),
            host.focus == self.name,
            [],
            0,
            0,
            active=host.active == self.name,
        )
        lines = self.display_lines(host)
        visible = max(1, rect.h - 3)
        self.set_scroll(host, min(self.scroll(host), max(0, len(lines) - visible)))
        for offset in range(visible):
            index = self.scroll(host) + offset
            if index >= len(lines):
                break
            painter.addstr(rect.y + 1 + offset, rect.x + 2, lines[index], painter.attr(STYLE_DIM))

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        if key in ENTER_KEYS and not active:
            # Enter is the host's "select this box" key.
            return False
        if key in {"\x1b", "q", "Q"} and active:
            if key in {"q", "Q"}:
                host.request_quit()
            else:
                host.toggle_pane(self.name)
            return True
        if key == "f":
            host.toggle_fullscreen()
            return True
        vim = host.vim_movements
        if key == "g" or key == curses.KEY_HOME:
            if vim or active:
                self.set_scroll(host, 0)
                return True
            return False
        if key == "G" or key == curses.KEY_END:
            if vim or active:
                self.set_scroll(host, self.scroll_limit(host))
                return True
            return False
        rect = host.layout_rects().get(self.name)
        if key in PAGE_DOWN_KEYS:
            self.scroll_by(host, max(1, self.page(host, rect)))
            return True
        if key in PAGE_UP_KEYS:
            self.scroll_by(host, -max(1, self.page(host, rect)))
            return True
        if key == curses.KEY_DOWN:
            # In Vim mode the arrows are free (h/j/k/l move the focus), so they
            # scroll; in standard mode the host uses them to move the focus.
            if active or vim:
                self.scroll_by(host, 1)
                return True
            return False
        if key == curses.KEY_UP:
            if active or vim:
                self.scroll_by(host, -1)
                return True
            return False
        if active and key == "j":
            self.scroll_by(host, 1)
            return True
        if active and key == "k":
            self.scroll_by(host, -1)
            return True
        return False
