"""The Report box: the run summary written by the report stage.

One box, one file.  It is a read-only navigable pane like the others and shows
``report.md`` from the current job (summary, warnings, TODOs and suggested
commands).  Its lines soft-wrap instead of being clipped at the right edge.
"""

from __future__ import annotations

import curses
from typing import Any

from ..pane_api import PAGE_DOWN_KEYS, PAGE_UP_KEYS, Pane, PaneHost, PaneRect
from ..tui_render import STYLE_DIM, wrap

PLACEHOLDER = "No report yet. The run report appears here after the report stage."


class ReportPane(Pane):
    """The read-only run report viewer."""

    name = "report"

    def has_selection(self) -> bool:
        """The report viewer scrolls a read-only buffer."""
        return False

    def title(self, host: PaneHost) -> str:
        # ``host.report_lines`` is loaded from the job's ``report.md``.
        return f"Report · {len(host.report_lines)} lines"

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        return [], -1

    # -- scroll --------------------------------------------------------------- #

    def display_lines(self, host: PaneHost) -> list[str]:
        """The report text soft-wrapped to the width of the box."""
        rect = host.layout_rects().get(self.name)
        width = max(1, (rect.w - 4) if rect is not None else 40)
        source = host.report_lines or [PLACEHOLDER]
        lines: list[str] = []
        for line in source:
            lines.extend(wrap(line, width))
        return lines

    def scroll_limit(self, host: PaneHost) -> int:
        return max(0, len(self.display_lines(host)) - 1)

    def scroll(self, host: PaneHost) -> int:
        return host.report_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.report_scroll = max(0, min(self.scroll_limit(host), value))

    def scroll_by(self, host: PaneHost, delta: int) -> None:
        self.set_scroll(host, self.scroll(host) + delta)

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
        if key in {"q", "Q"} and active:
            host.request_quit()
            return True
        if key == "f" and active:
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
