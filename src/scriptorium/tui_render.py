"""Rendering primitives shared by the TUI host and the box modules.

A pane draws through a :class:`Painter` and nothing else: it gets a screen
handle, the colour map, the text helpers and :meth:`Painter.draw_box`.  Keeping
these primitives here (instead of on the application object) is what lets a box
live in its own module without importing ``tui.py``.

Styles are names, not curses attributes: the application resolves a name into a
terminal attribute through the colour map it initialised at start-up.
"""

from __future__ import annotations

import curses
import textwrap
from typing import Any

STYLE_TITLE = "title"
STYLE_ADDED = "added"
STYLE_REMOVED = "removed"
STYLE_HUNK = "hunk"
STYLE_META = "meta"
STYLE_HEADING = "heading"
STYLE_DIM = "dim"
STYLE_SELECTED = "selected"
STYLE_ACTIVE = "active"
STYLE_ACTIVE_BORDER = "active_border"
STYLE_FOCUS = "focus"
STYLE_MODE_STANDARD = "mode_standard"
STYLE_MODE_NORMAL = "mode_normal"
STYLE_MODE_INSERT = "mode_insert"
# Status colours for the wizard review page: a green tick, a yellow warning
# sign and a red cross, each used for the glyph and its explanation only.
STYLE_OK = "ok"
STYLE_WARN = "warn"
STYLE_ERROR = "error"

#: One row of pane content: the text and the style name used to draw it.
Row = tuple[str, str]


def one_line(text: Any) -> str:
    """Collapse a value into one display line."""
    return " ".join(str(text).replace("\t", "    ").splitlines()).strip()


def clip(text: str, width: int) -> str:
    """Shorten ``text`` to ``width`` cells, marking the cut with an ellipsis."""
    if width <= 0:
        return ""
    text = one_line(text)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def wrap(text: str, width: int) -> list[str]:
    """Split one logical line into display rows no wider than ``width``.

    Words stay whole while they fit; a token longer than the available width
    is hard-broken so no content is lost off the right edge.
    """
    if width <= 0:
        return [""]
    return textwrap.wrap(
        one_line(text),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def diff_style(line: str) -> str:
    """Pick the style for one line of a unified diff."""
    if line.startswith(("+++", "---", "diff ", "index ")):
        return STYLE_META
    if line.startswith("@@"):
        return STYLE_HUNK
    if line.startswith("+"):
        return STYLE_ADDED
    if line.startswith("-"):
        return STYLE_REMOVED
    return STYLE_DIM


class Painter:
    """Draw text and boxes on one curses screen.

    ``colors`` is the map filled by the application at start-up (style name ->
    curses attribute); it is read by reference so a test can install its own
    attributes without rebuilding the painter.
    """

    def __init__(self, screen: Any, colors: dict[str, int]):
        self.screen = screen
        self.colors = colors

    # -- attributes ---------------------------------------------------------- #

    def attr(self, style: str) -> int:
        """Resolve a style name into a curses attribute."""
        if style == STYLE_SELECTED:
            return self.attr(STYLE_TITLE) | curses.A_REVERSE | curses.A_BOLD
        if style == STYLE_ACTIVE:
            return self.colors.get(STYLE_ACTIVE, curses.A_BOLD) | curses.A_BOLD
        if style == STYLE_FOCUS:
            return self.colors.get(STYLE_FOCUS, curses.A_BOLD) | curses.A_BOLD
        if style == STYLE_TITLE:
            return self.colors.get(STYLE_TITLE, curses.A_BOLD)
        if style == STYLE_DIM:
            return self.colors.get(STYLE_DIM, curses.A_DIM) | curses.A_DIM
        return self.colors.get(style, 0)

    # -- output -------------------------------------------------------------- #

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        """Write one line, clipped to the terminal instead of raising."""
        try:
            rows, cols = self.screen.getmaxyx()
        except (curses.error, AttributeError):
            return
        if y < 0 or y >= rows or x < 0 or x >= cols or not text:
            return
        available = cols - x - 1
        if available <= 0:
            return
        try:
            self.screen.addnstr(y, x, one_line(text), available, attr)
        except curses.error:
            pass

    def addnstr_raw(self, y: int, x: int, text: str, width: int, attr: int = 0) -> None:
        """Write exactly ``width`` cells, keeping whitespace.

        ``addstr`` normalises its text and would drop a run of spaces, which a
        filled chip (the mode badge) needs to stay visible.
        """
        try:
            self.screen.addnstr(y, x, text, width, attr)
        except (curses.error, AttributeError):
            pass

    def fill_rect(self, y: int, x: int, h: int, w: int) -> None:
        """Paint an opaque background over ``h`` rows before a floating box.

        The blank run is written directly: ``addstr`` normalises its text and
        would drop a whitespace-only string, leaving the panes visible through
        the window.
        """
        try:
            rows, cols = self.screen.getmaxyx()
        except (curses.error, AttributeError):
            return
        available = min(w, cols - x - 1)
        if available <= 0:
            return
        blank = " " * available
        for offset in range(max(0, h)):
            row = y + offset
            if row < 0 or row >= rows:
                continue
            try:
                self.screen.addnstr(row, x, blank, available, 0)
            except curses.error:
                pass

    def draw_box(
        self,
        y: int,
        x: int,
        h: int,
        w: int,
        title: str,
        footer: str,
        focused: bool,
        rows: list[Row],
        scroll: int,
        selected_row: int,
        active: bool = False,
    ) -> int:
        """Draw a bordered box and return the scroll offset actually used.

        Long rows wrap instead of being clipped; ``scroll`` is a display-row
        offset and the selected logical row is kept inside the viewport, so the
        caller can store the returned value as its new scroll position.
        """
        if h < 3 or w < 4:
            return scroll
        if active:
            border_attr = self.colors.get(STYLE_ACTIVE_BORDER, curses.A_BOLD)
        elif focused:
            border_attr = self.attr(STYLE_FOCUS) | curses.A_BOLD
        else:
            border_attr = self.attr(STYLE_DIM)
        # Focused and selected boxes use a double outline so their state reads
        # even on terminals where the colour pair is barely distinguishable.
        heavy = active or focused
        top_left, top_right = ("╔", "╗") if heavy else ("┌", "┐")
        bottom_left, bottom_right = ("╚", "╝") if heavy else ("└", "┘")
        horizontal, vertical = ("═", "║") if heavy else ("─", "│")
        self.addstr(y, x, top_left + horizontal * (w - 2) + top_right, border_attr)
        self.addstr(y, x + 2, " " + clip(title, w - 6) + " ", border_attr | curses.A_BOLD)
        for offset in range(1, h - 1):
            self.addstr(y + offset, x, vertical, border_attr)
            self.addstr(y + offset, x + w - 1, vertical, border_attr)
        self.addstr(y + h - 1, x, bottom_left + horizontal * (w - 2) + bottom_right, border_attr)
        if footer:
            self.addstr(
                y + h - 1,
                x + 2,
                " " + clip(footer, w - 6) + " ",
                border_attr | curses.A_DIM,
            )

        inner_h = max(1, h - 2)
        inner_w = max(1, w - 2)
        display: list[tuple[str, str, int]] = []
        for index, (text, style) in enumerate(rows):
            for piece in wrap(text, inner_w):
                display.append((piece, style, index))
        selected_display: int | None = None
        for position, (_, _, index) in enumerate(display):
            if index == selected_row:
                selected_display = position
                break
        # A read-only pane passes -1: it has no selection to follow, so the
        # stored scroll offset is honoured instead of snapping back to the top.
        if selected_display is not None:
            if selected_display < scroll:
                scroll = selected_display
            elif selected_display >= scroll + inner_h:
                scroll = selected_display - inner_h + 1
        scroll = max(0, min(scroll, max(0, len(display) - 1)))
        for offset in range(inner_h):
            index = scroll + offset
            if index >= len(display):
                break
            text, style, _ = display[index]
            if active and style == STYLE_SELECTED:
                style = STYLE_ACTIVE
            self.addstr(y + 1 + offset, x + 1, text, self.attr(style))
        return scroll
