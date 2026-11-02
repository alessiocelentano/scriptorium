"""The Input box: one source path at a time, plus the slash commands.

One box, one file.  The line is a Vim editor (``movement = "vim"``) or a
modeless field (``movement = "standard"``); the pane owns the editor, the
completion popup anchor and the mode badge, and calls back into the host for
the actions that cross boxes (add a source, move the focus, open settings,
refresh the suggestions, list the commands).

The popup follows the typed text: every edit opens or filters the candidates,
so completion never needs a Tab keystroke.  Typing ``/`` shows the slash
commands; typing a path shows the matching source files.  ``Enter`` on a
candidate adds that source to the sidebar and clears the line, and ``Enter``
on a command runs it.
"""

from __future__ import annotations

import curses
from typing import Any

from ..pane_api import ENTER_KEYS, MODE_INSERT, MODE_NORMAL, Pane, PaneHost
from ..tui_render import (
    STYLE_DIM,
    STYLE_MODE_INSERT,
    STYLE_MODE_NORMAL,
    STYLE_MODE_STANDARD,
    STYLE_SELECTED,
    clip,
)

INPUT_PROMPT = "› "
PLACEHOLDER = "add sources and type /run"


class InputPane(Pane):
    """The single-line input: source paths, slash commands and completion."""

    name = "input"
    is_editor = True

    def title(self, host: PaneHost) -> str:
        return "Input"

    def footer(self, host: PaneHost) -> str:
        return ""

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        return [], -1

    def has_selection(self) -> bool:
        """The input line is edited, not navigated."""
        return False

    # -- geometry ------------------------------------------------------------- #

    def geometry(self, rect_y: int, rect_x: int, rect_w: int, rect_h: int) -> tuple[int, int, int]:
        """Return the prompt row, prompt column and editable width of the box."""
        return rect_y + 1, rect_x + 2, max(1, rect_w - 6)

    def scroll_offset(self, host: PaneHost, width: int) -> int:
        """First visible character index, so the cursor stays in view."""
        if host.cursor <= width - 1:
            return 0
        return host.cursor - width + 1

    def place_cursor(self, host: PaneHost, y: int, x: int, w: int, h: int) -> None:
        """Put the hardware cursor on the insertion point."""
        prompt_y, prompt_x, width = self.geometry(y, x, w, h)
        start = self.scroll_offset(host, width)
        column = prompt_x + len(INPUT_PROMPT) + max(0, min(width - 1, host.cursor - start))
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            host.painter.screen.move(prompt_y, column)
        except (curses.error, AttributeError):
            pass

    # -- drawing -------------------------------------------------------------- #

    def draw(self, host: PaneHost, rect) -> None:  # noqa: ANN001 - PaneRect
        if rect.h < 3 or rect.w < 4:
            return
        painter = host.painter
        focused = host.focus == self.name
        prompt_y, prompt_x, width = self.geometry(rect.y, rect.x, rect.w, rect.h)
        start = self.scroll_offset(host, width)
        active = (
            focused and host.active == self.name and host.settings is None and host.modal is None
        )
        title = self.title(host)
        painter.draw_box(
            rect.y,
            rect.x,
            rect.h,
            rect.w,
            title,
            self.footer(host),
            focused,
            [],
            0,
            0,
            active,
        )
        # The title is drawn by the box with the same colours as its border;
        # the mode chip follows it so the label reads before the mode.
        self.draw_mode_badge(host, rect, len(title))
        visible = host.input_text[start : start + width]
        # Input text is user content: keep it in the terminal's default colour
        # instead of tinting it like chrome.
        if host.input_text:
            painter.addstr(prompt_y, prompt_x, INPUT_PROMPT + visible, 0)
        else:
            painter.addstr(prompt_y, prompt_x, INPUT_PROMPT, 0)
            painter.addstr(
                prompt_y,
                prompt_x + len(INPUT_PROMPT),
                clip(PLACEHOLDER, width),
                painter.attr(STYLE_DIM),
            )

        if not focused or not active:
            return
        if host.mode == MODE_NORMAL and host.vim_movements:
            # Vim normal mode: a block cursor sits on the character (or on a
            # blank cell at the end of the line), never between characters.
            index = max(0, host.cursor - start)
            baseline = visible if host.input_text else PLACEHOLDER
            char = baseline[index] if index < len(baseline) else " "
            painter.addstr(
                prompt_y,
                prompt_x + len(INPUT_PROMPT) + index,
                char,
                painter.attr(STYLE_SELECTED) | curses.A_REVERSE,
            )
        else:
            try:
                curses.curs_set(1)
            except curses.error:
                pass

    def draw_mode_badge(self, host: PaneHost, rect, title_width: int) -> None:  # noqa: ANN001
        """Draw the filled NORMAL/INSERT chip just after the input title."""
        if not host.vim_movements:
            label, style = "STANDARD", STYLE_MODE_STANDARD
        elif host.mode == MODE_INSERT:
            label, style = "INSERT", STYLE_MODE_INSERT
        else:
            label, style = "NORMAL", STYLE_MODE_NORMAL
        painter = host.painter
        start = rect.x + 2 + title_width + 2
        limit = rect.x + rect.w - 2
        # One blank cell on each side keeps the label off the chip edges.
        width = len(label) + 2
        if start + width > limit:
            return
        attr = painter.attr(style) | curses.A_BOLD
        if style not in painter.colors:
            # Terminals without colour still need a visible button.
            attr |= curses.A_REVERSE
        # The chip is painted with a direct write because the normal writer
        # strips whitespace padding.
        painter.addnstr_raw(rect.y, start, " " * width, width, attr)
        painter.addstr(rect.y, start + 1, label, attr)

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool = True) -> bool:
        """Handle a key while the input line is selected."""
        vim = host.vim_movements
        if vim and key in {
            curses.KEY_UP,
            curses.KEY_DOWN,
            curses.KEY_LEFT,
            curses.KEY_RIGHT,
        }:
            # Vim movement uses h/j/k/l; the arrow keys never reach the editor.
            return True
        popup = host.completion
        if popup is not None:
            if key == "\t" or (not vim and key == curses.KEY_DOWN):
                host.cycle_completion(1)
                return True
            if not vim and key == curses.KEY_UP:
                host.cycle_completion(-1)
                return True
            if key in ENTER_KEYS:
                if popup.is_command:
                    host.accept_command_completion()
                else:
                    host.accept_source_completion()
                return True
            if key == "\x1b":
                host.completion = None
                if vim:
                    host.sync_editor_from_fields()
                    host.editor.exit_insert()
                    host.pull_editor()
                return True
            host.completion = None

        host.sync_editor_from_fields()

        if key == "\x1b" and not vim:
            # Standard mode has no normal mode: Escape never leaves the field.
            return True

        if key == "\t":
            if host.editor.mode == MODE_NORMAL:
                host.editor.enter_insert(host.editor.cursor)
                host.pull_editor()
            host.start_completion()
            return True
        if key in ENTER_KEYS:
            host.submit_input()
            return True

        if not vim:
            if key == curses.KEY_UP:
                host.move_focus("k")
                host.pull_editor()
                return True
            if key == curses.KEY_DOWN:
                host.move_focus("j")
                host.pull_editor()
                return True

        # Vim-normal keys that belong to the application, not the line editor.
        if vim and host.editor.mode == MODE_NORMAL:
            if key in {"q", "Q"}:
                host.request_quit()
                return True
            if key == "s":
                host.open_settings()
                return True
            if key == "?":
                host.show_help()
                return True
            if key in {"j", "k"}:
                host.move_focus(str(key))
                host.pull_editor()
                return True

        host.editor.handle_key(key)
        host.pull_editor()
        host.refresh_completion()
        return True
