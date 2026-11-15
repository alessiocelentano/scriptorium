"""The Logs box: process output, warnings and failures.

One box, one file.  ``Ctrl-L`` toggles it (the key comes from the pane's
``spawn_key`` or the ``key`` configured under ``[tui_panes.logs]``).  While it
is hidden its cells are redistributed to the other panes; when shown it is a
navigable pane like the others.  Its lines soft-wrap instead of being clipped
at the right edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..pane_api import PaneHost, PaneRect, ScrollablePane
from ..tui_render import STYLE_DIM, wrap


class LogsPane(ScrollablePane):
    """The inline log viewer."""

    name = "logs"
    spawn_key = "C-l"

    #: True while the view is pinned to the tail: a new log line scrolls it down.
    _follow_tail: bool = True

    def has_selection(self) -> bool:
        """The log viewer scrolls a read-only buffer."""
        return False

    def title(self, host: PaneHost) -> str:
        return f"Logs · {len(host.log_lines)} lines"

    def footer(self, host: PaneHost) -> str:
        return "c copy · w write"

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        return [], -1

    # -- scroll --------------------------------------------------------------- #

    def display_lines(
        self, host: PaneHost, rect: PaneRect | None = None
    ) -> list[tuple[str, str]]:
        """The log text soft-wrapped to the width of the box.

        One message is one entry in ``host.log_lines``, so each message is
        wrapped on its own: its continuation rows stay together and the blank
        separator only ever lands between two messages.
        """
        width = self.content_width(host, rect)
        lines: list[tuple[str, str]] = []
        for index, message in enumerate(host.log_lines or ["(no logs yet)"]):
            if index:
                lines.append(("", STYLE_DIM))
            lines.extend((piece, STYLE_DIM) for piece in wrap(message, width))
        return lines

    def scroll(self, host: PaneHost) -> int:
        return host.log_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.log_scroll = self.clamp_scroll(host, value)
        # Reaching the bottom re-attaches the view to the tail; any offset above
        # it means the user scrolled up and must keep their place.
        self._follow_tail = host.log_scroll >= self.scroll_limit(host)

    def on_shown(self, host: PaneHost) -> None:
        """Open the viewer at its last page, following the tail from there."""
        self.set_scroll(host, self.scroll_limit(host))

    def on_log(self, host: PaneHost) -> None:
        """Follow the new last line unless the reader scrolled up."""
        if self._follow_tail and host.pane_shown(self.name):
            self.set_scroll(host, self.scroll_limit(host))

    # -- copy / write ---------------------------------------------------------- #

    def copy(self, host: PaneHost) -> None:
        """Copy the log buffer to the system clipboard."""
        if not host.log_lines:
            host.status = "no logs to copy"
            return
        if host.copy_text("\n".join(host.log_lines)):
            host.status = f"copied {len(host.log_lines)} log lines to the clipboard"
        else:
            host.status = "could not copy: no clipboard backend available"

    def write(self, host: PaneHost) -> None:
        """Write the log buffer to a deterministic path under the artifacts dir."""
        if not host.log_lines:
            host.status = "no logs to write"
            return
        target = Path(host.config.artifacts_dir) / "logs.txt"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(host.log_lines) + "\n", encoding="utf-8")
        except OSError as exc:
            host.status = f"could not write logs: {exc}"
            return
        host.status = f"wrote {len(host.log_lines)} log lines to {target}"

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        if key in {"c", "C"}:
            self.copy(host)
            return True
        if key in {"w", "W"}:
            self.write(host)
            return True
        if key in {"\x1b", "q", "Q"} and active:
            if key in {"q", "Q"}:
                host.request_quit()
            else:
                host.toggle_pane(self.name)
            return True
        if key == "f":
            host.toggle_fullscreen()
            return True
        return super().handle_key(host, key, active=active)
