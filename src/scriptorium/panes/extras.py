"""The Extras box: off-topic material kept out of the notes.

One box, one file.  Extras are read-only inside the vault: they are displayed
here, copied to the clipboard, or written to a path the user explicitly chose.
It is a navigable box in the layout like any other.
"""

from __future__ import annotations

from typing import Any

from ..extra_notes import format_extra_notes
from ..pane_api import PaneHost, PaneRect, ScrollablePane
from ..tui_render import STYLE_DIM, wrap


class ExtrasPane(ScrollablePane):
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

    def display_lines(self, host: PaneHost, rect: PaneRect | None = None) -> list[tuple[str, str]]:
        """Rows soft-wrapped to the current width of the box."""
        lines: list[tuple[str, str]] = []
        for text, style in self.rows(host)[0]:
            lines.extend((piece, style) for piece in wrap(text, self.content_width(host, rect)))
        return lines

    def scroll(self, host: PaneHost) -> int:
        return host.extras_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.extras_scroll = self.clamp_scroll(host, value)

    def on_asides(self, host: PaneHost) -> None:
        """Keep the scroll inside the content after the host reloads asides."""
        self.set_scroll(host, self.scroll(host))

    # -- actions -------------------------------------------------------------- #

    def copy(self, host: PaneHost) -> None:
        """Copy every aside to the system clipboard as Markdown text."""
        if not host.asides:
            host.status = "no extras to copy"
            return
        if host.copy_text(format_extra_notes(host.asides)):
            host.status = f"copied {len(host.asides)} extras to the clipboard"
        else:
            host.status = "could not copy: no clipboard backend available"

    def write(self, host: PaneHost) -> None:
        """Offer the configured roots and a free path to write the extras."""
        host.open_extras_export()

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        if key in {"c", "C"}:
            self.copy(host)
            return True
        if key in {"w", "W"}:
            self.write(host)
            return True
        if super().handle_key(host, key, active=active):
            return True
        if not active:
            return False
        if key in {"q", "Q"}:
            host.request_quit()
            return True
        if key == "f":
            host.toggle_fullscreen()
            return True
        return False
