"""The Changes box: vault writes, diffs and the merge review.

One box, one file.  The pane lists the files the apply stage created, merged or
touched in the MOC, expands a file into its unified diff, and hosts the merge
review: a merge that needs a decision is answered inside this box, never in a
floating window.
"""

from __future__ import annotations

import curses
from typing import Any

from ..pane_api import Pane, PaneHost
from ..state import Stage
from ..tui_layout import PaneRect
from ..tui_render import STYLE_DIM, STYLE_HEADING, STYLE_SELECTED, diff_style, wrap

#: Stages that report their progress in this box.
LEFT_STAGES = {
    Stage.PREPARE,
    Stage.AUDIO,
    Stage.TRANSCRIBE,
    Stage.EXTRACT,
    Stage.OUTLINE,
    Stage.COMPOSE,
    Stage.APPLY,
    Stage.CLEANUP,
}

GROUP_LABELS = (
    ("Created files", "created"),
    ("Added to existing notes", "merged"),
    ("MOC updates", "moc"),
)

#: Merge-review answers: key -> decision forwarded to the pipeline.
MERGE_ANSWERS = {"y": "apply", "n": "skip", "m": "manual", "a": "all", "q": "quit"}


class ChangesPane(Pane):
    """Vault changes, their diffs and the pending merge review."""

    name = "changes"
    owns_merge_review = True

    def title(self, host: PaneHost) -> str:
        activity = ""
        if host.running and host.current_stage:
            try:
                stage = Stage(host.current_stage)
            except ValueError:
                stage = None
            if stage in LEFT_STAGES:
                activity = " · " + host.stage_text()
        return "Changes" + activity

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        rows: list[tuple[str, str]] = []
        selected_row = 0
        entries = list(host.diff_entries)
        selected = (
            entries[host.diff_index] if entries and 0 <= host.diff_index < len(entries) else None
        )
        for label, key in GROUP_LABELS:
            group = [entry for entry in entries if entry.group == key]
            if not group:
                continue
            rows.append((f" {label}", STYLE_HEADING))
            for entry in group:
                is_selected = entry is selected
                if is_selected:
                    selected_row = len(rows)
                marker = "▾ " if entry.expanded else "▸ "
                rows.append((marker + entry.title, STYLE_SELECTED if is_selected else "normal"))
                if entry.expanded:
                    for line in entry.diff_lines:
                        rows.append(("   " + line, diff_style(line)))
        if not rows:
            rows.append(("No vault writes yet. Start a job with Enter in the input.", STYLE_DIM))
        return rows, selected_row

    # -- selection ------------------------------------------------------------ #

    def item_count(self, host: PaneHost) -> int:
        return len(host.diff_entries)

    def selection(self, host: PaneHost) -> int:
        return host.diff_index

    def set_selection(self, host: PaneHost, index: int) -> None:
        if not host.diff_entries:
            host.diff_index = 0
            return
        host.diff_index = max(0, min(len(host.diff_entries) - 1, index))

    def toggle_item(self, host: PaneHost, index: int) -> None:
        entries = host.diff_entries
        if not entries or not 0 <= index < len(entries):
            return
        entries[index].expanded = not entries[index].expanded

    def scroll(self, host: PaneHost) -> int:
        return host.diff_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.diff_scroll = max(0, value)

    # -- drawing -------------------------------------------------------------- #

    def draw(self, host: PaneHost, rect: PaneRect) -> None:
        if host.merge_review is not None:
            self.draw_merge_review(host, rect)
            return
        super().draw(host, rect)

    def draw_merge_review(self, host: PaneHost, rect: PaneRect) -> None:
        """Render a pending merge review inside this box."""
        review = host.merge_review
        if review is None or rect.h < 3 or rect.w < 4:
            return
        painter = host.painter
        y, x, h, w = rect.y, rect.x, rect.h, rect.w
        note = review.payload.get("note")
        preview = str(review.payload.get("preview") or "")
        footer = (
            "y apply · n skip · m manual · a all · q quit · j/k scroll"
            if host.vim_movements
            else "y apply · n skip · m manual · a all · q quit · ↑/↓ scroll"
        )
        painter.draw_box(y, x, h, w, "Review a merge", footer, True, [], 0, 0, active=True)
        path = getattr(note, "path", "")
        anchor = getattr(note, "anchor", None) or "(end of note)"
        inner_w = max(1, w - 4)
        bottom = y + h - 1
        row = y + 1
        for piece in wrap(f"Note: {path}", inner_w):
            if row >= bottom:
                break
            painter.addstr(row, x + 2, piece, painter.attr(STYLE_HEADING))
            row += 1
        for piece in wrap(f"Anchor: {anchor}", inner_w):
            if row >= bottom:
                break
            painter.addstr(row, x + 2, piece, painter.attr(STYLE_DIM))
            row += 1
        preview_lines = preview.rstrip("\n").splitlines() or ["(empty preview)"]
        lines: list[tuple[str, str]] = []
        for line in preview_lines:
            style = diff_style(line) if line.startswith(("+", "-", "@@")) else "normal"
            for piece in wrap(line, inner_w):
                lines.append((piece, style))
        visible = max(0, bottom - row)
        review.scroll = max(0, min(review.scroll, max(0, len(lines) - visible)))
        for offset in range(min(visible, len(lines) - review.scroll)):
            line, style = lines[review.scroll + offset]
            painter.addstr(row + offset, x + 2, line, painter.attr(style))

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        review = host.merge_review
        if review is not None:
            return self._handle_review_key(host, review, key)
        return super().handle_key(host, key, active=active)

    def _handle_review_key(self, host: PaneHost, review: Any, key: Any) -> bool:
        """Answer the pending merge, or scroll its preview."""
        if key == "\x1b":
            host.answer_merge("skip")
            return True
        if isinstance(key, str) and key.lower() in MERGE_ANSWERS:
            host.answer_merge(MERGE_ANSWERS[key.lower()])
            return True
        if self._moves_down(host, key):
            review.scroll += 1
            return True
        if self._moves_up(host, key):
            review.scroll = max(0, review.scroll - 1)
            return True
        return False

    @staticmethod
    def _moves_down(host: PaneHost, key: Any) -> bool:
        return key == curses.KEY_DOWN or (host.vim_movements and key == "j")

    @staticmethod
    def _moves_up(host: PaneHost, key: Any) -> bool:
        return key == curses.KEY_UP or (host.vim_movements and key == "k")
