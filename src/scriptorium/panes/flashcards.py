"""The Flashcards box: the draft cards produced by the compose stage.

One box, one file.  Cards stay drafts: the pane only displays what the job
wrote to ``cards.json`` and says which writer handled each note.
"""

from __future__ import annotations

from typing import Any

from ..pane_api import Pane, PaneHost
from ..state import Stage
from ..tui_render import STYLE_DIM, STYLE_HEADING, STYLE_SELECTED, one_line

#: Stages that report their progress in this box.
RIGHT_STAGES = {Stage.COMPOSE, Stage.FLASHCARDS}


class FlashcardsPane(Pane):
    """Draft cards, grouped by the note they belong to."""

    name = "flashcards"

    def title(self, host: PaneHost) -> str:
        activity = ""
        if host.running and host.current_stage:
            try:
                stage = Stage(host.current_stage)
            except ValueError:
                stage = None
            if stage in RIGHT_STAGES:
                activity = " · " + host.stage_text()
        return "Flashcards" + activity

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        rows: list[tuple[str, str]] = []
        selected_row = 0
        groups = list(host.card_groups)
        selected = (
            groups[host.card_index] if groups and 0 <= host.card_index < len(groups) else None
        )
        for group in groups:
            is_selected = group is selected
            if is_selected:
                selected_row = len(rows)
            marker = "▾ " if group.expanded else "▸ "
            rows.append((marker + group.title, STYLE_SELECTED if is_selected else "normal"))
            if group.expanded:
                for number, card in enumerate(group.cards, start=1):
                    front = one_line(card.front)
                    rows.append((f"  {number}. {front}", STYLE_HEADING))
                    back = one_line(card.back)
                    if back:
                        rows.append((f"     {back}", STYLE_DIM))
                if group.writer:
                    rows.append(
                        (
                            f"     written by {group.writer}; "
                            f"{group.written} written, {group.skipped} skipped",
                            STYLE_DIM,
                        )
                    )
        if not rows:
            rows.append(("No flashcards yet. They appear after the compose stage.", STYLE_DIM))
        return rows, selected_row

    # -- selection ------------------------------------------------------------ #

    def item_count(self, host: PaneHost) -> int:
        return len(host.card_groups)

    def selection(self, host: PaneHost) -> int:
        return host.card_index

    def set_selection(self, host: PaneHost, index: int) -> None:
        if not host.card_groups:
            host.card_index = 0
            return
        host.card_index = max(0, min(len(host.card_groups) - 1, index))

    def toggle_item(self, host: PaneHost, index: int) -> None:
        groups = host.card_groups
        if not groups or not 0 <= index < len(groups):
            return
        groups[index].expanded = not groups[index].expanded

    def scroll(self, host: PaneHost) -> int:
        return host.card_scroll

    def set_scroll(self, host: PaneHost, value: int) -> None:
        host.card_scroll = max(0, value)

    # -- keys ----------------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        return super().handle_key(host, key, active=active)
