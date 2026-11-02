"""Console UI adapter that routes prompts into the TUI event loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import PlannedNote
from .ui import ConsoleUI, MocDecision
from .vault import MocMatch


class TuiConsoleUI(ConsoleUI):
    """ConsoleUI that never touches stdin, stdout, or stderr directly."""

    def __init__(
        self,
        log: Callable[[str], None],
        ask: Callable[[str, dict[str, Any]], Any],
    ):
        super().__init__(interactive=True)
        self._log = log
        self._ask_request = ask

    def print(self, message: str = "") -> None:
        self._log(message)

    def ask_language(self, default: str = "en") -> str:
        answer = self._ask_request("language", {"default": default})
        return str(answer or default)

    def choose_moc(
        self,
        course: str,
        matches: list[MocMatch],
        *,
        default_title: str = "",
    ) -> MocDecision:
        decision = self._ask_request(
            "moc",
            {"course": course, "matches": matches, "default_title": default_title},
        )
        if not isinstance(decision, MocDecision):
            return MocDecision.abort()
        return decision

    def choose_target(self, options: list[Any]) -> Any:
        return self._ask_request("target", {"options": list(options)})

    def review_merge(self, note: PlannedNote, preview: str) -> str:
        return self._ask_request("merge", {"note": note, "preview": preview})
