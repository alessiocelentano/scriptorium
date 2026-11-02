"""The input line's slash commands and their one-line descriptions.

The command list is data, not a branch in the input box: the completion popup
reads it, and adding a command means adding one entry here plus its action in
``TuiApp.submit_input``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    """One command typed in the input line."""

    name: str
    description: str

    @property
    def path(self) -> str:
        """The text inserted into the input line (the completion contract)."""
        return self.name

    @property
    def display(self) -> str:
        """The row shown in the completion popup."""
        return f"{self.name:<9}  {self.description}"


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/run", "start the pipeline with the collected sources"),
    SlashCommand("/new", "clear the current session"),
    SlashCommand("/history", "browse completed runs and restore one"),
    SlashCommand("/wizard", "reopen the first-run setup wizard"),
)


def matching_commands(text: str) -> list[SlashCommand]:
    """Commands whose name starts with the typed prefix, in list order."""
    prefix = text.strip()
    if not prefix.startswith("/"):
        return []
    return [command for command in SLASH_COMMANDS if command.name.startswith(prefix)]
