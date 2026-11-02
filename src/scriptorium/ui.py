"""Interactive console prompts, shared by the CLI and the TUI adapter."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TextIO

from .models import PlannedNote
from .targeting import NoteTarget
from .vault import MocMatch

# Below this score a MOC match is only a suggestion, not a decision.
MOC_CONFIDENCE = 0.55


def moc_title(value: str, fallback: str = "") -> str:
    """Normalize a user supplied MOC name so it always ends with ``MOC``."""
    title = re.sub(r"\s+", " ", (value or "").strip())
    title = re.sub(r"[\\/:*?\"<>|]", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" .")
    if not title:
        title = re.sub(r"\s+", " ", (fallback or "").strip()) or "Course"
    if re.search(r"\bmoc\s*$", title, flags=re.IGNORECASE):
        return re.sub(r"\bmoc\s*$", "MOC", title, flags=re.IGNORECASE)
    return f"{title} MOC"


@dataclass(frozen=True)
class MocDecision:
    """What to do about the course MOC: use one, create one, or abort."""

    action: str  # "use" | "create" | "abort"
    path: str | None = None
    title: str = ""

    @classmethod
    def use(cls, path: str) -> MocDecision:
        return cls(action="use", path=path)

    @classmethod
    def create(cls, title: str) -> MocDecision:
        return cls(action="create", title=title)

    @classmethod
    def abort(cls) -> MocDecision:
        return cls(action="abort")


class ConsoleUI:
    def __init__(
        self,
        *,
        yes: bool = False,
        accept_merges: bool = False,
        interactive: bool | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ):
        self.yes = yes
        self.accept_merges = accept_merges
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self.interactive = (
            interactive
            if interactive is not None
            else (self._stdin.isatty() and self._stdout.isatty())
        )

    # -- generic ------------------------------------------------------------- #

    def print(self, message: str = "") -> None:
        print(message, file=self._stdout, flush=True)

    def confirm(self, question: str, *, default: bool = False) -> bool:
        if self.yes:
            return True
        if not self.interactive:
            return default
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            answer = self._ask(f"{question} {suffix} ")
            if not answer:
                return default
            if answer.lower() in {"y", "yes"}:
                return True
            if answer.lower() in {"n", "no"}:
                return False
            self.print("Please answer y or n.")

    # -- transcription ------------------------------------------------------- #

    def ask_language(self, default: str = "en") -> str:
        """Ask for the spoken language once per run; "detect" is accepted."""
        if self.yes or not self.interactive:
            return default
        while True:
            answer = self._ask(f"Spoken language (code, or 'detect') [{default}]: ").strip()
            if not answer:
                return default
            return answer.lower()

    # -- MOC selection ------------------------------------------------------- #

    def choose_moc(
        self,
        course: str,
        matches: list[MocMatch],
        *,
        default_title: str = "",
    ) -> MocDecision:
        """Pick the MOC for this run, optionally creating a new one."""
        candidates = list(matches)
        top = candidates[0] if candidates else None
        confident = top is not None and top.score >= MOC_CONFIDENCE
        fallback = moc_title(default_title or f"{course} MOC")

        if self.yes or not self.interactive:
            if confident and top is not None:
                return MocDecision.use(top.path)
            return MocDecision.create(fallback)

        self.print(f"Course: {course}")
        if candidates:
            self.print("Candidate MOCs:")
            for index, match in enumerate(candidates, start=1):
                self.print(f"  {index}. {match.path}  (score {match.score:.2f})")
        else:
            self.print("No MOC matched this material yet.")
        self.print(f"  c. create a new MOC ({fallback})")
        self.print("  q. abort")

        while True:
            answer = self._ask(
                f"Choose 1-{len(candidates)} [1], c to create, q to abort: "
            ).strip()
            lowered = answer.lower()
            if lowered == "q":
                return MocDecision.abort()
            if lowered == "c":
                title = self._ask_moc_title(fallback)
                if title is None:
                    continue
                return MocDecision.create(title)
            if not answer and confident and top is not None:
                return MocDecision.use(top.path)
            if not answer and not candidates:
                return MocDecision.create(fallback)
            if answer.isdigit() and 1 <= int(answer) <= len(candidates):
                return MocDecision.use(candidates[int(answer) - 1].path)
            self.print("Invalid choice.")

    def _ask_moc_title(self, fallback: str) -> str | None:
        """Ask for a MOC name; the final ``MOC`` suffix is always shown."""
        while True:
            answer = self._ask(f"MOC name [{fallback}], or q to go back: ").strip()
            if answer.lower() == "q":
                return None
            title = moc_title(answer, fallback)
            self.print(f"New MOC: {title}")
            confirm = self._ask("Create it? [Y/n] ").strip().lower()
            if confirm in {"", "y", "yes"}:
                return title
            if confirm in {"n", "no"}:
                continue
            self.print("Please answer y or n.")

    # -- note target --------------------------------------------------------- #

    def choose_target(self, options: list[NoteTarget]) -> NoteTarget | None:
        """Pick where a planned note goes; ``None`` means "keep a new note"."""
        if self.yes or not self.interactive:
            return None
        title = next((option.title for option in options if option.kind == "create"), "")
        self.print("")
        self.print(f"Where should '{title}' go?")
        for index, option in enumerate(options, start=1):
            score = f"  (score {option.score:.2f})" if option.score else ""
            self.print(f"  {index}. {option.label}{score}")
        self.print("  q. keep this as a new note without asking again")
        while True:
            answer = self._ask(f"Choose 1-{len(options)} [q]: ").strip().lower()
            if answer in {"q", ""}:
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                chosen = options[int(answer) - 1]
                if isinstance(chosen, NoteTarget):
                    return chosen
                return None
            self.print("Invalid choice.")

    # -- merge review -------------------------------------------------------- #

    def review_merge(self, note: PlannedNote, preview: str) -> str:
        """Return one of: apply, skip, manual, all, quit.

        ``preview`` is the unified diff the merge would write.  A merge may
        change pre-existing lines, so the diff, not the new body, is what the
        user approves; ``manual`` writes a git-style conflict file outside the
        vault and leaves the note untouched.
        """
        if self.accept_merges:
            return "apply"
        if self.yes or not self.interactive:
            return "skip"

        self.print("")
        self.print(f"Merge into existing note: {note.path}")
        self.print(f"Anchor: {note.anchor or '(end of note)'}")
        self.print(f"Change: {note.action}")
        self.print("Proposed diff:")
        self.print("-" * 60)
        for line in preview.strip().splitlines()[:40]:
            self.print(f"  {line}")
        if len(preview.strip().splitlines()) > 40:
            self.print("  [...]")
        self.print("-" * 60)
        while True:
            answer = self._ask(
                "Apply this merge? [y]es / [n]o / [m]anual conflict file / [a]ll / [q]uit: "
            ).strip().lower()
            if answer in {"y", "yes"}:
                return "apply"
            if answer in {"n", "no", ""}:
                return "skip"
            if answer in {"m", "manual"}:
                return "manual"
            if answer in {"a", "all"}:
                return "all"
            if answer in {"q", "quit"}:
                return "quit"
            self.print("Invalid choice.")

    # -- internals ----------------------------------------------------------- #

    def _ask(self, prompt: str) -> str:
        self._stdout.write(prompt)
        self._stdout.flush()
        line = self._stdin.readline()
        if not line:
            return ""
        return line.rstrip("\n")
