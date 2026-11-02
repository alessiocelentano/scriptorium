"""Curses terminal user interface for scriptorium.

Layout::

    +-----------+---------------------------+----------------------+
    | Sources   | Changes                   | Flashcards           |
    |           |                           |                      |
    +-----------+                           |                      |
    | Extras    |                           |                      |
    +-----------+---------------------------+----------------------+
    | Input                                                        |
    +--------------------------------------------------------------+

The Extras pane holds off-topic material from the transcript (exam and course
comments, university logistics, links, curiosities); it is never written to the
notes root.  The input line adds one source path at a time to the Sources pane (recordings
and/or documents: slides, notes, PDFs, spreadsheets).  ``Enter`` clears the
line and appends the path; ``/run`` starts the pipeline with every collected
source, and ``/new`` clears the current session so the next job starts from an
empty screen.  ``/history`` opens the completed-run window.  Typing a path
opens a popup with the matching source files as soon as the first character is
typed; ``Enter`` adds the highlighted one and clears the line, so completion
never needs a ``Tab``.  Typing ``/`` opens a popup with the commands and a
one-line description of each, and ``c`` opens the full command list.  The MOC
is selected in a later phase, not typed here.

The layout is a tree of splits and `tui_layout` in `config.toml` is its only
source of truth: `a | b` places two boxes side by side, `a / b` stacks them,
parentheses nest, and `name:N` sets a box's weight inside the enclosing split.
The input line always keeps three terminal rows.  The inline log viewer is an
ordinary pane toggled by its spawn key (Ctrl-L by default).

``movement = "vim"`` keeps modal editing: ``h``/``l`` move focus left/right,
``j``/``k`` move it up/down, and inside the selected input line ``i`` enters
insert mode while ``Esc`` returns to normal mode.  ``movement = "standard"``
removes the modes: once the input line is selected it is always editable,
``←``/``→`` move its cursor and ``↑``/``↓`` move focus.  Arrow keys are disabled
in Vim mode, ``h``/``j``/``k``/``l`` are typed in standard mode.  ``Enter`` adds
the typed path to Sources, ``/run``
starts the pipeline, ``/new`` resets the session, ``d`` removes the selected
source, ``f`` expands the
focused pane over every box except the input line (and restores the layout on
a second press), ``Ctrl-L`` toggles the
log window without moving focus, ``c`` lists the available commands, ``s``
edits settings, and ``q`` quits.

``Enter`` on a focused box selects it (its border turns green) so the movement
keys walk that box's options: ``j``/``k`` in Vim mode, ``↑``/``↓`` in standard
mode.  A second ``Enter`` toggles the selected entry, and ``Esc`` leaves the box
and returns to spatial focus navigation.  The same rule applies to the input
line: it must be selected with ``Enter`` (or with ``i`` from another box, which
focuses it and enters insert mode) before it accepts text.  Inside the selected
input line ``Esc`` leaves insert mode and keeps the box selected, so Vim normal
commands keep editing it; a second ``Esc`` in normal mode leaves the box.
Submitting a source with ``Enter`` clears the line, leaves the box and returns
it to normal mode.  The Extras pane can be selected too: ``c`` copies every
aside to the system clipboard, and ``w`` opens a floating window that writes
them to a file.  The window offers the search paths from the settings
as destinations and one ``new path`` entry whose field completes files and
directories; an existing file is appended to, never overwritten.  A merge that
needs review is shown inside the Changes box, never in a
floating window.  Focused boxes get a white border while a box selected with
``Enter`` gets a green one, and the input box carries a coloured badge with the
current Vim mode (``NORMAL`` or ``INSERT``).  The
log window is a navigable pane while it is visible, and the header row shows the
current status, with an animated spinner at the far left while a job runs.  A
failure is written to the log (pane and job log file) and the header keeps a
short pointer to it, so a long exception never takes over the interface.  The
pane layout is read from
``tui_layout`` in ``config.toml``; navigation always follows the geometry, not a
fixed cyclic order.

Process output (stdout/stderr) is captured and shown only in the log window, so
library progress bars never paint over the interface.  Content in the panes,
the log window, the merge review and every floating window soft-wraps at the
view's inner width instead of being clipped at the right edge, so long paths,
diff lines and option labels stay readable.  Only single-row chrome (the
header, box titles and the footers that list the available keystrokes) is
clipped.
"""

from __future__ import annotations

import curses
import os
import queue
import shlex
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import wizard
from .clipboard import copy_to_clipboard
from .config import (
    MOVEMENT_VIM,
    Config,
    llm_api_key,
    mark_setup_complete,
    write_config_values,
)
from .errors import ConfigurationError, ScriptoriumError, VaultError
from .extra_notes import DEFAULT_EXPORT_NAME, write_extra_notes
from .history import HistoryEntry, HistoryStore
from .log import Logger
from .models import Aside
from .pane_api import MODE_INSERT, MODE_NORMAL, Pane
from .pane_registry import PaneRegistry, build_registry
from .pipeline import Pipeline, RunOptions
from .settings import (
    ALL_FIELDS,
    KIND_CHOICE,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_INT,
    KIND_LIST,
    KIND_PASSWORD,
    KIND_PATH,
    PATH_DIR,
    PATH_FILE,
    SECTIONS,
    SettingField,
    SettingsRow,
    cycle_choice,
    section_by_key,
)
from .slash_commands import SLASH_COMMANDS, SlashCommand, matching_commands
from .sources import clean_course_name
from .state import STAGE_LABELS, Stage
from .tui_completion import (
    FileMatch,
    complete_directories,
    complete_files,
    complete_note_paths,
    complete_paths,
    replace_token,
)
from .tui_data import CardGroup, DiffEntry, load_asides, load_card_groups, load_diff_entries
from .tui_input import missing_paths, parse_input
from .tui_layout import (
    PANE_INPUT,
    PaneRect,
    clamp_point,
    content_area,
    edge_point,
    layout_panes,
    layout_text,
    neighbor_rects,
    pane_rects,
    split_rect,
)
from .tui_prompt import TuiConsoleUI
from .tui_render import (  # noqa: F401 - styles are re-exported for tests
    STYLE_ACTIVE,
    STYLE_ACTIVE_BORDER,
    STYLE_ADDED,
    STYLE_DIM,
    STYLE_FOCUS,
    STYLE_HEADING,
    STYLE_HUNK,
    STYLE_META,
    STYLE_MODE_INSERT,
    STYLE_MODE_NORMAL,
    STYLE_MODE_STANDARD,
    STYLE_REMOVED,
    STYLE_SELECTED,
    STYLE_TITLE,
    Painter,
)
from .tui_render import clip as _clip
from .tui_render import wrap as _wrap
from .ui import MocDecision, moc_title
from .vault import MocMatch, Vault
from .vimline import VimLineEditor

# Status-line spinner shown while a job runs.  Braille glyphs share one cell
# width on every terminal that ships a decent Unicode font.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.1


@dataclass
class CompletionPopup:
    start: int
    end: int
    matches: Sequence[FileMatch | SlashCommand]
    index: int = 0
    is_command: bool = False


@dataclass
class PromptModal:
    kind: str
    payload: dict[str, Any]
    box: dict[str, Any]
    event: threading.Event
    selected: int = 0
    scroll: int = 0
    editor: VimLineEditor | None = None
    completion: CompletionPopup | None = None
    message: str = ""


@dataclass
class SettingsState:
    """The settings modal: every Config field, one section at a time."""

    section: str = "paths"
    selected: int = 0
    editing: SettingField | None = None
    adding: bool = False
    editor: VimLineEditor | None = None
    completion: CompletionPopup | None = None
    message: str = ""
    values: dict[str, object] = field(default_factory=dict)


@dataclass
class WizardState:
    """The first-run setup wizard: a few essential pages."""

    page: str = wizard.PAGE_VAULT
    selected: int = 0
    editing: SettingField | None = None
    adding: bool = False
    list_index: int = -1
    editor: VimLineEditor | None = None
    completion: CompletionPopup | None = None
    message: str = ""
    values: dict[str, object] = field(default_factory=dict)


# The settings modal and the first-run wizard share the field-editing helpers:
# both expose the same working store and the same transient edit slots.
FormState = SettingsState | WizardState


SETTINGS_SECTIONS = tuple(section.key for section in SECTIONS)
SETTINGS_LABELS = {section.key: section.label for section in SECTIONS}


def _key_token(value: object) -> str | None:
    """Normalize a configured spawn key: "" / null means "always visible"."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith(("ctrl+", "ctrl-", "c-", "c+")):
        char = text.split("+")[-1] if "+" in text else text.split("-")[-1]
        char = char.strip().lower()
        if len(char) == 1:
            return chr(ord(char.upper()) - 64)
    if len(text) == 1:
        return text
    return {
        "tab": "\t",
        "enter": "\n",
        "esc": "\x1b",
        "space": " ",
    }.get(lowered, text)


class _LogStream:
    """Capture stdout/stderr lines into the TUI instead of the terminal.

    Libraries such as transformers and tqdm write progress and warnings to the
    process streams.  Letting them reach the terminal would paint over the
    curses UI, so every completed line is queued as a TUI log entry and shown
    only in the log window (``Ctrl-L``).
    """

    def __init__(self, app: TuiApp, original: Any):
        self._app = app
        self._original = original
        self._pending = ""
        self._lock = threading.Lock()
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "replace")

    def write(self, data: Any) -> int:
        text = str(data)
        with self._lock:
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                if line.strip():
                    self._app.events.put(("log", line))
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._pending.strip():
                self._app.events.put(("log", self._pending))
            self._pending = ""

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        try:
            return self._original.fileno()
        except (AttributeError, OSError, ValueError) as exc:
            raise OSError("stream has no file descriptor") from exc


@dataclass
class TuiApp:
    config: Config
    verbose: bool = False
    search_roots: list[Path] = field(default_factory=list)
    input_text: str = ""
    cursor: int = 0
    source_paths: list[Path] = field(default_factory=list)
    source_index: int = 0
    source_scroll: int = 0
    moc_hint: Path | None = None
    focus: str = PANE_INPUT
    active: str | None = None
    mode: str = MODE_NORMAL
    editor: VimLineEditor = field(default_factory=VimLineEditor)
    status: str = "ready"
    done: bool = False
    quit_requested: bool = False

    diff_entries: list[DiffEntry] = field(default_factory=list)
    diff_index: int = 0
    diff_scroll: int = 0
    asides: list[Aside] = field(default_factory=list)
    extras_scroll: int = 0
    card_groups: list[CardGroup] = field(default_factory=list)
    card_index: int = 0
    card_scroll: int = 0

    current_job_id: str | None = None
    current_stage: str = ""
    current_group: str = ""
    group_position: str = ""
    stage_status: dict[str, str] = field(default_factory=dict)
    running: bool = False
    worker: threading.Thread | None = None

    events: queue.Queue[Any] = field(default_factory=queue.Queue)
    prompt_requests: queue.Queue[PromptModal] = field(default_factory=queue.Queue)
    modal: PromptModal | None = None
    merge_review: PromptModal | None = None
    completion: CompletionPopup | None = None
    settings: SettingsState | None = None
    wizard: WizardState | None = None
    log_lines: list[str] = field(default_factory=list)
    spawn_visible: dict[str, bool] = field(default_factory=dict)
    log_scroll: int = 0
    fullscreen: str | None = None
    _spawn_return_focus: dict[str, str] = field(default_factory=dict)
    _anchor: tuple[float, float] | None = None

    _colors: dict[str, int] = field(default_factory=dict)
    _cursor_shape: bool | None = None
    panes: PaneRegistry = field(default_factory=build_registry)

    def __post_init__(self) -> None:
        roots = [Path(root).expanduser() for root in self.config.search_paths]
        self.search_roots = roots or [Path.home()]
        self.mode = self._base_mode()
        self.editor = self._new_editor(
            self.input_text,
            cursor=self.cursor,
            mode=self.mode,
        )
        # The registry follows the configuration, so a Config built directly by
        # a test still resolves the built-in boxes.
        self.panes = build_registry(self.config.tui_panes)
        # Capability lookup: the host finds the editor and the merge-review box
        # by what they declare, never by a hard-coded pane name.
        self._editor_name = next(
            (name for name in self.panes if self.panes[name].is_editor), PANE_INPUT
        )
        self._merge_name = next(
            (name for name in self.panes if self.panes[name].owns_merge_review), None
        )

    def pane_shown(self, name: str) -> bool:
        """Whether a pane with a spawn key is currently shown."""
        return bool(self.spawn_visible.get(name))

    def set_pane_shown(self, name: str, shown: bool) -> None:
        self.spawn_visible[name] = shown

    def spawn_key(self, name: str) -> str | None:
        """The key that toggles a pane, or None when it is always shown."""
        pane = self.panes.get(name)
        if pane is None:
            return None
        if name in self.config.tui_pane_keys:
            return _key_token(self.config.tui_pane_keys[name])
        return _key_token(pane.spawn_key)

    def hidden_panes(self) -> frozenset[str]:
        """Panes hidden right now because their spawn key has not been pressed."""
        return frozenset(
            name
            for name in self.panes
            if self.spawn_key(name) is not None and not self.pane_shown(name)
        )

    def toggle_pane(self, name: str) -> None:
        """Show or hide a pane that has a spawn key."""
        pane = self.panes.get(name)
        key = self.spawn_key(name)
        if pane is None or key is None:
            return
        if self.pane_shown(name):
            self.set_pane_shown(name, False)
            if self.active == name:
                self.active = None
            if self.fullscreen == name:
                self.fullscreen = None
            if self.focus == name:
                target = self._spawn_return_focus.get(name, self._editor_name)
                rects = self._layout_rects()
                if target not in rects:
                    target = (
                        self._editor_name
                        if self._editor_name in rects
                        else next(iter(rects), self._editor_name)
                    )
                self._set_focus(target)
            return
        self._spawn_return_focus[name] = self.focus
        self.set_pane_shown(name, True)
        pane.on_shown(self)

    def _pane_for_spawn_key(self, key: Any) -> str | None:
        for name in self.panes:
            token = self.spawn_key(name)
            if token is not None and key == token:
                return name
        return None

    @property
    def vim_movements(self) -> bool:
        """True when the configured movement style is Vim's modal editing."""
        return self.config.movement == MOVEMENT_VIM

    def _base_mode(self) -> str:
        """The input mode when the field does not hold focus."""
        return MODE_NORMAL if self.vim_movements else MODE_INSERT

    def new_editor(
        self,
        text: str = "",
        *,
        cursor: int | None = None,
        mode: str = MODE_INSERT,
    ) -> VimLineEditor:
        """Build an editor whose arrow keys follow the movement setting."""
        return VimLineEditor(
            text,
            cursor=cursor,
            mode=mode,
            arrow_movements=not self.vim_movements,
        )

    # -- input editor bridge ------------------------------------------------- #

    def sync_editor_from_fields(self) -> None:
        """Rebuild the editor when a test or callback set the fields directly."""
        desired = self.mode if self.vim_movements else MODE_INSERT
        if (
            self.editor.text != self.input_text
            or self.editor.cursor != self.cursor
            or self.editor.mode != desired
            or self.mode != desired
            or self.editor.arrow_movements != (not self.vim_movements)
        ):
            self.mode = desired
            self.editor = self._new_editor(
                self.input_text,
                cursor=min(self.cursor, len(self.input_text)),
                mode=desired,
            )

    def pull_editor(self) -> None:
        self.input_text = self.editor.text
        self.cursor = self.editor.cursor
        self.mode = self.editor.mode

    def show_help(self) -> None:
        if self.vim_movements:
            self.status = (
                "i/a/A insert · Esc normal · h l w b e motions · d c y operators "
                "· x X r ~ p P · u undo · Ctrl-R redo · . repeat "
                "· h/j/k/l panes · Enter select · Esc leave · f full · g/G ends "
                "· type to filter sources · Enter add source · /run start "
                "· /new reset · /history restore · /wizard setup "
                "· Ctrl-L logs · c commands "
                "· s settings · q quit"
            )
        else:
            self.status = (
                "type to edit · ←/→ cursor · Home/End line ends "
                "· Esc cancels · arrows panes · Enter select · Esc leave "
                "· type to filter sources · Enter add source · /run start "
                "· /new reset · /history restore · /wizard setup "
                "· Ctrl-L logs · f full "
                "· c commands · s settings · q quit"
            )

    # -- lifecycle ----------------------------------------------------------- #

    def run(self) -> int:
        try:
            curses.wrapper(self._main)
        except KeyboardInterrupt:
            return 130
        except curses.error as exc:
            print(f"tui error: {exc}")
            return 1
        return 0

    def _main(self, screen: Any) -> None:
        self.scr = screen
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        try:
            self.scr.keypad(True)
        except curses.error:
            pass
        # ncurses waits ESCDELAY (one second by default) before deciding that a
        # lone Escape is not the start of a function-key sequence.  Shrink it
        # so leaving insert mode feels immediate.
        try:
            curses.set_escdelay(25)
        except (AttributeError, curses.error):
            pass
        self.scr.timeout(120)
        self._init_colors()
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = _LogStream(self, stdout)
        sys.stderr = _LogStream(self, stderr)
        try:
            while not self.done:
                self._drain_events()
                self._maybe_accept_prompt()
                self._draw()
                key = self._read_key()
                if key is not None:
                    self._handle_key(key)
        finally:
            sys.stdout = stdout
            sys.stderr = stderr
        try:
            curses.curs_set(1)
        except curses.error:
            pass

    def _read_key(self) -> Any:
        try:
            return self.scr.get_wch()
        except curses.error:
            return None
        except KeyboardInterrupt:
            self.done = True
            return None

    # -- colors -------------------------------------------------------------- #

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
        except curses.error:
            return
        green = curses.COLOR_GREEN
        pairs = {
            STYLE_TITLE: (curses.COLOR_CYAN, -1),
            STYLE_ADDED: (curses.COLOR_GREEN, -1),
            STYLE_REMOVED: (curses.COLOR_RED, -1),
            STYLE_HUNK: (curses.COLOR_MAGENTA, -1),
            STYLE_META: (curses.COLOR_BLUE, -1),
            STYLE_HEADING: (curses.COLOR_YELLOW, -1),
            STYLE_DIM: (curses.COLOR_WHITE, -1),
            STYLE_ACTIVE: (curses.COLOR_BLACK, green),
            STYLE_ACTIVE_BORDER: (green, -1),
            STYLE_FOCUS: (curses.COLOR_WHITE, -1),
            # The mode badge is a filled chip: black text on a coloured
            # background so it reads as a button, not as plain chrome.
            STYLE_MODE_STANDARD: (curses.COLOR_BLACK, curses.COLOR_CYAN),
            STYLE_MODE_NORMAL: (curses.COLOR_BLACK, curses.COLOR_BLUE),
            STYLE_MODE_INSERT: (curses.COLOR_BLACK, curses.COLOR_GREEN),
        }
        for index, (style, (foreground, background)) in enumerate(pairs.items(), start=1):
            try:
                curses.init_pair(index, foreground, background)
            except curses.error:
                continue
            self._colors[style] = curses.color_pair(index)

    # -- worker events ------------------------------------------------------- #

    def _drain_events(self) -> None:
        while True:
            try:
                item = self.events.get_nowait()
            except queue.Empty:
                return
            kind, payload = item
            if kind == "log":
                self._on_log(str(payload))
            elif kind == "event":
                self._on_pipeline_event(dict(payload))
            elif kind == "failed":
                # The header is a one-line progress slot: a raw exception there
                # steals it from the job state and can be arbitrarily long.
                # Keep the failure in the log and leave a pointer in the status.
                self.running = False
                self.status = "job failed · Ctrl-L for the log"
                for line in str(payload).splitlines() or [str(payload)]:
                    self._on_log(f"[error] {line}")
                if self.quit_requested:
                    self.done = True
            elif kind == "done":
                self.running = False
                state = payload
                if state is not None:
                    self.current_job_id = getattr(state, "job_id", self.current_job_id)
                    self._reload_artifacts()
                self.current_stage = ""
                self.status = "job finished"
                if self.quit_requested:
                    self.done = True

    def _on_log(self, message: str) -> None:
        clean = " ".join(message.strip().splitlines())
        if not clean:
            return
        self.log_lines.append(clean)
        self.log_lines = self.log_lines[-400:]
        for name in self.panes:
            self.panes[name].on_log(self)

    def _on_pipeline_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "job_started":
            self.current_job_id = str(event.get("job_id") or "") or None
            self.running = True
            self.stage_status = {}
            self.current_stage = ""
            self.diff_entries = []
            self.card_groups = []
            self.asides = []
            self.diff_index = 0
            self.card_index = 0
            self.extras_scroll = 0
        elif kind == "group_start":
            self.current_group = str(event.get("topic") or event.get("group_id") or "")
            index = event.get("index")
            total = event.get("total")
            self.group_position = f"{index}/{total}" if index and total else ""
            self.status = f"group {self.group_position}: {self.current_group}"
        elif kind == "stage_start":
            self.current_stage = str(event.get("stage") or "")
            self.stage_status[self.current_stage] = "running"
            if event.get("group_topic"):
                self.current_group = str(event.get("group_topic"))
            self.status = self._stage_text()
        elif kind == "stage_done":
            stage = str(event.get("stage") or "")
            self.stage_status[stage] = "done"
            if stage == Stage.APPLY.value:
                self._reload_diffs()
            elif stage in {Stage.COMPOSE.value, Stage.FLASHCARDS.value}:
                self._reload_cards()
            elif stage == Stage.EXTRACT.value:
                self._reload_asides()
            elif stage == Stage.REPORT.value:
                self._reload_artifacts()
        elif kind == "stage_failed":
            stage = str(event.get("stage") or "")
            self.stage_status[stage] = "failed"
        elif kind == "file_changed":
            self._upsert_diff(
                DiffEntry(
                    path=str(event.get("path") or ""),
                    action=str(event.get("action") or "update"),
                    added=int(event.get("added_lines") or 0),
                    removed=int(event.get("removed_lines") or 0),
                    diff=str(event.get("diff") or ""),
                )
            )
        elif kind == "file_skipped":
            self._on_log(f"{event.get('reason')}: {event.get('path')}")
        elif kind == "cards_ready":
            self._reload_cards()
        elif kind == "job_done":
            self.running = False
            self.current_stage = ""
            self._reload_artifacts()
            if self.quit_requested:
                self.done = True

    def _spinner_frame(self) -> str:
        """The current spinner glyph, advancing with wall-clock time."""
        index = int(time.monotonic() / SPINNER_INTERVAL) % len(SPINNER_FRAMES)
        return SPINNER_FRAMES[index]

    def _status_text(self) -> str:
        """Header text: the product name plus the current status, if any."""
        text = self.status.strip()
        if self.running:
            text = f"{self._spinner_frame()} {text or 'working'}"
        if text:
            return f"scriptorium · {text}"
        return "scriptorium"

    def stage_text(self) -> str:
        if not self.current_stage:
            return "idle"
        try:
            label = STAGE_LABELS[Stage(self.current_stage)]
        except (ValueError, KeyError):
            label = self.current_stage
        base = f"{self.current_stage}: {label}"
        if self.current_group:
            position = f"{self.group_position} " if self.group_position else ""
            return f"{position}{self.current_group} · {base}"
        return base

    # -- artifacts ----------------------------------------------------------- #

    def _job_path(self, name: str) -> Path | None:
        if not self.current_job_id:
            return None
        return self.config.artifacts_dir / "jobs" / self.current_job_id / name

    def _reload_diffs(self) -> None:
        path = self._job_path("apply.json")
        if path is None:
            return
        expanded = {entry.path for entry in self.diff_entries if entry.expanded}
        self.diff_entries = load_diff_entries(path)
        for entry in self.diff_entries:
            entry.expanded = entry.path in expanded
        self.diff_index = min(self.diff_index, max(0, len(self.diff_entries) - 1))

    def _reload_cards(self) -> None:
        plan = self._job_path("05.plan.json")
        cards = self._job_path("cards.json")
        if plan is None or cards is None:
            return
        previous = {group.path: group.expanded for group in self.card_groups}
        self.card_groups = load_card_groups(plan, cards)
        for group in self.card_groups:
            group.expanded = previous.get(group.path, True)
        self.card_index = min(self.card_index, max(0, len(self.card_groups) - 1))

    def _reload_asides(self) -> None:
        plan = self._job_path("05.plan.json")
        if plan is None:
            return
        extract = self._job_path("03.extract.json")
        self.asides = load_asides(plan, extract)
        for name in self.panes:
            self.panes[name].on_asides(self)

    def _reload_artifacts(self) -> None:
        self._reload_diffs()
        self._reload_cards()
        self._reload_asides()

    def _upsert_diff(self, entry: DiffEntry) -> None:
        if not entry.group:
            entry.group = {
                "create": "created",
                "merge": "merged",
                "moc": "moc",
            }.get(entry.action, "merged")
        for index, existing in enumerate(self.diff_entries):
            if existing.path == entry.path and existing.action == entry.action:
                entry.expanded = existing.expanded
                self.diff_entries[index] = entry
                return
        self.diff_entries.append(entry)

    # -- job control --------------------------------------------------------- #

    def submit_input(self) -> None:
        """Add one source to the sidebar, or run the collected sources."""
        self._sync_editor_from_fields()
        text = self.input_text.strip()
        if not text:
            self.status = "type a source path or /run"
            return
        if self._is_slash_command(text):
            if text == "/run":
                self._clear_input()
                self._start_job()
                return
            if text == "/new":
                self._clear_input()
                self._new_session()
                return
            if text == "/history":
                self._clear_input()
                self.open_history()
                return
            if text == "/wizard":
                self._clear_input()
                self.open_setup_wizard()
                return
            self.status = f"unknown command {text}"
            return

        # The whole line is one path, so a filename may contain spaces.  A line
        # that only looks like several paths is rejected instead of silently
        # adding more than one source.
        plan = parse_input(shlex.quote(text), self.search_roots)
        missing = missing_paths(plan)
        if missing:
            split = parse_input(text, self.search_roots)
            if len(split.sources) > 1 and not missing_paths(split):
                self.status = "add one source at a time"
            else:
                self.status = f"file not found: {missing[0]}"
            return
        if plan.moc_hints:
            self.moc_hint = plan.moc_hints[0]
        if not plan.sources:
            self._clear_input()
            if self.moc_hint is not None:
                self.status = f"MOC hint set to {self.moc_hint.name}"
            return
        self._add_source(plan.sources[0])

    def _is_slash_command(self, text: str) -> bool:
        """True when the whole line is a slash command, not a path."""
        return (
            text.startswith("/")
            and "/" not in text[1:]
            and not any(char.isspace() for char in text)
        )

    def _add_source(self, source: Path) -> None:
        """Add one existing path to the sidebar and clear the input line."""
        source = Path(source).expanduser()
        if not source.exists():
            self.status = f"file not found: {source}"
            return
        plan = parse_input(shlex.quote(str(source)), self.search_roots)
        if plan.moc_hints:
            self.moc_hint = plan.moc_hints[0]
            self._clear_input()
            self.status = f"MOC hint set to {self.moc_hint.name}"
            return
        if source not in self.source_paths:
            self.source_paths.append(source)
        self.source_index = self.source_paths.index(source)
        self._clear_input()
        hint = f" · MOC hint {self.moc_hint.name}" if self.moc_hint else ""
        self.status = (
            f"added {source.name} · {len(self.source_paths)} source(s){hint} · /run to start"
        )

    def _clear_input(self) -> None:
        # Submitting leaves the field in its resting mode: normal mode under Vim
        # movements, insert mode otherwise.  The box also leaves the selected
        # state, so typing requires a fresh Enter.
        mode = self._base_mode()
        self.input_text = ""
        self.cursor = 0
        self.mode = mode
        self.editor = self._new_editor("", cursor=0, mode=mode)
        self.completion = None
        if self.active == self._editor_name:
            self.active = None

    def _new_session(self) -> None:
        """Clear the current session so the next job starts from scratch.

        Only in-memory UI state is reset: collected sources, MOC hint, job
        artifacts and logs.  Job directories under ``$SCRIPTORIUM_ARTIFACTS_DIR`` stay
        on disk, so a completed job can still be resumed from the CLI.
        """
        if self.running:
            self.status = "a job is already running"
            return
        self.source_paths = []
        self.source_index = 0
        self.source_scroll = 0
        self.moc_hint = None
        self.diff_entries = []
        self.diff_index = 0
        self.diff_scroll = 0
        self.asides = []
        self.extras_scroll = 0
        self.card_groups = []
        self.card_index = 0
        self.card_scroll = 0
        self.current_job_id = None
        self.current_stage = ""
        self.current_group = ""
        self.group_position = ""
        self.stage_status = {}
        self.log_lines = []
        self.log_scroll = 0
        self.completion = None
        # Boxes own their view state: let each one clear itself, so a pane added
        # by configuration is reset by the same command without host changes.
        self.panes.reset(self)
        self.status = "new session · add sources and type /run"

    # -- commands ------------------------------------------------------------ #

    def open_commands(self) -> None:
        """Show the list of slash commands currently available."""
        self.active = None
        self.status = ""
        self.completion = None
        self.modal = PromptModal(
            kind="commands",
            payload={"commands": list(SLASH_COMMANDS)},
            box={},
            event=threading.Event(),
        )
        self.editor.set_mode(self._base_mode())
        self._pull_editor()

    def _handle_commands_key(self, modal: PromptModal, key: Any) -> None:
        commands = list(modal.payload.get("commands") or SLASH_COMMANDS)
        if not commands:
            self.modal = None
            return
        if self._is_down_key(key):
            modal.selected = min(len(commands) - 1, modal.selected + 1)
            return
        if self._is_up_key(key):
            modal.selected = max(0, modal.selected - 1)
            return
        if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
            command = commands[modal.selected]
            self.modal = None
            self.focus = self._editor_name
            self.active = self._editor_name
            self.input_text = command.name
            self.cursor = len(command.name)
            self.mode = MODE_INSERT
            self.editor = self._new_editor(command.name, cursor=self.cursor, mode=MODE_INSERT)
            self.refresh_command_completion()
            return
        if key in {"q", "Q", "\x1b"}:
            self.modal = None

    # -- history ------------------------------------------------------------- #

    def open_history(self) -> None:
        """Open the floating list of completed runs."""
        if self.running:
            self.status = "a job is already running"
            return
        entries = HistoryStore(self.config.history_dir).entries()
        self.active = None
        self.status = ""
        self.modal = PromptModal(
            kind="history",
            payload={"entries": entries},
            box={},
            event=threading.Event(),
        )
        self.editor.set_mode(self._base_mode())
        self._pull_editor()

    def _handle_history_key(self, modal: PromptModal, key: Any) -> None:
        entries: list[HistoryEntry] = list(modal.payload.get("entries") or [])
        if key in {"\x1b", "q", "Q"}:
            self.modal = None
            return
        if not entries:
            return
        if self._is_down_key(key) or key == "\t":
            modal.selected = min(len(entries) - 1, modal.selected + 1)
            return
        if self._is_up_key(key):
            modal.selected = max(0, modal.selected - 1)
            return
        if isinstance(key, str) and len(key) == 1 and key in "123456789":
            index = "123456789".index(key)
            if index < len(entries):
                modal.selected = index
                self._load_history(entries[index])
            return
        if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
            self._load_history(entries[modal.selected])

    def _load_history(self, entry: HistoryEntry) -> None:
        """Restore a snapshot into the panes, as if that run had just finished."""
        self.modal = None
        self.current_job_id = entry.job_id or None
        self.source_paths = [Path(path) for path in entry.sources]
        self.source_index = 0
        self.source_scroll = 0
        self.moc_hint = None
        self.diff_entries = list(entry.changes)
        self.diff_index = 0
        self.diff_scroll = 0
        self.card_groups = list(entry.card_groups)
        self.card_index = 0
        self.card_scroll = 0
        self.asides = list(entry.extras)
        self.extras_scroll = 0
        self.current_stage = ""
        self.current_group = ""
        self.group_position = ""
        self.stage_status = {}
        self.log_lines = []
        self.log_scroll = 0
        self.running = False
        self.active = None
        self.completion = None
        self.panes.reset(self)
        self.status = f"history · {entry.course or entry.job_id}"

    def start_job(self) -> None:
        if self.running:
            self.status = "a job is already running"
            return
        if not self.source_paths:
            self.status = "add a source first: type a path and press Enter"
            return
        missing = [
            path
            for path in [*self.source_paths, self.moc_hint]
            if path is not None and not path.exists()
        ]
        if missing:
            self.status = f"file not found: {missing[0]}"
            return
        sources = list(self.source_paths)
        try:
            llm_api_key(self.config)
        except ScriptoriumError as exc:
            self.status = str(exc)
            return

        course = clean_course_name(sources[0].stem)
        options = RunOptions(
            course=course,
            moc=str(self.moc_hint) if self.moc_hint else None,
        )
        self.running = True
        self.quit_requested = False
        self.current_stage = ""
        self.current_group = ""
        self.group_position = ""
        self.stage_status = {}
        self.diff_entries = []
        self.card_groups = []
        self.asides = []
        self.diff_index = 0
        self.card_index = 0
        self.extras_scroll = 0
        self.current_job_id = None
        label = sources[0].name if len(sources) == 1 else f"{len(sources)} sources"
        self.status = f"starting job for {label}"
        self.editor.set_mode(self._base_mode())
        self._pull_editor()
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(sources, options),
            name="scriptorium-worker",
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, sources: list[Path], options: RunOptions) -> None:
        logger = Logger(
            None,
            verbose=self.verbose,
            sink=lambda message: self.events.put(("log", message)),
        )
        ui = TuiConsoleUI(
            log=lambda message: self.events.put(("log", message)),
            ask=self._ask_from_worker,
        )
        pipeline = Pipeline(
            self.config,
            ui,
            logger,
            on_event=lambda event: self.events.put(("event", event)),
        )
        try:
            state = pipeline.run(sources, options)
        except Exception as exc:  # keep the TUI alive and show the failure
            self.events.put(("failed", str(exc)))
            return
        self.events.put(("done", state))

    def _ask_from_worker(self, kind: str, payload: dict[str, Any]) -> Any:
        box: dict[str, Any] = {}
        event = threading.Event()
        request = PromptModal(kind=kind, payload=payload, box=box, event=event)
        self.prompt_requests.put(request)
        while not event.wait(0.2):
            if self.done:
                return None
        return box.get("response")

    def _maybe_accept_prompt(self) -> None:
        if self.modal is not None or self.merge_review is not None:
            return
        try:
            request = self.prompt_requests.get_nowait()
        except queue.Empty:
            return
        if request.kind == "merge":
            # A merge is reviewed inside the Changes box, never in a floating
            # window: the diff and the y/n/m/a/q prompt replace its contents.
            self.merge_review = request
            if self._merge_name is not None:
                self.focus = self._merge_name
                self.active = self._merge_name
            self.fullscreen = None
            self.editor.set_mode(self._base_mode())
            self._pull_editor()
            return
        self.modal = request
        if request.kind == "language":
            default = str(request.payload.get("default") or "en")
            request.editor = self._new_editor(default, mode=MODE_INSERT)
            request.message = "Enter accept · Esc use the default"
        self.editor.set_mode(self._base_mode())
        self._pull_editor()

    def _start_moc_name(self, modal: PromptModal) -> None:
        default = moc_title(
            str(modal.payload.get("default_title") or modal.payload.get("course") or "Course")
        )
        modal.kind = "moc_name"
        modal.message = "Enter create · Esc back · the MOC suffix is added automatically"
        modal.editor = self._new_editor(default)
        modal.editor.enter_insert(len(default))

    def _start_moc_manual(self, modal: PromptModal) -> None:
        modal.kind = "moc_manual"
        modal.message = "type a note name or vault path · Tab completion · Enter select"
        modal.completion = None
        modal.editor = self._new_editor("")
        modal.editor.enter_insert(0)

    def _moc_note_paths(self) -> list[str]:
        vault = Vault(self.config.notes_root)
        return vault.find_mocs(self.config.search_paths)

    def _start_moc_completion(self, modal: PromptModal) -> None:
        editor = modal.editor
        if editor is None:
            return
        matches = complete_note_paths(self._moc_note_paths(), editor.text)
        if not matches:
            modal.completion = None
            modal.message = "no MOC matches that name"
            return
        modal.message = "Tab cycle · Enter accept · Esc close"
        modal.completion = CompletionPopup(start=0, end=len(editor.text), matches=matches, index=0)
        self._apply_moc_completion(modal)
        if len(matches) == 1:
            modal.completion = None

    def _cycle_moc_completion(self, modal: PromptModal, delta: int) -> None:
        popup = modal.completion
        if popup is None or not popup.matches:
            return
        popup.index = (popup.index + delta) % len(popup.matches)
        self._apply_moc_completion(modal)

    def _apply_moc_completion(self, modal: PromptModal) -> None:
        popup = modal.completion
        editor = modal.editor
        if popup is None or editor is None:
            return
        match = popup.matches[popup.index]
        if not isinstance(match, FileMatch):
            return
        text = match.relative
        editor.set_text(text, cursor=len(text))
        popup.start = 0
        popup.end = len(text)

    def _commit_manual_moc(self, modal: PromptModal) -> None:
        editor = modal.editor
        if editor is None:
            return
        raw = editor.text.strip().strip("'\"")
        if not raw:
            modal.message = "the MOC name cannot be empty"
            return
        vault = Vault(self.config.notes_root)
        path: str | None = None
        try:
            candidate = Path(raw).expanduser()
            if candidate.is_absolute():
                path = vault.relative(candidate)
            elif vault.exists(raw.replace("\\", "/")):
                path = vault.relative(vault.resolve(raw.replace("\\", "/")))
        except (ValueError, VaultError):
            path = None
        if path is not None and Path(path).suffix.lower() not in {".md", ".markdown"}:
            modal.message = "the MOC must be a Markdown note"
            return
        if path is None or not vault.exists(path):
            wanted = Path(raw.replace("\\", "/")).stem.casefold()
            matches = [
                candidate
                for candidate in self._moc_note_paths()
                if Path(candidate).stem.casefold() == wanted
            ]
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                modal.message = "several MOCs have that name; type a vault path"
                return
        if path is None:
            modal.message = f"no such MOC in the vault: {raw}"
            return
        self._answer_modal(MocDecision.use(path))

    def _answer_modal(self, response: Any) -> None:
        if self.modal is None:
            return
        self.modal.box["response"] = response
        self.modal.event.set()
        self.modal = None

    def answer_merge(self, response: str) -> None:
        review = self.merge_review
        if review is None:
            return
        review.box["response"] = response
        review.event.set()
        self.merge_review = None
        self.active = None
        self.status = "merge reviewed"

    def request_quit(self) -> None:
        if self.running:
            if self.quit_requested:
                self.status = "waiting for the job to finish; press Ctrl-C to force quit"
            else:
                self.quit_requested = True
                self.status = "the TUI will close when the current job finishes"
            return
        self.done = True

    # -- key handling -------------------------------------------------------- #

    def _handle_key(self, key: Any) -> None:
        if key == curses.KEY_RESIZE:
            self._draw()
            return
        if key == "\x03":  # Ctrl-C always exits, even while a worker is running.
            self.done = True
            return
        if key == "\x1b":
            # Escape dismisses transient feedback so the header does not keep
            # stale messages or mode hints.
            self.status = ""
        if self.wizard is not None:
            self._handle_wizard_key(key)
            return
        if self.settings is not None:
            self._handle_settings_key(key)
            return
        if self.modal is not None:
            self._handle_modal_key(key)
            return
        if self.merge_review is not None:
            self._handle_merge_key(key)
            return
        spawn = self._pane_for_spawn_key(key)
        if spawn is not None:
            self.toggle_pane(spawn)
            return
        if self.active is not None and key == "\x1b":
            if (
                self.focus == self._editor_name
                and self.active == self._editor_name
                and self.vim_movements
                and self.mode == MODE_INSERT
            ):
                # Leaving insert mode keeps the input box selected: normal
                # commands must keep editing the line instead of dropping back
                # to spatial focus navigation.
                self._handle_input_key(key)
                return
            # Escape leaves the selected box without moving the focus.
            self.active = None
            self.completion = None
            if self.focus == self._editor_name:
                self.editor.set_mode(self._base_mode())
                self._pull_editor()
            return
        if self.focus == self._editor_name and self.active == self._editor_name:
            self._handle_input_key(key)
            return
        self._handle_normal_key(key)

    def _handle_input_key(self, key: Any) -> None:
        """Forward one key to the editor box."""
        pane = self.panes.get(self._editor_name)
        if pane is not None:
            pane.handle_key(self, key, active=True)

    def _handle_normal_key(self, key: Any) -> None:
        """Keys for the focused box, then the application-wide shortcuts.

        The focused pane sees the key first, so a box can own a shortcut inside
        its own bounds (the log viewer closes itself on ``q``).  Everything it
        does not consume falls through to the host: focus movement, overlays and
        the global commands.
        """
        if self.active is not None:
            self._handle_active_key(key)
            return
        if self.focus != self._editor_name:
            pane = self.panes.get(self.focus)
            if pane is not None and pane.handle_key(self, key, active=False):
                return
        spawn = self._pane_for_spawn_key(key)
        if spawn is not None:
            self.toggle_pane(spawn)
            return
        if isinstance(key, str):
            if key == "q":
                self.request_quit()
            elif key == "i":
                self.set_focus(self._editor_name)
                self.activate_box()
            elif key == "s":
                self.open_settings()
            elif key == "c":
                self.open_commands()
            elif key == "r":
                self.start_job()
            elif key == "\t":
                self.cycle_focus(1)
            elif key in {"\n", "\r"}:
                self.activate_box()
            elif key == "?":
                self.show_help()
            elif key == "f":
                self.toggle_fullscreen()
            elif self.vim_movements and key in {"h", "l", "j", "k"}:
                self.move_focus(key)
            return
        if self.vim_movements:
            if key in {curses.KEY_ENTER, 10, 13}:
                self.activate_box()
            return
        if key == curses.KEY_DOWN:
            self.move_focus("j")
        elif key == curses.KEY_UP:
            self.move_focus("k")
        elif key == curses.KEY_LEFT:
            self.move_focus("h")
        elif key == curses.KEY_RIGHT:
            self.move_focus("l")
        elif key in {curses.KEY_ENTER, 10, 13}:
            self.activate_box()

    def activate_box(self) -> None:
        """Enter selects the focused box so its options can be navigated."""
        if self.focus == self._editor_name:
            self.active = self._editor_name
            self.editor.enter_insert(self.editor.cursor)
            self._pull_editor()
            return
        self.active = self.focus

    def _handle_active_key(self, key: Any) -> None:
        """Movement and actions inside the selected box, using the configured style."""
        if key == "i" and self.vim_movements:
            # ``i`` always moves the selection to the input line, even when a
            # different box is already selected, and enters insert mode there.
            self.set_focus(self._editor_name)
            self.activate_box()
            return
        spawn = self._pane_for_spawn_key(key)
        if spawn is not None:
            self.toggle_pane(spawn)
            return
        pane = self.panes.get(self.active or "")
        if pane is not None and pane.handle_key(self, key, active=True):
            return
        if key in {"q", "Q"}:
            self.request_quit()
        elif key == "f":
            self.toggle_fullscreen()
        elif key == "c":
            self.open_commands()

    def _handle_merge_key(self, key: Any) -> None:
        """Review a pending merge: the merge-review box owns the keys."""
        if self.merge_review is None or self._merge_name is None:
            return
        pane = self.panes.get(self._merge_name)
        if pane is not None:
            pane.handle_key(self, key, active=True)

    def _is_down_key(self, key: Any) -> bool:
        """True for the configured "next item" key of an overlay list."""
        return key == "j" if self.vim_movements else key == curses.KEY_DOWN

    def _is_up_key(self, key: Any) -> bool:
        """True for the configured "previous item" key of an overlay list."""
        return key == "k" if self.vim_movements else key == curses.KEY_UP

    def toggle_fullscreen(self) -> None:
        """Expand the focused pane over every box except the input line.

        Pressing ``f`` again restores the configured layout, and pressing it
        while a pane is expanded always restores it, so the key is never a
        dead end (for example after the focus moved to the input line, which
        cannot be expanded).
        """
        if self.fullscreen is not None:
            self.fullscreen = None
            return
        if self.focus == self._editor_name:
            return
        self.fullscreen = self.focus

    def layout_rects(self) -> dict[str, PaneRect]:
        """Pane rectangles as drawn, including spawned and carved panes.

        The layout places the *placeable* panes, minus any pane whose spawn key
        has not been pressed.  Every other registered pane is carved out of its
        parent's rectangle (or out of the expanded pane when one is fullscreen),
        so the topology comes from the registry and never from a hard-coded list
        here.
        """
        rects = self._pane_rects()
        if not rects:
            return rects
        expanded = self._validated_fullscreen(rects)
        if expanded is not None:
            area = content_area(rects)
            if area is not None:
                # A full-screen pane keeps the editor line and takes the rest.
                rects = {name: r for name, r in rects.items() if name == self._editor_name}
                rects[expanded] = area
        for name in self.panes.carved():
            self._carve_pane(name, rects, expanded)
        return rects

    def _carve_pane(
        self,
        name: str,
        rects: dict[str, PaneRect],
        expanded: str | None,
    ) -> None:
        """Cut one child pane out of its base rectangle, in place."""
        pane = self.panes[name]
        if expanded == name or not pane.visible(self):
            return
        parent = pane.parent
        if parent is not None and parent in rects:
            base_name = parent
        elif expanded is not None and pane.splits_fullscreen and expanded in rects:
            # A transient pane may split the expanded box it normally shares.
            base_name = expanded
        else:
            return
        main, child = split_rect(
            rects[base_name],
            pane.side,
            pane.split_ratio(self),
            collapse=pane.collapse,
        )
        if child is None:
            return
        if main is None:
            rects.pop(base_name, None)
        else:
            rects[base_name] = main
        rects[name] = child

    def _validated_fullscreen(self, rects: dict[str, PaneRect]) -> str | None:
        """The expanded pane, or None when it can no longer be shown."""
        expanded = self.fullscreen
        if expanded is None:
            return None
        pane = self.panes.get(expanded)
        if pane is None or expanded == self._editor_name or not pane.visible(self):
            self.fullscreen = None
            return None
        if content_area(rects) is None:
            self.fullscreen = None
            return None
        return expanded

    def _pane_rects(self) -> dict[str, PaneRect]:
        """Rectangles for the current terminal, or nothing if it is not ready."""
        try:
            rows, cols = self.scr.getmaxyx()
        except (curses.error, AttributeError):
            return {}
        return pane_rects(
            self.config.tui_layout,
            rows,
            cols,
            self.hidden_panes(),
        )

    def move_focus(self, direction: str) -> None:
        """Move focus to the adjacent pane, tiling-manager style.

        The movement carries an anchor point: a pane edge or the entry point
        used by the previous step.  Pressing a direction with no pane keeps the
        anchor on that side, so the following perpendicular move lands on the
        box on that side (right then up reaches the upper-right pane).
        """
        rects = self._layout_rects()
        origin = rects.get(self.focus)
        anchor = self._anchor
        if origin is not None and anchor is not None:
            anchor = clamp_point(anchor, origin)
        target = neighbor_rects(rects, self.focus, direction, point=anchor)
        if target is None:
            if origin is not None:
                base = anchor if anchor is not None else (origin.center_x, origin.center_y)
                self._anchor = edge_point(origin, direction, base)
            return
        target_rect = rects[target]
        self._set_focus(target)
        if anchor is None:
            self._anchor = None
        else:
            self._anchor = clamp_point(anchor, target_rect)

    def cycle_focus(self, delta: int) -> None:
        """Tab moves in reading order, independent of the spatial arrows."""
        order = list(layout_panes(self.config.tui_layout))
        rects = self.layout_rects()
        order = [pane for pane in order if pane in rects]
        # Carved panes (Extras, the log viewer) follow their parent in the ring.
        for name in self.panes.carved():
            if name not in rects:
                continue
            parent = self.panes[name].parent
            if parent in order:
                order.insert(order.index(parent) + 1, name)
            else:
                order.append(name)
        if not order or self.focus not in order:
            return
        index = order.index(self.focus)
        self.set_focus(order[(index + delta) % len(order)])

    def set_focus(self, target: str) -> None:
        if self.fullscreen is not None and target != self.fullscreen:
            # Leaving the expanded pane restores the configured layout, so the
            # focus never lands on a box that is not on screen.
            self.fullscreen = None
        self.focus = target
        self.active = None
        # Any focus change other than a spatial move starts from the centre.
        self._anchor = None
        if target != self._editor_name:
            self.editor.set_mode(self._base_mode())
            self._pull_editor()
        self.completion = None

    # -- completion ---------------------------------------------------------- #

    def start_completion(self) -> None:
        """Open the suggestion popup for the text already on the line."""
        self.refresh_completion()

    def refresh_completion(self) -> None:
        """Open, filter or close the suggestion popup after an edit.

        The popup follows the typed text, so a source match never requires a
        Tab keystroke.  The line itself is left untouched: the highlighted
        candidate is only applied when the user presses Enter.
        """
        self._sync_editor_from_fields()
        if self._command_input():
            self.refresh_command_completion()
            return
        text = self.input_text
        if not text.strip():
            self.completion = None
            return
        matches = complete_files(self.search_roots, text)
        if not matches:
            self.completion = None
            return
        previous = self.completion.index if self.completion is not None else 0
        self.completion = CompletionPopup(
            start=0,
            end=len(text),
            matches=matches,
            index=min(previous, len(matches) - 1),
        )

    def _command_input(self) -> bool:
        """True while the line is a slash command, not an absolute path."""
        text = self.input_text
        return (
            text.startswith("/")
            and "/" not in text[1:]
            and not any(char.isspace() for char in text)
        )

    def refresh_command_completion(self) -> None:
        """Open, filter or close the command suggestions as the user types."""
        if not self._command_input():
            if self.completion is not None and self.completion.is_command:
                self.completion = None
            return
        matches = matching_commands(self.input_text)
        if not matches:
            self.completion = None
            return
        previous = self.completion.index if self.completion is not None else 0
        self.completion = CompletionPopup(
            start=0,
            end=len(self.input_text),
            matches=matches,
            index=min(previous, len(matches) - 1),
            is_command=True,
        )

    def accept_command_completion(self) -> None:
        """Insert the selected command and submit it."""
        popup = self.completion
        if popup is not None and popup.is_command:
            self._apply_completion()
        self.completion = None
        self.submit_input()

    def accept_source_completion(self) -> None:
        """Add the highlighted source to the sidebar and clear the input."""
        popup = self.completion
        if popup is None or popup.is_command or not popup.matches:
            self.submit_input()
            return
        match = popup.matches[popup.index]
        self.completion = None
        self._add_source(Path(match.path))

    def cycle_completion(self, delta: int) -> None:
        popup = self.completion
        if popup is None or not popup.matches:
            return
        popup.index = (popup.index + delta) % len(popup.matches)
        if not popup.is_command:
            # Only the highlight moves: the line keeps the typed prefix until
            # Enter accepts the candidate.
            return

    def _apply_completion(self) -> None:
        popup = self.completion
        if popup is None:
            return
        match = popup.matches[popup.index]
        new_text, cursor = replace_token(self.input_text, popup.start, popup.end, str(match.path))
        self.input_text = new_text
        self.cursor = cursor
        self.editor.set_text(new_text, cursor)
        popup.end = cursor

    # -- settings ------------------------------------------------------------ #

    def open_settings(self) -> None:
        self.active = None
        self.settings = SettingsState(values=self._settings_snapshot())
        self.editor.set_mode(self._base_mode())
        self._pull_editor()

    def _settings_snapshot(self) -> dict[str, object]:
        """Copy every editable Config value into the modal's working store."""
        config = self.config
        return {
            "notes_root": str(config.notes_root),
            "artifacts_dir": str(config.artifacts_dir),
            "search_paths": [str(path) for path in config.user_search_paths or ()],
            "attachment_dir": str(config.attachment_dir),
            "flashcard_skill": str(config.flashcard_skill),
            "llm_model": config.llm_model,
            "llm_base_url": config.llm_base_url,
            "llm_timeout": str(config.llm_timeout),
            "llm_api_key": config.llm_api_key,
            "llm_json_mode": config.llm_json_mode,
            "asr_model": config.asr_model,
            "asr_api_model": config.asr_api_model,
            "asr_api_base_url": config.asr_api_base_url,
            "asr_api_protocol": config.asr_api_protocol,
            "asr_api_timeout": str(config.asr_api_timeout),
            "asr_api_key": config.asr_api_key,
            "language": config.language or "detect",
            "device": config.device,
            "batch_size": str(config.batch_size),
            "flashcard_max_per_note": str(config.flashcard_max_per_note),
            "extra_context_chars": str(config.extra_context_chars),
            "image_input": config.image_input,
            "scanned_pdf": config.scanned_pdf,
            "ocr_languages": config.ocr_languages,
            "max_scanned_pages": str(config.max_scanned_pages),
            "merge_policy": config.merge_policy,
            "target_policy": config.target_policy,
            "movement": config.movement,
            "target_auto_above": str(config.target_auto_above),
            "target_min_gap": str(config.target_min_gap),
            "disabled_warnings": list(config.disabled_warnings),
            "tui_layout": layout_text(config.tui_layout),
        }

    def _list_value(self, state: FormState, key: str) -> list:
        """Return a list field's entries as a fresh list (or ``[]`` when unset)."""
        value = state.values.get(key)
        return list(value) if isinstance(value, list) else []

    def _section_rows(self, state: SettingsState) -> list[SettingsRow]:
        """The selectable rows of the active section."""
        return self._rows_for_fields(state, section_by_key(state.section).fields)

    def _rows_for_fields(
        self,
        state: SettingsState | WizardState,
        fields: Sequence[SettingField],
        *,
        add_list_row: bool = False,
    ) -> list[SettingsRow]:
        """Expand fields into selectable rows (lists and flags take one per item)."""
        rows: list[SettingsRow] = []
        for setting in fields:
            if setting.kind == KIND_LIST:
                entries = self._list_value(state, setting.key)
                if not entries:
                    rows.append(SettingsRow(setting, -1))
                else:
                    rows.extend(SettingsRow(setting, index) for index in range(len(entries)))
                    if add_list_row:
                        rows.append(SettingsRow(setting, -1))
            elif setting.kind == KIND_FLAGS:
                rows.extend(SettingsRow(setting, index) for index in range(len(setting.choices)))
            else:
                rows.append(SettingsRow(setting, -1))
        return rows

    def _row_text(self, state: FormState, row: SettingsRow) -> str:
        field = row.field
        if field.kind == KIND_FLAGS:
            code = field.choices[row.index].value
            disabled = code in self._list_value(state, field.key)
            checked = " " if disabled else "x"
            return f"[{checked}] {code}"
        if field.kind == KIND_LIST:
            entries = self._list_value(state, field.key)
            if row.index < 0:
                if not entries:
                    return f"{field.label}: (empty; press a to add)"
                return f"+ add another {field.label.lower().rstrip('s')}"
            if row.index == 0:
                return f"{field.label}: {entries[row.index]}"
            return f"  {entries[row.index]}"
        value = state.values.get(field.key, "")
        if field.kind == KIND_PASSWORD:
            shown = "•" * len(str(value)) if value else "(unset)"
        else:
            shown = field.display_value(value) or "(unset)"
        return f"{field.label}: {shown}"

    def _handle_settings_key(self, key: Any) -> None:
        state = self.settings
        if state is None:
            return

        if state.editor is not None:
            if key == "\t":
                if state.completion is not None:
                    self._cycle_settings_completion(state, 1)
                else:
                    self._start_settings_completion(state)
                return
            if state.completion is not None:
                if key == curses.KEY_DOWN:
                    self._cycle_settings_completion(state, 1)
                    return
                if key == curses.KEY_UP:
                    self._cycle_settings_completion(state, -1)
                    return
                if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                    state.completion = None
                    return
                if key == "\x1b":
                    state.completion = None
                    return
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._commit_settings_edit(state)
                return
            if key == "\x1b":
                state.editor = None
                state.completion = None
                state.adding = False
                state.editing = None
                state.message = ""
                return
            state.editor.handle_key(key)
            state.completion = None
            return

        if key in {"\x1b", "q"}:
            self.settings = None
            return
        if key == "\t" or key == curses.KEY_BTAB:
            direction = -1 if key == curses.KEY_BTAB else 1
            index = (SETTINGS_SECTIONS.index(state.section) + direction) % len(SETTINGS_SECTIONS)
            state.section = SETTINGS_SECTIONS[index]
            state.selected = 0
            return
        if key == "s":
            self._save_settings(state)
            return

        rows = self._section_rows(state)
        if self._is_down_key(key):
            if rows:
                state.selected = min(len(rows) - 1, state.selected + 1)
            return
        if self._is_up_key(key):
            state.selected = max(0, state.selected - 1)
            return

        row = rows[state.selected] if 0 <= state.selected < len(rows) else None
        if row is None:
            return
        field = row.field

        if field.kind == KIND_LIST:
            if key in {"a", "+"}:
                self._begin_field_edit(state, field, adding=True)
                return
            if key in {"e", "\n", "\r", curses.KEY_ENTER, 10, 13}:
                entries = self._list_value(state, field.key)
                self._begin_field_edit(state, field, adding=not entries)
                return
            if key in {"d", curses.KEY_DC}:
                entries = self._list_value(state, field.key)
                if entries:
                    del entries[state.selected]
                    state.values[field.key] = entries
                    state.selected = max(0, min(state.selected, len(entries) - 1))
                    state.message = "removed; press s to save"
            return

        if field.kind == KIND_FLAGS:
            if key in {" ", "\n", "\r", curses.KEY_ENTER, 10, 13}:
                code = field.choices[row.index].value
                disabled = self._list_value(state, field.key)
                if code in disabled:
                    disabled.remove(code)
                else:
                    disabled.append(code)
                state.values[field.key] = disabled
                state.message = "press s to save"
            return

        if field.kind == KIND_CHOICE:
            if key in {" ", "e", "\n", "\r", curses.KEY_ENTER, 10, 13}:
                state.values[field.key] = cycle_choice(field, str(state.values.get(field.key, "")))
                state.message = "press s to save"
            return

        if key in {"e", "\n", "\r", curses.KEY_ENTER, 10, 13}:
            self._begin_field_edit(state, field, adding=False)

    def _begin_field_edit(
        self,
        state: FormState,
        field: SettingField,
        *,
        adding: bool,
        help_message: bool = True,
    ) -> None:
        if field.kind == KIND_LIST:
            entries = self._list_value(state, field.key)
            current = "~/" if adding else str(entries[state.selected])
        else:
            current = str(state.values.get(field.key, ""))
        state.editing = field
        state.adding = adding
        state.editor = self._new_editor(current, mode=MODE_INSERT)
        state.completion = None
        if help_message:
            state.message = field.help or ("new entry" if adding else "edit and press Enter")
        else:
            state.message = ""

    def _start_settings_completion(self, state: FormState) -> None:
        field = state.editing
        editor = state.editor
        if field is None or editor is None or field.kind not in (KIND_PATH, KIND_LIST):
            return
        if field.target == PATH_FILE:
            matches = complete_paths(editor.text, include_files=True)
        else:
            matches = complete_directories(editor.text)
        if not matches:
            state.completion = None
            return
        popup = CompletionPopup(start=0, end=len(editor.text), matches=matches, index=0)
        state.completion = popup
        self._apply_settings_completion(state)
        if len(matches) == 1:
            state.completion = None

    def _cycle_settings_completion(self, state: FormState, delta: int) -> None:
        popup = state.completion
        if popup is None or not popup.matches:
            return
        popup.index = (popup.index + delta) % len(popup.matches)
        self._apply_settings_completion(state)

    def _apply_settings_completion(self, state: FormState) -> None:
        popup = state.completion
        editor = state.editor
        if popup is None or editor is None:
            return
        match = popup.matches[popup.index]
        text = match.display
        editor.set_text(text, cursor=len(text))
        popup.start = 0
        popup.end = len(text)

    def _commit_settings_edit(self, state: SettingsState) -> None:
        field = state.editing
        editor = state.editor
        if field is None or editor is None:
            return
        state.completion = None
        raw = editor.text.strip()
        try:
            converted = self._convert_field_value(state, field, raw)
        except ValueError as exc:
            state.message = str(exc)
            return
        if field.kind == KIND_LIST:
            entries = self._list_value(state, field.key)
            if state.adding or not entries:
                entries.append(converted)
                state.selected = len(entries) - 1
            else:
                entries[state.selected] = converted
            state.values[field.key] = entries
        else:
            state.values[field.key] = converted
        state.editor = None
        state.editing = None
        state.adding = False
        state.message = "press s to save"

    def _convert_field_value(
        self,
        state: FormState,
        field: SettingField,
        raw: str,
    ) -> object:
        """Validate and convert an edited string into the field's value type."""
        if field.kind == KIND_INT:
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"{field.label} must be an integer") from None
            self._check_range(field, value)
            return value
        if field.kind == KIND_FLOAT:
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{field.label} must be a number") from None
            self._check_range(field, value)
            return value
        if field.kind in (KIND_PATH, KIND_LIST):
            return self._convert_path(state, field, raw)
        if not raw and field.empty_value:
            return field.empty_value
        return raw

    def _check_range(self, field: SettingField, value: float) -> None:
        if field.minimum is not None:
            if field.exclusive_min and value <= field.minimum:
                raise ValueError(f"{field.label} must be greater than {field.minimum:g}")
            if not field.exclusive_min and value < field.minimum:
                raise ValueError(f"{field.label} must be at least {field.minimum:g}")
        if field.maximum is not None and value > field.maximum:
            raise ValueError(f"{field.label} must be at most {field.maximum:g}")

    def _convert_path(self, state: FormState, field: SettingField, raw: str) -> str:
        if not raw:
            raise ValueError(f"{field.label} cannot be empty")
        candidate = Path(raw).expanduser()
        if field.key == "attachment_dir":
            root = Path(self.config.notes_root).resolve()
            resolved = (
                candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
            )
            try:
                return str(resolved.relative_to(root))
            except ValueError:
                return str(resolved)
        if field.must_exist:
            if field.target == PATH_DIR and not candidate.is_dir():
                raise ValueError(f"not a directory: {candidate}")
            if field.target == PATH_FILE and not candidate.is_file():
                raise ValueError(f"not a file: {candidate}")
        if field.kind == KIND_LIST:
            return str(candidate.resolve())
        return str(candidate)

    def _save_settings(self, state: SettingsState) -> None:
        raw = dict(state.values)
        file_values: dict[str, object] = {}
        overrides: dict[str, object] = {}
        for setting in ALL_FIELDS:
            if setting.key not in raw:
                continue
            value = raw[setting.key]
            if setting.kind == KIND_INT:
                typed: object = int(str(value))
            elif setting.kind == KIND_FLOAT:
                typed = float(str(value))
            elif setting.kind in (KIND_LIST, KIND_FLAGS):
                typed = list(value) if isinstance(value, list) else []
            else:
                typed = str(value)
            file_values[setting.key] = typed
            overrides[setting.key] = typed
        # Config stores these two as Path and does not re-normalize them in
        # __post_init__, so they need an explicit Path before with_overrides.
        overrides["notes_root"] = Path(str(raw["notes_root"])).expanduser()
        overrides["artifacts_dir"] = Path(str(raw["artifacts_dir"])).expanduser()
        try:
            write_config_values(file_values)
        except OSError as exc:
            state.message = f"cannot save: {exc}"
            return
        try:
            config = self.config.with_overrides(**overrides)
        except ConfigurationError as exc:
            state.message = f"invalid settings: {exc}"
            return
        self.config = config
        self.search_roots = list(config.search_paths)
        # Movement and layout are read live from the config; movement also
        # drives the input editor's modal behaviour, so rebuild it now.
        self.mode = self._base_mode()
        self.sync_editor_from_fields()
        self.settings = None

    # -- first-run wizard ---------------------------------------------------- #

    def open_setup_wizard(self) -> None:
        """Open the first-run setup wizard seeded from the current config."""
        self.active = None
        self.settings = None
        self.wizard = WizardState(values=wizard.default_values(self.config))
        self.editor.set_mode(self._base_mode())
        self._pull_editor()

    def _wizard_row_text(self, state: WizardState, row: SettingsRow) -> str:
        """One wizard row; list entries rely on the list header for their label."""
        field = row.field
        if field.kind == KIND_LIST:
            entries = self._list_value(state, field.key)
            if row.index < 0:
                return f"+ add another {field.label.lower().rstrip('s')}"
            return str(entries[row.index])
        return self._row_text(state, row)

    def _wizard_rows(self, state: WizardState) -> list[SettingsRow]:
        return self._rows_for_fields(
            state,
            wizard.visible_fields(state.page, state.values),
            add_list_row=True,
        )

    def _wizard_move_page(self, state: WizardState, delta: int) -> None:
        count = len(wizard.PAGES)
        state.page = wizard.PAGE_KEYS[(wizard.page_index(state.page) + delta) % count]
        state.selected = 0
        state.editor = None
        state.editing = None
        state.adding = False
        state.list_index = -1
        state.completion = None
        state.message = ""

    def _begin_wizard_list_edit(
        self, state: WizardState, field: SettingField, row: SettingsRow
    ) -> None:
        entries = self._list_value(state, field.key)
        if not entries or row.index < 0:
            self._begin_field_edit(state, field, adding=True, help_message=False)
            return
        current = str(entries[row.index])
        state.editing = field
        state.adding = False
        state.list_index = row.index
        state.editor = self._new_editor(current, mode=MODE_INSERT)
        state.completion = None
        state.message = ""

    def _commit_wizard_edit(self, state: WizardState) -> None:
        field = state.editing
        editor = state.editor
        if field is None or editor is None:
            return
        state.completion = None
        raw = editor.text.strip()
        try:
            converted = self._convert_field_value(state, field, raw)
        except ValueError as exc:
            state.message = str(exc)
            return
        if field.kind == KIND_LIST:
            entries = self._list_value(state, field.key)
            if state.adding or not 0 <= state.list_index < len(entries):
                entries.append(converted)
            else:
                entries[state.list_index] = converted
            state.values[field.key] = entries
        else:
            state.values[field.key] = converted
        state.editor = None
        state.editing = None
        state.adding = False
        state.list_index = -1
        state.message = ""

    def _save_wizard(self) -> None:
        """Write the wizard values, reload the config and record that setup ran."""
        state = self.wizard
        if state is None:
            return
        try:
            file_values, overrides = wizard.wizard_to_config(state.values)
        except ValueError as exc:
            state.message = str(exc)
            return
        notes_root = Path(str(state.values.get("notes_root", ""))).expanduser()
        try:
            notes_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            state.message = f"cannot create notes root: {exc}"
            return
        try:
            write_config_values(file_values)
        except OSError as exc:
            state.message = f"cannot save: {exc}"
            return
        try:
            config = self.config.with_overrides(**overrides)
        except ConfigurationError as exc:
            state.message = f"invalid settings: {exc}"
            return
        self.config = config
        self.search_roots = list(config.search_paths)
        self.mode = self._base_mode()
        self.sync_editor_from_fields()
        mark_setup_complete()
        self.wizard = None

    def _handle_wizard_key(self, key: Any) -> None:
        state = self.wizard
        if state is None:
            return

        if state.editor is not None:
            if key == "\t":
                if state.completion is not None:
                    self._cycle_settings_completion(state, 1)
                else:
                    self._start_settings_completion(state)
                return
            if state.completion is not None:
                if key == curses.KEY_DOWN:
                    self._cycle_settings_completion(state, 1)
                    return
                if key == curses.KEY_UP:
                    self._cycle_settings_completion(state, -1)
                    return
                if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                    state.completion = None
                    return
                if key == "\x1b":
                    state.completion = None
                    return
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._commit_wizard_edit(state)
                return
            if key == "\x1b":
                state.editor = None
                state.completion = None
                state.adding = False
                state.editing = None
                state.list_index = -1
                state.message = ""
                return
            state.editor.handle_key(key)
            state.completion = None
            return

        if key == "\x1b":
            state.message = "setup must be completed: Enter saves it on the review page"
            return
        if key == "\t" or key == "n":
            self._wizard_move_page(state, 1)
            return
        if key == curses.KEY_BTAB or key == "b":
            self._wizard_move_page(state, -1)
            return
        if state.page == wizard.PAGE_REVIEW:
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._save_wizard()
            return

        rows = self._wizard_rows(state)
        if key in {"j", curses.KEY_DOWN}:
            if rows:
                state.selected = min(len(rows) - 1, state.selected + 1)
            return
        if key in {"k", curses.KEY_UP}:
            state.selected = max(0, state.selected - 1)
            return
        row = rows[state.selected] if 0 <= state.selected < len(rows) else None
        if row is None:
            return
        field = row.field

        if field.kind == KIND_LIST:
            if key in {"a", "+"}:
                self._begin_field_edit(state, field, adding=True, help_message=False)
                return
            if key in {"e", "\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._begin_wizard_list_edit(state, field, row)
                return
            if key in {"d", curses.KEY_DC}:
                entries = self._list_value(state, field.key)
                if entries and row.index >= 0:
                    del entries[row.index]
                    state.values[field.key] = entries
                    state.selected = max(0, state.selected - 1)
                return
            return

        if field.kind == KIND_CHOICE:
            if key in {" ", "e", "\n", "\r", curses.KEY_ENTER, 10, 13}:
                state.values[field.key] = cycle_choice(field, str(state.values.get(field.key, "")))
                state.selected = min(state.selected, max(0, len(self._wizard_rows(state)) - 1))
            return

        if key in {"e", "\n", "\r", curses.KEY_ENTER, 10, 13}:
            self._begin_field_edit(state, field, adding=False, help_message=False)

    # -- modal handling ------------------------------------------------------ #

    def _handle_modal_key(self, key: Any) -> None:
        modal = self.modal
        if modal is None:
            return
        if modal.kind == "commands":
            self._handle_commands_key(modal, key)
            return
        if modal.kind == "history":
            self._handle_history_key(modal, key)
            return
        if modal.kind == "extras_export":
            self._handle_extras_export_key(modal, key)
            return
        if modal.kind == "moc":
            matches: list[MocMatch] = modal.payload.get("matches", [])
            rows = len(matches) + 2  # create a new MOC, or pick one by hand
            if self._is_down_key(key):
                modal.selected = min(rows - 1, modal.selected + 1)
            elif self._is_up_key(key):
                modal.selected = max(0, modal.selected - 1)
            elif key in {"c", "C", "+"}:
                self._start_moc_name(modal)
            elif key in {"m", "M"}:
                self._start_moc_manual(modal)
            elif key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                if modal.selected < len(matches):
                    self._answer_modal(MocDecision.use(matches[modal.selected].path))
                elif modal.selected == len(matches):
                    self._start_moc_name(modal)
                else:
                    self._start_moc_manual(modal)
            elif key in {"q", "Q", "\x1b"}:
                self._answer_modal(MocDecision.abort())
            return

        if modal.kind == "language":
            editor = modal.editor
            if editor is None:
                return
            default = str(modal.payload.get("default") or "en")
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._answer_modal(editor.text.strip() or default)
                return
            if key in {"\x1b", "\x03"}:
                self._answer_modal(default)
                return
            editor.handle_key(key)
            return

        if modal.kind == "moc_manual":
            editor = modal.editor
            if editor is None:
                return
            popup = modal.completion
            if key == "\t":
                if popup is not None:
                    self._cycle_moc_completion(modal, 1)
                else:
                    self._start_moc_completion(modal)
                return
            if popup is not None:
                if not self.vim_movements and key in {curses.KEY_DOWN, curses.KEY_UP}:
                    self._cycle_moc_completion(modal, 1 if key == curses.KEY_DOWN else -1)
                    return
                if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                    modal.completion = None
                    return
                if key == "\x1b":
                    modal.completion = None
                    return
                modal.completion = None
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._commit_manual_moc(modal)
                return
            if key == "\x1b":
                modal.editor = None
                modal.completion = None
                modal.message = ""
                modal.kind = "moc"
                return
            editor.handle_key(key)
            return

        if modal.kind == "moc_name":
            editor = modal.editor
            if editor is None:
                return
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                if not editor.text.strip():
                    modal.message = "the name cannot be empty"
                    return
                title = moc_title(editor.text, str(modal.payload.get("default_title") or ""))
                self._answer_modal(MocDecision.create(title))
                return
            if key in {"\x1b", "\x03"}:
                modal.editor = None
                modal.message = ""
                modal.kind = "moc"
                return
            editor.handle_key(key)
            return

        if modal.kind == "target":
            options = list(modal.payload.get("options") or [])
            if not options:
                self._answer_modal(None)
                return
            if self._is_down_key(key):
                modal.selected = min(len(options) - 1, modal.selected + 1)
            elif self._is_up_key(key):
                modal.selected = max(0, modal.selected - 1)
            elif key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._answer_modal(options[modal.selected])
            elif key in {"q", "Q", "\x1b"}:
                self._answer_modal(None)
            elif isinstance(key, str) and key.isdigit() and 1 <= int(key) <= len(options):
                self._answer_modal(options[int(key) - 1])
            return

    # -- drawing ------------------------------------------------------------- #

    def _draw(self) -> None:
        try:
            rows, cols = self.scr.getmaxyx()
        except curses.error:
            return
        self.scr.erase()
        if rows < 10 or cols < 40:
            self._addstr(0, 0, "terminal too small for scriptorium TUI", self._attr(STYLE_TITLE))
            self.scr.refresh()
            return

        if self.wizard is not None:
            self._draw_wizard(rows, cols)
            try:
                self.scr.refresh()
            except curses.error:
                pass
            return

        hint = " Ctrl-L logs · f full · c commands · s settings "
        hint_width = len(hint) + 1
        if cols > hint_width + 1:
            self._addstr(0, cols - len(hint) - 1, hint, self._attr(STYLE_DIM))
            status_width = cols - hint_width - 3
        else:
            status_width = cols - 2
        status = self._status_text()
        self._addstr(0, 1, _clip(status, status_width), self._attr(STYLE_HEADING))

        # Every box draws itself: the host only decides where it goes.
        rects = self.layout_rects()
        for name in self._draw_order(rects):
            pane = self.panes.get(name)
            rect = rects[name]
            if pane is None or rect.h < 3:
                continue
            pane.draw(self, rect)

        input_rect = rects.get(self._editor_name)
        if input_rect is not None and self.completion is not None:
            self._draw_completion(input_rect.y, rows, cols)
        if self.settings is not None:
            self._draw_settings(rows, cols)
        if self.modal is not None:
            self._draw_modal(rows, cols)

        show_input_cursor = (
            self.focus == self._editor_name
            and self.active == self._editor_name
            and (not self.vim_movements or self.mode == MODE_INSERT)
            and self.settings is None
            and self.modal is None
            and self.merge_review is None
        )
        settings_editor = self.settings is not None and self.settings.editor is not None
        modal_editor = self.modal is not None and self.modal.editor is not None
        self._apply_cursor_shape(show_input_cursor or settings_editor or modal_editor)
        if show_input_cursor:
            if input_rect is not None:
                self._place_cursor(input_rect.y, input_rect.x, input_rect.w, input_rect.h)
        elif not settings_editor and not modal_editor:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
        try:
            self.scr.refresh()
        except curses.error:
            pass

    def _draw_order(self, rects: dict[str, PaneRect]) -> list[str]:
        """Draw in reading order, then the carved panes on top of their parent."""
        order = list(layout_panes(self.config.tui_layout))
        for name in self.panes.carved():
            if name in rects:
                order.append(name)
        return [name for name in order if name in rects]

    def _apply_cursor_shape(self, insert: bool) -> None:
        """Ask the terminal for a bar (insert) or block (normal) cursor.

        This mirrors pivim's DECSCUSR handling.  Fakes used in tests do not
        expose the curses window internals and are silently skipped.
        """
        if not hasattr(self.scr, "getbkgd") or self._cursor_shape is insert:
            return
        self._cursor_shape = insert
        try:
            os.write(sys.stdout.fileno(), b"\x1b[6 q" if insert else b"\x1b[2 q")
        except (OSError, ValueError, AttributeError):
            pass

    def copy_text(self, text: str) -> bool:
        """Copy text to the system clipboard; False when no backend works."""
        return copy_to_clipboard(text)

    def open_extras_export(self) -> None:
        """Offer the configured roots and a free path where to write the extras."""
        if not self.asides:
            self.status = "no extras to write"
            return
        self.status = ""
        self.modal = PromptModal(
            kind="extras_export",
            payload={"roots": list(self.search_roots)},
            box={},
            event=threading.Event(),
        )
        self.editor.set_mode(self._base_mode())
        self._pull_editor()

    def _extras_export_roots(self, modal: PromptModal) -> list[Path]:
        return [Path(root) for root in modal.payload.get("roots") or []]

    def _start_extras_export_edit(self, modal: PromptModal) -> None:
        roots = self._extras_export_roots(modal)
        start = str(roots[0]) + os.sep if roots else str(Path.home()) + os.sep
        modal.editor = self._new_editor(start, mode=MODE_INSERT)
        modal.completion = None
        modal.message = "type a file path · Tab completes files and directories"

    def _handle_extras_export_key(self, modal: PromptModal, key: Any) -> None:
        if modal.editor is not None:
            if key == "\t":
                if modal.completion is not None:
                    self._cycle_extras_export_completion(modal, 1)
                else:
                    self._start_extras_export_completion(modal)
                return
            if modal.completion is not None:
                if key == curses.KEY_DOWN:
                    self._cycle_extras_export_completion(modal, 1)
                    return
                if key == curses.KEY_UP:
                    self._cycle_extras_export_completion(modal, -1)
                    return
                if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                    modal.completion = None
                    return
                if key == "\x1b":
                    modal.completion = None
                    return
                modal.completion = None
            if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
                self._commit_extras_export_path(modal)
                return
            if key == "\x1b":
                modal.editor = None
                modal.completion = None
                modal.message = ""
                return
            modal.editor.handle_key(key)
            modal.completion = None
            return

        roots = self._extras_export_roots(modal)
        if key in {"\x1b", "q", "Q"}:
            self.modal = None
            return
        if self._is_down_key(key):
            modal.selected = min(len(roots), modal.selected + 1)
            return
        if self._is_up_key(key):
            modal.selected = max(0, modal.selected - 1)
            return
        if key in {"n", "N", "+"}:
            self._start_extras_export_edit(modal)
            return
        if key in {"\n", "\r", curses.KEY_ENTER, 10, 13}:
            if modal.selected < len(roots):
                self._write_extras(modal, roots[modal.selected] / DEFAULT_EXPORT_NAME)
            else:
                self._start_extras_export_edit(modal)

    def _start_extras_export_completion(self, modal: PromptModal) -> None:
        editor = modal.editor
        if editor is None:
            return
        text = editor.text
        if text and not text.startswith("~") and not Path(text).expanduser().is_absolute():
            roots = self.search_roots or [Path.home()]
            text = str(roots[0]) + os.sep + text
        matches = complete_paths(text)
        if not matches:
            modal.completion = None
            return
        popup = CompletionPopup(start=0, end=len(editor.text), matches=matches, index=0)
        modal.completion = popup
        self._apply_extras_export_completion(modal)
        if len(matches) == 1:
            modal.completion = None

    def _cycle_extras_export_completion(self, modal: PromptModal, delta: int) -> None:
        popup = modal.completion
        if popup is None or not popup.matches:
            return
        popup.index = (popup.index + delta) % len(popup.matches)
        self._apply_extras_export_completion(modal)

    def _apply_extras_export_completion(self, modal: PromptModal) -> None:
        popup = modal.completion
        editor = modal.editor
        if popup is None or editor is None:
            return
        match = popup.matches[popup.index]
        text = match.display
        editor.set_text(text, cursor=len(text))
        popup.start = 0
        popup.end = len(text)

    def _commit_extras_export_path(self, modal: PromptModal) -> None:
        editor = modal.editor
        if editor is None:
            return
        modal.completion = None
        raw = editor.text.strip().strip("'\"")
        if not raw:
            modal.message = "the path is empty"
            return
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            roots = self.search_roots or [Path.home()]
            candidate = roots[0] / candidate
        if candidate.is_dir():
            candidate = candidate / DEFAULT_EXPORT_NAME
        if not candidate.parent.is_dir():
            modal.message = f"directory does not exist: {candidate.parent}"
            return
        self._write_extras(modal, candidate)

    def _write_extras(self, modal: PromptModal, target: Path) -> None:
        try:
            appended = write_extra_notes(target, self.asides)
        except (OSError, ValueError) as exc:
            modal.message = f"cannot write {target}: {exc}"
            return
        self.modal = None
        verb = "appended to" if appended else "wrote"
        self.status = f"{verb} {target} · {len(self.asides)} extras"

    def _floating_rect(
        self, rows: int, cols: int, width: int, height: int
    ) -> tuple[int, int, int, int]:
        """Clamp a floating window to half the terminal and centre it.

        Floating windows stay compact: they cover at most half of the
        terminal width and half of its height, so the panes underneath remain
        readable and a popup never looks like a full-screen page.
        """
        w = max(24, min(width, cols // 2, cols - 4))
        h = max(8, min(height, rows // 2, rows - 2))
        y = max(1, (rows - h) // 2)
        x = max(1, (cols - w) // 2)
        return y, x, h, w

    def _draw_completion(self, input_y: int, rows: int, cols: int) -> None:
        self._draw_completion_popup(self.completion, input_y, rows, cols)

    def _draw_completion_popup(
        self, popup: CompletionPopup | None, anchor_y: int, rows: int, cols: int
    ) -> None:
        if popup is None:
            return
        w = max(24, min(cols - 2, cols // 2))
        inner_w = max(1, w - 5)
        display: list[tuple[str, int]] = []
        for index, match in enumerate(popup.matches):
            marker = "› " if index == popup.index else "  "
            for piece in _wrap(marker + match.display, inner_w):
                display.append((piece, index))
        selected_display = 0
        for position, (_, index) in enumerate(display):
            if index == popup.index:
                selected_display = position
                break
        desired = min(len(display) + 2, 12)
        _, x, _, _ = self._floating_rect(rows, cols, w, desired)
        height = max(3, min(desired, rows // 2, anchor_y))
        y = max(0, anchor_y - height)
        self._fill_rect(y, x, height, w)
        title = "Commands" if popup.is_command else "Sources"
        verb = "run" if popup.is_command else "add"
        cycle = "Tab cycle" if self.vim_movements else "Tab/↑/↓ cycle"
        self._draw_box(
            y,
            x,
            height,
            w,
            f"{title} · {popup.index + 1}/{len(popup.matches)}",
            f"{cycle} · Enter {verb} · Esc close",
            True,
            [],
            0,
            0,
        )
        visible = max(1, height - 2)
        first = 0
        if selected_display >= visible:
            first = selected_display - visible + 1
        for offset in range(visible):
            index = first + offset
            if index >= len(display):
                break
            text, match_index = display[index]
            style = STYLE_SELECTED if match_index == popup.index else "normal"
            self._addstr(y + 1 + offset, x + 2, text, self._attr(style))

    def _draw_settings(self, rows: int, cols: int) -> None:
        state = self.settings
        if state is None:
            return
        w = max(28, min(cols - 4, (cols * 3) // 4))
        section_rows = self._section_rows(state)
        rows_needed = max(12, len(section_rows) + 9)
        if state.editor is not None:
            rows_needed += 1
        h = min(rows - 2, max(rows_needed, rows // 2))
        y = max(1, (rows - h) // 2)
        x = max(1, (cols - w) // 2)
        footer = (
            "Tab complete · Enter save · Esc cancel"
            if state.editor is not None
            else (
                "Tab section · e edit · space toggle/cycle · a add · d delete · s save · Esc cancel"
            )
        )
        # Fill the window before drawing it: a floating box must be opaque, so
        # the panes underneath never show through its interior.
        self._fill_rect(y, x, h, w)
        self._draw_box(y, x, h, w, "Settings", footer, True, [], 0, 0)

        # Sections live in a left column; the active one is highlighted and
        # the selected section's fields are drawn to its right.
        label_width = max(len(label) for label in SETTINGS_LABELS.values()) + 3
        left_w = max(12, min(label_width, (w * 2) // 5))
        separator_x = x + left_w + 2
        content_x = separator_x + 2
        content_w = max(1, w - left_w - 5)
        # A vertical rule makes the left column read as a section list rather
        # than as a second, unrelated list of values.
        for offset in range(1, h - 1):
            self._addstr(y + offset, separator_x, "│", self._attr(STYLE_DIM))
        self._addstr(
            y + 1,
            x + 2,
            _clip("Sections", left_w),
            self._attr(STYLE_DIM) | curses.A_BOLD,
        )
        for offset, section_key in enumerate(SETTINGS_SECTIONS):
            row = y + 2 + offset
            if row >= y + h - 1:
                break
            active = section_key == state.section
            marker = "› " if active else "  "
            self._addstr(
                row,
                x + 2,
                _clip(marker + SETTINGS_LABELS[section_key], left_w),
                self._attr(STYLE_SELECTED if active else STYLE_DIM),
            )

        list_y = y + 2
        editor_rows = 1 if state.editor is not None else 0
        available = max(1, h - 6 - editor_rows)
        display: list[tuple[str, int]] = []
        for index, row in enumerate(section_rows):
            marker = "› " if index == state.selected else "  "
            line = marker + self._row_text(state, row)
            for piece in _wrap(line, content_w):
                display.append((piece, index))

        if display:
            selected_display = 0
            for position, (_, index) in enumerate(display):
                if index == state.selected:
                    selected_display = position
                    break
            first = 0
            if selected_display >= available:
                first = selected_display - available + 1
            for offset in range(available):
                index = first + offset
                if index >= len(display):
                    break
                line, item_index = display[index]
                style = STYLE_SELECTED if item_index == state.selected else STYLE_DIM
                self._addstr(list_y + offset, content_x, line, self._attr(style))
        elif not section_rows:
            self._addstr(list_y, content_x, "  (empty)", self._attr(STYLE_REMOVED))

        if state.section == "warnings":
            hint_y = y + h - 4
            for offset, line in enumerate(_wrap("[x] = warning kept (recorded)", content_w)):
                if hint_y + offset >= y + h - 2:
                    break
                self._addstr(hint_y + offset, content_x, line, self._attr(STYLE_DIM))

        edit_y = 0
        field_x = content_x
        field_w = 1
        if state.editor is not None and state.editing is not None:
            edit_y = min(list_y + available, y + h - 2)
            label = "new: " if state.adding else "edit: "
            self._addstr(edit_y, content_x, label, self._attr(STYLE_TITLE))
            field_x = content_x + len(label)
            field_w = max(1, content_x + content_w - field_x)
            text = state.editor.text
            if state.editing.kind == KIND_PASSWORD:
                text = "•" * len(text)
            start = max(0, state.editor.cursor - field_w + 1)
            visible = text[start : start + field_w]
            self._addstr(edit_y, field_x, visible, self._attr(STYLE_HEADING))
            try:
                curses.curs_set(1)
            except curses.error:
                pass
        if state.message:
            message_lines = _wrap(state.message, content_w)
            start = max(y + 1, y + h - 1 - len(message_lines))
            for offset, line in enumerate(message_lines):
                row = start + offset
                if row >= y + h - 1:
                    break
                self._addstr(row, content_x, line, self._attr(STYLE_DIM))
        if state.completion is not None and state.editor is not None:
            self._draw_completion_popup(state.completion, edit_y, rows, cols)

        if state.editor is not None and state.editing is not None:
            # Park the hardware cursor last: every ``addstr`` above advances the
            # terminal cursor, so placing it earlier ends up on a message line
            # instead of the text being edited.
            start = max(0, state.editor.cursor - field_w + 1)
            try:
                self.scr.move(
                    edit_y,
                    field_x + min(field_w - 1, max(0, state.editor.cursor - start)),
                )
            except curses.error:
                pass

    def _draw_wizard(self, rows: int, cols: int) -> None:
        state = self.wizard
        if state is None:
            return
        field_rows = self._wizard_rows(state) if state.page != wizard.PAGE_REVIEW else []
        w = max(36, min(cols - 4, (cols * 4) // 5))
        height_needed = max(14, len(field_rows) + 12)
        if state.editor is not None:
            height_needed += 1
        h = min(rows - 2, max(height_needed, rows // 2))
        y = max(1, (rows - h) // 2)
        x = max(1, (cols - w) // 2)
        if state.page == wizard.PAGE_REVIEW:
            footer = "Enter save and start · b back"
        elif state.editor is not None:
            footer = "Tab complete · Enter accept · Esc cancel"
        else:
            footer = "Tab/n next · b back · Enter edit · space cycle"
        # The wizard owns the whole screen: fill it before drawing so the
        # boxes underneath never show through.
        self._fill_rect(y, x, h, w)
        self._draw_box(y, x, h, w, "Welcome to Scriptorium", footer, True, [], 0, 0)
        self._addstr(
            y + 1,
            x + 2,
            _clip("First-run setup", w - 4),
            self._attr(STYLE_HEADING),
        )

        # Page tabs, one per wizard page.
        tab_x = x + 2
        for index, page in enumerate(wizard.PAGES):
            label = f" {index + 1} {page.label} "
            style = STYLE_SELECTED if page.key == state.page else STYLE_DIM
            self._addstr(y + 2, tab_x, label, self._attr(style))
            tab_x += len(label) + 1

        content_x = x + 3
        content_w = max(1, w - 5)
        list_y = y + 4
        bottom = y + h - 2
        available = max(1, bottom - list_y)
        if state.editor is not None:
            available = max(1, available - 1)

        display: list[list[tuple[str, str]]] = []
        selected_line = 0
        if state.page == wizard.PAGE_REVIEW:
            for label, value in wizard.summary_rows(state.values):
                for piece in _wrap(f"{label}: {value}", content_w):
                    display.append([(piece, STYLE_DIM)])
        else:
            # While the editor is open the hardware cursor marks the typing
            # position; keep the rows plain so no reverse-video block looks
            # like a second cursor on the option being edited.
            editing = state.editor is not None and state.editing is not None
            described: set[str] = set()
            headers: set[str] = set()
            for index, row in enumerate(field_rows):
                field = row.field
                # A list gets one header and one description; its entries are
                # plain values below it, never ``Label: value`` on one line.
                if field.kind == KIND_LIST and field.key not in headers:
                    headers.add(field.key)
                    display.append([(field.label, STYLE_HEADING)])
                    help_text = field.help.strip()
                    if help_text:
                        described.add(field.key)
                        indent = "    "
                        for piece in _wrap(help_text, max(1, content_w - len(indent))):
                            display.append([(indent + piece, STYLE_DIM)])
                        display.append([])
                selected = index == state.selected
                style = STYLE_SELECTED if selected and not editing else STYLE_FOCUS
                marker = "› " if selected else "  "
                line_start = len(display)
                main_lines = _wrap(marker + self._wizard_row_text(state, row), content_w)
                display.extend([(piece, style)] for piece in main_lines)
                if selected:
                    selected_line = line_start
                # Scalar descriptions sit under their option, indented and
                # dimmed, followed by a blank line so options stay separate.
                help_text = field.help.strip()
                if field.kind != KIND_LIST and help_text and field.key not in described:
                    described.add(field.key)
                    indent = "    "
                    for piece in _wrap(help_text, max(1, content_w - len(indent))):
                        display.append([(indent + piece, STYLE_DIM)])
                    if index < len(field_rows) - 1:
                        display.append([])

        first = 0
        if selected_line >= available:
            first = selected_line - available + 1
        for offset in range(available):
            index = first + offset
            if index >= len(display):
                break
            column = 0
            for text, style in display[index]:
                self._addstr(
                    list_y + offset,
                    content_x + column,
                    text,
                    self._attr(style),
                )
                column += len(text)

        edit_y = 0
        field_x = content_x
        field_w = 1
        if state.editor is not None and state.editing is not None:
            edit_y = min(list_y + available, y + h - 2)
            label = "new: " if state.adding else "edit: "
            self._addstr(edit_y, content_x, label, self._attr(STYLE_TITLE))
            field_x = content_x + len(label)
            field_w = max(1, content_x + content_w - field_x)
            text = state.editor.text
            if state.editing.kind == KIND_PASSWORD:
                text = "•" * len(text)
            start = max(0, state.editor.cursor - field_w + 1)
            visible = text[start : start + field_w]
            self._addstr(edit_y, field_x, visible, self._attr(STYLE_HEADING))
            try:
                curses.curs_set(1)
            except curses.error:
                pass
        else:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            # Some terminals ignore the hide request; park the hardware cursor
            # outside the content so it can never sit on a description line.
            try:
                self.scr.move(rows - 1, cols - 1)
            except curses.error:
                pass

        if state.message:
            message_lines = _wrap(state.message, content_w)
            start = max(y + 1, y + h - 1 - len(message_lines))
            for offset, line in enumerate(message_lines):
                row = start + offset
                if row >= y + h - 1:
                    break
                self._addstr(row, content_x, line, self._attr(STYLE_DIM))
        if state.completion is not None and state.editor is not None:
            self._draw_completion_popup(state.completion, edit_y, rows, cols)

        if state.editor is not None and state.editing is not None:
            # Park the hardware cursor last: every ``addstr`` above advances the
            # terminal cursor, so placing it earlier ends up on a description or
            # popup line instead of the text being edited.
            start = max(0, state.editor.cursor - field_w + 1)
            try:
                self.scr.move(
                    edit_y,
                    field_x + min(field_w - 1, max(0, state.editor.cursor - start)),
                )
            except curses.error:
                pass

    def _draw_modal(self, rows: int, cols: int) -> None:
        modal = self.modal
        if modal is None:
            return
        if modal.kind == "commands":
            self._draw_commands_modal(modal, rows, cols)
        elif modal.kind == "history":
            self._draw_history_modal(modal, rows, cols)
        elif modal.kind == "moc":
            self._draw_moc_modal(modal, rows, cols)
        elif modal.kind == "extras_export":
            self._draw_extras_export_modal(modal, rows, cols)
        elif modal.kind == "moc_name":
            self._draw_moc_name_modal(modal, rows, cols)
        elif modal.kind == "moc_manual":
            self._draw_moc_manual_modal(modal, rows, cols)
        elif modal.kind == "target":
            self._draw_target_modal(modal, rows, cols)
        elif modal.kind == "language":
            self._draw_language_modal(modal, rows, cols)

    def _draw_language_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        editor = modal.editor
        if editor is None:
            return
        w = max(32, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        message_lines = _wrap(modal.message, inner_w)
        height = max(7, min(rows - 2, 7 + len(message_lines)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        self._draw_box(
            y,
            x,
            h,
            w,
            "Spoken language",
            "type a code, or detect · Enter accept · Esc default",
            True,
            [],
            0,
            0,
        )
        label = "Language: "
        field_x = x + 2 + len(label)
        field_w = max(1, w - (field_x - x) - 2)
        start = max(0, editor.cursor - field_w + 1)
        visible = editor.text[start : start + field_w]
        self._addstr(y + 1, x + 2, label, self._attr(STYLE_TITLE))
        self._addstr(y + 1, field_x, visible, self._attr(STYLE_HEADING))
        row = y + 3
        for line in message_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_DIM))
            row += 1
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            self.scr.move(y + 1, field_x + min(field_w - 1, max(0, editor.cursor - start)))
        except curses.error:
            pass

    def _draw_commands_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        commands: list[SlashCommand] = list(modal.payload.get("commands") or SLASH_COMMANDS)
        w = max(24, min(cols - 4, (cols * 2) // 3))
        inner_w = max(1, w - 4)
        display: list[tuple[str, int]] = []
        for index, command in enumerate(commands):
            marker = "› " if index == modal.selected else "  "
            display.append((f"{marker}{command.name}", index))
            for piece in _wrap("  " + command.description, inner_w):
                display.append((piece, index))
        needed = len(display) + 5
        height = max(8, min(rows - 2, max(10, needed)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        footer = (
            "j/k choose · Enter insert · Esc close"
            if self.vim_movements
            else "↑/↓ choose · Enter insert · Esc close"
        )
        self._draw_box(y, x, h, w, "Commands", footer, True, [], 0, 0)
        available = max(1, h - 2)
        selected_display = 0
        for position, (_, index) in enumerate(display):
            if index == modal.selected:
                selected_display = position
                break
        modal.scroll = max(0, min(modal.scroll, max(0, len(display) - available)))
        if selected_display < modal.scroll:
            modal.scroll = selected_display
        elif selected_display >= modal.scroll + available:
            modal.scroll = selected_display - available + 1
        for offset in range(available):
            index = modal.scroll + offset
            if index >= len(display):
                break
            text, command_index = display[index]
            style = STYLE_SELECTED if command_index == modal.selected else "normal"
            self._addstr(y + 1 + offset, x + 2, text, self._attr(style))

    def _draw_history_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        entries: list[HistoryEntry] = list(modal.payload.get("entries") or [])
        w = max(24, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        display: list[tuple[str, int]] = []
        if not entries:
            display.append(("No completed runs yet.", 0))
        else:
            for index, entry in enumerate(entries):
                marker = "› " if index == modal.selected else "  "
                for piece in _wrap(f"{marker}{index + 1}. {entry.label}", inner_w):
                    display.append((piece, index))
                for piece in _wrap("     " + entry.summary, inner_w):
                    display.append((piece, index))
        needed = len(display) + 5
        height = max(8, min(rows - 2, max(10, needed)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        footer = (
            "j/k choose · Enter restore · Esc close"
            if self.vim_movements
            else "↑/↓ choose · Enter restore · Esc close"
        )
        self._draw_box(y, x, h, w, "Run history", footer, True, [], 0, 0)
        if not entries:
            self._addstr(y + 1, x + 2, display[0][0], self._attr(STYLE_DIM))
            return
        available = max(1, h - 2)
        selected_display = 0
        for position, (_, index) in enumerate(display):
            if index == modal.selected:
                selected_display = position
                break
        modal.scroll = max(0, min(modal.scroll, max(0, len(display) - available)))
        if selected_display < modal.scroll:
            modal.scroll = selected_display
        elif selected_display >= modal.scroll + available:
            modal.scroll = selected_display - available + 1
        for offset in range(available):
            index = modal.scroll + offset
            if index >= len(display):
                break
            text, entry_index = display[index]
            style = STYLE_SELECTED if entry_index == modal.selected else "normal"
            self._addstr(y + 1 + offset, x + 2, text, self._attr(style))

    def _draw_extras_export_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        roots = self._extras_export_roots(modal)
        w = max(24, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        hint_lines = _wrap(
            "Choose a configured root or write to a new path. An existing file is appended to.",
            inner_w,
        )
        entries = [str(root / DEFAULT_EXPORT_NAME) for root in roots]
        entries.append("write to a new path…")
        display: list[tuple[str, int]] = []
        for index, entry in enumerate(entries):
            marker = "› " if index == modal.selected else "  "
            for piece in _wrap(marker + entry, inner_w):
                display.append((piece, index))
        message_lines = _wrap(modal.message, inner_w) if modal.message else []
        needed = len(hint_lines) + len(display) + len(message_lines) + 7
        height = max(10, min(rows - 2, max(12, needed)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        footer = (
            "Tab complete · Enter write · Esc back"
            if modal.editor is not None
            else "j/k choose · Enter write · n new path · Esc close"
            if self.vim_movements
            else "↑/↓ choose · Enter write · n new path · Esc close"
        )
        self._draw_box(
            y,
            x,
            h,
            w,
            "Write Extras to a file",
            footer,
            True,
            [],
            0,
            0,
        )
        row = y + 1
        for line in hint_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_DIM))
            row += 1
        list_y = row + 1
        bottom = y + h - 1
        editor_rows = 1 if modal.editor is not None else 0
        available = max(1, bottom - list_y - editor_rows - len(message_lines))
        selected_display = 0
        for position, (_, index) in enumerate(display):
            if index == modal.selected:
                selected_display = position
                break
        modal.scroll = max(0, min(modal.scroll, max(0, len(display) - available)))
        if selected_display < modal.scroll:
            modal.scroll = selected_display
        elif selected_display >= modal.scroll + available:
            modal.scroll = selected_display - available + 1
        for offset in range(available):
            index = modal.scroll + offset
            if index >= len(display):
                break
            text, entry_index = display[index]
            style = STYLE_SELECTED if entry_index == modal.selected else "normal"
            self._addstr(list_y + offset, x + 2, text, self._attr(style))

        edit_y = 0
        if modal.editor is not None:
            edit_y = min(list_y + available, y + h - 2)
            label = "Path: "
            field_x = x + 2 + len(label)
            field_w = max(1, w - (field_x - x) - 2)
            start = max(0, modal.editor.cursor - field_w + 1)
            visible = modal.editor.text[start : start + field_w]
            self._addstr(edit_y, x + 2, label, self._attr(STYLE_TITLE))
            self._addstr(edit_y, field_x, visible, self._attr(STYLE_HEADING))
            try:
                curses.curs_set(1)
            except curses.error:
                pass
            try:
                self.scr.move(
                    edit_y,
                    field_x + min(field_w - 1, max(0, modal.editor.cursor - start)),
                )
            except curses.error:
                pass
        message_y = edit_y + 2 if modal.editor is not None else y + h - 2
        for offset, line in enumerate(message_lines):
            if message_y + offset >= y + h - 1:
                break
            self._addstr(message_y + offset, x + 2, line, self._attr(STYLE_DIM))
        if modal.completion is not None and modal.editor is not None:
            self._draw_completion_popup(modal.completion, edit_y, rows, cols)

    def _draw_moc_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        matches: list[MocMatch] = modal.payload.get("matches", [])
        course = str(modal.payload.get("course") or "")
        w = max(24, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        course_lines = _wrap(f"Course: {course}", inner_w)
        entries = [f"{match.path}  (score {match.score:.2f})" for match in matches]
        entries.append(f"create a new MOC ({modal.payload.get('default_title') or 'new MOC'})")
        entries.append("select an existing MOC by name or path")
        display: list[tuple[str, int]] = []
        for index, entry in enumerate(entries):
            marker = "› " if index == modal.selected else "  "
            for piece in _wrap(marker + entry, inner_w):
                display.append((piece, index))
        height = max(8, min(rows - 2, max(12, len(course_lines) + len(display) + 6)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        self._draw_box(
            y,
            x,
            h,
            w,
            "Select the course MOC",
            "j/k choose · Enter confirm · c create · m manual · q abort"
            if self.vim_movements
            else "↑/↓ choose · Enter confirm · c create · m manual · q abort",
            True,
            [],
            0,
            0,
        )
        row = y + 1
        for line in course_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_HEADING))
            row += 1
        list_y = row + 1
        bottom = y + h - 1
        available = max(0, bottom - list_y)
        if available <= 0:
            return
        selected_display = 0
        for position, (_, index) in enumerate(display):
            if index == modal.selected:
                selected_display = position
                break
        modal.scroll = max(0, min(modal.scroll, max(0, len(display) - available)))
        if selected_display < modal.scroll:
            modal.scroll = selected_display
        elif selected_display >= modal.scroll + available:
            modal.scroll = selected_display - available + 1
        for offset in range(available):
            index = modal.scroll + offset
            if index >= len(display):
                break
            text, entry_index = display[index]
            style = STYLE_SELECTED if entry_index == modal.selected else "normal"
            self._addstr(list_y + offset, x + 2, text, self._attr(style))

    def _draw_moc_name_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        editor = modal.editor
        if editor is None:
            return
        course = str(modal.payload.get("course") or "")
        default = str(modal.payload.get("default_title") or "")
        title = moc_title(editor.text or default or course, default)
        w = max(24, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        course_lines = _wrap(f"Course: {course}", inner_w)
        final_lines = _wrap(f"Final name: {title}", inner_w)
        message_lines = _wrap(modal.message, inner_w) if modal.message else []
        needed = len(course_lines) + len(final_lines) + len(message_lines) + 7
        height = max(10, min(rows - 2, max(12, needed)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        self._draw_box(
            y,
            x,
            h,
            w,
            "Create a course MOC",
            "type the name · Enter create · Esc back",
            True,
            [],
            0,
            0,
        )
        row = y + 1
        for line in course_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_HEADING))
            row += 1
        edit_y = min(row + 1, y + h - 2)
        label = "Name: "
        field_x = x + 2 + len(label)
        field_w = max(1, w - (field_x - x) - 2)
        start = max(0, editor.cursor - field_w + 1)
        visible = editor.text[start : start + field_w]
        self._addstr(edit_y, x + 2, label, self._attr(STYLE_TITLE))
        self._addstr(edit_y, field_x, visible, self._attr(STYLE_HEADING))
        row = edit_y + 2
        for line in final_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_DIM))
            row += 1
        for line in message_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_DIM))
            row += 1
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            self.scr.move(edit_y, field_x + min(field_w - 1, max(0, editor.cursor - start)))
        except curses.error:
            pass

    def _draw_moc_manual_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        editor = modal.editor
        if editor is None:
            return
        course = str(modal.payload.get("course") or "")
        w = max(24, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        course_lines = _wrap(f"Course: {course}", inner_w)
        hint_lines = _wrap(
            "Type a MOC name, or a path relative to the vault. "
            "Tab completes the MOC notes found in the configured roots.",
            inner_w,
        )
        message_lines = _wrap(modal.message, inner_w) if modal.message else []
        needed = len(course_lines) + len(hint_lines) + len(message_lines) + 7
        height = max(10, min(rows - 2, max(12, needed)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        self._draw_box(
            y,
            x,
            h,
            w,
            "Select an existing MOC",
            "type name/path · Tab completion · Enter select · Esc back",
            True,
            [],
            0,
            0,
        )
        row = y + 1
        for line in course_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_HEADING))
            row += 1
        for line in hint_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_DIM))
            row += 1
        edit_y = min(row + 1, y + h - 2)
        label = "MOC: "
        field_x = x + 2 + len(label)
        field_w = max(1, w - (field_x - x) - 2)
        start = max(0, editor.cursor - field_w + 1)
        visible = editor.text[start : start + field_w]
        self._addstr(edit_y, x + 2, label, self._attr(STYLE_TITLE))
        self._addstr(edit_y, field_x, visible, self._attr(STYLE_HEADING))
        row = edit_y + 2
        for line in message_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_DIM))
            row += 1
        if modal.completion is not None and modal.editor is not None:
            self._draw_completion_popup(modal.completion, edit_y, rows, cols)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        try:
            self.scr.move(edit_y, field_x + min(field_w - 1, max(0, editor.cursor - start)))
        except curses.error:
            pass

    def _draw_target_modal(self, modal: PromptModal, rows: int, cols: int) -> None:
        options = list(modal.payload.get("options") or [])
        title = next(
            (option.title for option in options if getattr(option, "kind", "") == "create"),
            "note",
        )
        w = max(24, min(cols - 4, cols // 2))
        inner_w = max(1, w - 4)
        question_lines = _wrap(f"Where should '{title}' go?", inner_w)
        display: list[tuple[str, int]] = []
        for index, option in enumerate(options):
            score = f" (score {option.score:.2f})" if option.score else ""
            for piece in _wrap(f"{index + 1}. {option.label}{score}", inner_w):
                display.append((piece, index))
        height = max(8, min(rows - 2, max(12, len(question_lines) + len(display) + 6)))
        y, x, h, w = self._floating_rect(rows, cols, w, height)
        self._fill_rect(y, x, h, w)
        self._draw_box(
            y,
            x,
            h,
            w,
            "Choose a target note",
            "j/k move · 1-9 choose · enter select · q keep a new note"
            if self.vim_movements
            else "↑/↓ move · 1-9 choose · enter select · q keep a new note",
            True,
            [],
            0,
            0,
        )
        row = y + 1
        for line in question_lines:
            if row >= y + h - 1:
                break
            self._addstr(row, x + 2, line, self._attr(STYLE_HEADING))
            row += 1
        list_y = row + 1
        bottom = y + h - 1
        available = max(0, bottom - list_y)
        if available <= 0:
            return
        selected_display = 0
        for position, (_, index) in enumerate(display):
            if index == modal.selected:
                selected_display = position
                break
        modal.scroll = max(0, min(modal.scroll, max(0, len(display) - available)))
        if selected_display < modal.scroll:
            modal.scroll = selected_display
        elif selected_display >= modal.scroll + available:
            modal.scroll = selected_display - available + 1
        for offset in range(available):
            index = modal.scroll + offset
            if index >= len(display):
                break
            text, option_index = display[index]
            style = STYLE_SELECTED if option_index == modal.selected else "normal"
            self._addstr(list_y + offset, x + 2, text, self._attr(style))

    # -- rendering helpers --------------------------------------------------- #

    @property
    def painter(self) -> Painter:
        """The drawing surface every pane uses: one screen, one colour map."""
        return Painter(self.scr, self._colors)

    def _attr(self, style: str) -> int:
        return self.painter.attr(style)

    def _addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        self.painter.addstr(y, x, text, attr)

    def _fill_rect(self, y: int, x: int, h: int, w: int) -> None:
        self.painter.fill_rect(y, x, h, w)

    def _draw_box(
        self,
        y: int,
        x: int,
        h: int,
        w: int,
        title: str,
        footer: str,
        focused: bool,
        rows: list[tuple[str, str]],
        scroll: int,
        selected_row: int,
        active: bool = False,
    ) -> int:
        return self.painter.draw_box(
            y, x, h, w, title, footer, focused, rows, scroll, selected_row, active
        )

    # -- pane helpers -------------------------------------------------------- #

    def pane(self, name: str) -> Pane | None:
        """The registered box called ``name``, if any."""
        return self.panes.get(name)

    def _place_cursor(self, y: int, x: int, w: int, h: int) -> None:
        pane = self.panes.get(self._editor_name)
        if pane is not None:
            pane.place_cursor(self, y, x, w, h)

    # -- compatibility aliases ----------------------------------------------- #
    # The host talks to the public names; these keep the historical private
    # names working for callbacks and tests written against the old layout.

    _layout_rects = layout_rects
    _move_focus = move_focus
    _set_focus = set_focus
    _cycle_focus = cycle_focus
    _toggle_fullscreen = toggle_fullscreen
    _request_quit = request_quit
    _show_help = show_help
    _open_settings = open_settings
    _open_commands = open_commands
    _refresh_completion = refresh_completion
    _start_job = start_job
    _submit_input = submit_input
    _start_completion = start_completion
    _cycle_completion = cycle_completion
    _sync_editor_from_fields = sync_editor_from_fields
    _pull_editor = pull_editor
    _new_editor = new_editor
    _open_extras_export = open_extras_export
    _answer_merge = answer_merge
    _stage_text = stage_text
    _activate_box = activate_box


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def run_tui(config: Config, *, verbose: bool = False, show_setup: bool = False) -> int:
    """Run the interactive TUI until the user quits.

    ``show_setup`` opens the first-run configuration wizard before the normal
    interface; the CLI decides that from ``config.is_first_run()``.
    """
    app = TuiApp(config=config, verbose=verbose)
    if show_setup:
        app.open_setup_wizard()
    return app.run()
