"""The pane contract: one box is one module exposing one :class:`Pane`.

A pane module is a self-contained extension.  It imports this contract, the
rendering primitives and the data it displays -- never ``tui.py`` -- so a box
can be written, tested and shipped in a single file and selected from
``config.toml``::

    [tui_panes]
    weather = "my_panes.weather:WeatherPane"
    tui_layout = ["sources | changes | weather", "input"]

The host owns the layout, the focus ring, the key routing and the shared
application state; a pane owns its box: its title, its rows, its selection and
scroll position, and the keys pressed while it is focused or selected.

This module is the public surface for writing a pane: everything a box needs is
re-exported here (the base class, the host services, the geometry types, the
style names and the text helpers), so a custom pane imports from one place and
never reaches into the host internals.  See ``docs/CUSTOM_PANES.md`` for the
full recipe.

Geometry metadata decides where a pane lives:

``placeable``
    True (the default) means the pane name may appear in ``tui_layout``.
``parent``
    When set, the pane is not placed by the layout: it is carved out of the
    named pane's rectangle.  ``side`` and ``split_ratio`` say how.  A pane with
    no parent is placed directly in ``tui_layout``.
``transient``
    The pane is only carved while :meth:`Pane.visible` returns True, so it can
    be toggled at run time without touching the layout.

Capabilities tell the host how to treat a box without the host knowing its
name:

``is_editor``
    True for the single box that edits text (the input line).  The host routes
    the editor, the hardware cursor and the insert/normal mode to it.
``owns_merge_review``
    True for the box that renders and answers a pending merge review.
"""

from __future__ import annotations

import curses
from typing import Any, ClassVar, Protocol, runtime_checkable

from .tui_layout import PaneRect
from .tui_render import (
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
    Row,
    clip,
    diff_style,
    one_line,
    wrap,
)

MODE_INSERT = "insert"
MODE_NORMAL = "normal"

ENTER_KEYS = frozenset({"\n", "\r", curses.KEY_ENTER, 10, 13})
PAGE_DOWN_KEYS = frozenset({curses.KEY_NPAGE, "\x04"})
PAGE_UP_KEYS = frozenset({curses.KEY_PPAGE, "\x15"})


@runtime_checkable
class PaneHost(Protocol):
    """The services a pane may use, implemented by ``tui.TuiApp``.

    Everything here is either read-only application state produced by the
    pipeline or a generic action that crosses box boundaries (starting a job,
    moving the focus, opening an overlay, copying text).  A pane never reaches
    past this surface, and the host never branches on a pane name outside the
    geometry solver.
    """

    config: Any
    focus: str
    active: str | None
    fullscreen: str | None
    status: str
    completion: Any
    merge_review: Any
    settings: Any
    modal: Any

    # Shared session state: written by the pipeline, read by the panes.
    source_paths: list[Any]
    source_index: int
    source_scroll: int
    moc_hint: Any
    diff_entries: list[Any]
    diff_index: int
    diff_scroll: int
    card_groups: list[Any]
    card_index: int
    card_scroll: int
    asides: list[Any]
    extras_scroll: int
    log_lines: list[str]
    log_scroll: int

    # Input line state, owned by the input pane but shared with the host.
    input_text: str
    cursor: int
    mode: str
    editor: Any

    @property
    def vim_movements(self) -> bool: ...

    @property
    def painter(self) -> Any: ...

    @property
    def running(self) -> bool: ...

    @property
    def current_stage(self) -> str: ...

    def stage_text(self) -> str: ...

    def layout_rects(self) -> dict[str, PaneRect]: ...

    def pane_shown(self, name: str) -> bool: ...

    # Cross-box actions.
    def set_focus(self, target: str) -> None: ...

    def activate_box(self) -> None: ...

    def move_focus(self, direction: str) -> None: ...

    def cycle_focus(self, delta: int) -> None: ...

    def toggle_fullscreen(self) -> None: ...

    def toggle_pane(self, name: str) -> None: ...

    def request_quit(self) -> None: ...

    def show_help(self) -> None: ...

    def open_settings(self) -> None: ...

    def answer_merge(self, response: str) -> None: ...

    def copy_text(self, text: str) -> bool: ...

    def start_job(self) -> None: ...

    def submit_input(self) -> None: ...

    def start_completion(self) -> None: ...

    def refresh_completion(self) -> None: ...

    def cycle_completion(self, delta: int) -> None: ...

    def refresh_command_completion(self) -> None: ...

    def accept_command_completion(self) -> None: ...

    def accept_source_completion(self) -> None: ...

    def open_commands(self) -> None: ...

    def sync_editor_from_fields(self) -> None: ...

    def pull_editor(self) -> None: ...

    def new_editor(self, text: str = "", *, cursor: int | None = None, mode: str = "") -> Any: ...

    def open_extras_export(self) -> None: ...


class Pane:
    """Base class for one box.

    Only :meth:`title` and :meth:`rows` are required.  The selection and scroll
    hooks default to a stateless pane, and :meth:`handle_key` implements the
    list behaviour (``j``/``k``, ``Enter``, ``g``/``G``) that most boxes want.
    """

    #: The name used by ``tui_layout`` and ``[tui_panes]``.
    name: ClassVar[str] = ""
    #: True when the layout may place this pane directly.
    placeable: ClassVar[bool] = True
    #: Pane whose rectangle this one is carved out of, if any.
    parent: ClassVar[str | None] = None
    #: Which edge of the parent the carve-out starts from.
    side: ClassVar[str] = "bottom"
    #: True when the pane only exists while :meth:`visible` says so.
    transient: ClassVar[bool] = False
    #: True when the pane may take over a fullscreen pane's area.
    splits_fullscreen: ClassVar[bool] = False
    #: True when the pane replaces its parent instead of leaving it too small.
    collapse: ClassVar[bool] = False
    #: Key that toggles the pane; None means it is always visible while placed.
    spawn_key: ClassVar[str | None] = None
    #: True when this box is the editor the host routes text input to.
    is_editor: ClassVar[bool] = False
    #: True when this box renders and answers the pending merge review.
    owns_merge_review: ClassVar[bool] = False

    # -- geometry ------------------------------------------------------------ #

    def visible(self, host: PaneHost) -> bool:
        """Whether a transient pane should be carved right now."""
        return True

    def split_ratio(self, host: PaneHost) -> float:
        """Fraction of the parent rectangle this pane takes."""
        return 0.5

    # -- content ------------------------------------------------------------- #

    def title(self, host: PaneHost) -> str:
        """The label drawn on the top border."""
        raise NotImplementedError

    def footer(self, host: PaneHost) -> str:
        """The hint drawn on the bottom border; empty hides it."""
        return ""

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        """Return ``(rows, selected_row)``; ``selected_row`` is -1 for none."""
        raise NotImplementedError

    def has_selection(self) -> bool:
        """True when the box has a movable selection."""
        return True

    # -- selection and scroll ------------------------------------------------- #

    def item_count(self, host: PaneHost) -> int:
        return 0

    def selection(self, host: PaneHost) -> int:
        return 0

    def set_selection(self, host: PaneHost, index: int) -> None:
        """Store a clamped selection index."""

    def toggle_item(self, host: PaneHost, index: int) -> None:
        """Act on ``Enter``: expand a group, open a note, and so on."""

    def scroll(self, host: PaneHost) -> int:
        return 0

    def scroll_limit(self, host: PaneHost) -> int:
        """Highest useful scroll offset for a read-only pane."""
        return 0

    def set_scroll(self, host: PaneHost, value: int) -> None:
        """Store a clamped scroll offset."""

    # -- drawing -------------------------------------------------------------- #

    def draw(self, host: PaneHost, rect: PaneRect) -> None:
        """Draw the box inside ``rect``."""
        rows, selected = self.rows(host)
        scroll = host.painter.draw_box(
            rect.y,
            rect.x,
            rect.h,
            rect.w,
            self.title(host),
            self.footer(host),
            host.focus == self.name,
            rows,
            self.scroll(host),
            selected,
            active=host.active == self.name,
        )
        if self.has_selection() and host.focus == self.name:
            self.set_scroll(host, scroll)

    # -- key handling --------------------------------------------------------- #

    def handle_key(self, host: PaneHost, key: Any, *, active: bool) -> bool:
        """Handle one key; return True when the pane consumed it."""
        if not self.has_selection():
            return False
        vim = host.vim_movements
        if key == "g":
            self.jump(host, first=True)
            return True
        if key == "G":
            self.jump(host, first=False)
            return True
        if not active:
            return False
        if key == "j" or (not vim and key == curses.KEY_DOWN):
            self.move(host, 1)
            return True
        if key == "k" or (not vim and key == curses.KEY_UP):
            self.move(host, -1)
            return True
        if key in ENTER_KEYS or key == " ":
            self.toggle_item(host, self.selection(host))
            return True
        if key == curses.KEY_HOME:
            self.jump(host, first=True)
            return True
        if key == curses.KEY_END:
            self.jump(host, first=False)
            return True
        return False

    # -- selection helpers ---------------------------------------------------- #

    def move(self, host: PaneHost, delta: int) -> None:
        count = self.item_count(host)
        if count <= 0:
            return
        self.set_selection(host, max(0, min(count - 1, self.selection(host) + delta)))

    def jump(self, host: PaneHost, *, first: bool) -> None:
        count = self.item_count(host)
        if count <= 0:
            return
        self.set_selection(host, 0 if first else count - 1)

    # -- session lifecycle ---------------------------------------------------- #

    def reset(self, host: PaneHost) -> None:
        """Clear view state so ``/new`` starts from an empty box."""
        self.set_selection(host, 0)
        self.set_scroll(host, 0)

    def on_shown(self, host: PaneHost) -> None:
        """Called after the host toggles this pane into view."""

    def on_log(self, host: PaneHost) -> None:
        """Called after the host appends lines to the shared log buffer."""

    def on_asides(self, host: PaneHost) -> None:
        """Called after the host reloads the shared asides."""

    # -- page helpers --------------------------------------------------------- #

    def place_cursor(self, host: PaneHost, y: int, x: int, w: int, h: int) -> None:
        """Put the hardware cursor on the editing position (input panes only)."""

    def page(self, host: PaneHost, rect: PaneRect | None) -> int:
        """Visible content rows, used by ``PgUp``/``PgDn`` in scrollable panes."""
        return max(1, (rect.h - 3) if rect is not None else 1)


__all__ = [
    "Pane",
    "PaneHost",
    "PaneRect",
    "Row",
    "MODE_INSERT",
    "MODE_NORMAL",
    "ENTER_KEYS",
    "PAGE_DOWN_KEYS",
    "PAGE_UP_KEYS",
    "STYLE_TITLE",
    "STYLE_ADDED",
    "STYLE_REMOVED",
    "STYLE_HUNK",
    "STYLE_META",
    "STYLE_HEADING",
    "STYLE_DIM",
    "STYLE_SELECTED",
    "STYLE_ACTIVE",
    "STYLE_ACTIVE_BORDER",
    "STYLE_FOCUS",
    "STYLE_MODE_STANDARD",
    "STYLE_MODE_NORMAL",
    "STYLE_MODE_INSERT",
    "one_line",
    "clip",
    "wrap",
    "diff_style",
]
