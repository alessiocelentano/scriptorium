# Writing a custom pane

A pane is the smallest unit of the scriptorium TUI: one box, one file.  A pane
module is a self-contained extension that owns how its box renders, how it
behaves, and the keys it handles.  The host (`tui.py`) owns the layout, the
focus ring, the key routing and the shared application state; it never branches
on a pane name.

This document is the contract for writing a pane.  Read it end to end before
implementing one, and follow `src/scriptorium/panes/*.py` as the reference
implementations of the same contract.

## The two rules

1. **A pane imports the contract, never the host.**  Import from
   `scriptorium.pane_api` (and `scriptorium.tui_render` if you need a helper
   that is not re-exported).  A pane module must never import `tui.py`.
   `tests/test_panes.py` enforces this.
2. **Core never imports a pane.**  The pipeline is shared by the CLI and the
   TUI, so `pipeline.py`, `planning.py`, `vault.py` and friends stay
   pane-agnostic.  A pane may import core modules (models, state, sources,
   extra_notes, ...) to format the data it displays, but never the reverse.

## One import surface

Everything a box needs is re-exported from `scriptorium.pane_api`:

```python
from scriptorium.pane_api import (
    Pane,          # the base class you subclass
    PaneHost,      # the services your box may use (a Protocol, not a class)
    PaneRect,      # rectangle in terminal cells: y, x, h, w
    Row,           # one content row: (text, style)
    MODE_INSERT, MODE_NORMAL,
    ENTER_KEYS, PAGE_DOWN_KEYS, PAGE_UP_KEYS,
    STYLE_DIM, STYLE_HEADING, STYLE_SELECTED,  # ... and the other STYLE_* names
    one_line, clip, wrap, diff_style,
)
```

Styles are names, not curses attributes; `host.painter` resolves them.  The
names available are `STYLE_TITLE`, `STYLE_ADDED`, `STYLE_REMOVED`, `STYLE_HUNK`,
`STYLE_META`, `STYLE_HEADING`, `STYLE_DIM`, `STYLE_SELECTED`, `STYLE_ACTIVE`,
`STYLE_ACTIVE_BORDER`, `STYLE_FOCUS`, `STYLE_MODE_STANDARD`, `STYLE_MODE_NORMAL`,
`STYLE_MODE_INSERT`.

## The minimal pane

Only `title` and `rows` are required.  A read-only, non-selectable box is a
handful of lines:

```python
"""A minimal read-only box."""

from scriptorium.pane_api import Pane, PaneHost, STYLE_DIM


class WeatherPane(Pane):
    name = "weather"

    def title(self, host: PaneHost) -> str:
        return "Weather"

    def rows(self, host: PaneHost) -> tuple[list[tuple[str, str]], int]:
        return [("sunny", "normal"), ("18 C", STYLE_DIM)], -1

    def has_selection(self) -> bool:
        return False
```

Register it in `config.toml` and place it in the layout:

```toml
tui_layout = [
  "sources | changes | weather",
  "input",
]

[tui_panes]
weather = "my_panes.weather:WeatherPane"
```

The `tui_layout` key must stay at the top level: a bare key written after a
`[table]` header belongs to that table.

The import target is `"module"` (the module exposes the pane as `Pane`) or
`"module:ClassName"`.  The pane's `name` must equal its `[tui_panes]` key; a
mismatch, a missing class, a class that is not a `Pane`, or a module that cannot
be imported is a configuration error raised at start-up.  The package must be
importable by the interpreter that runs scriptorium (installed in the same
environment, or on `PYTHONPATH`).

## Required vs optional hooks

| Hook | Required | Purpose |
|---|---|---|
| `name` | yes | The `tui_layout`/`[tui_panes]` key. |
| `title(host)` | yes | Label drawn on the top border. |
| `rows(host)` | yes | `(rows, selected_row)`; `selected_row` is -1 for none. |
| `footer(host)` | no | Hint on the bottom border; empty hides it. |
| `has_selection()` | no | Default True; return False for a read-only box. |
| `draw(host, rect)` | no | Override for a custom paint; the default draws the bordered box. |
| `handle_key(host, key, *, active)` | no | Override to own extra keys; call `super()` for the list keys. |

The default `handle_key` implements the list behaviour: `j`/`k` (or the arrows
in standard mode) move the selection, `Enter`/`Space` toggle the selected item,
`g`/`G` (and `Home`/`End`) jump to the ends.  Only override what you need and
return `True` when you consumed the key.

## Selection and scroll

A selectable pane keeps its view state on the host (so `/new` and history
restore can reset it) and exposes it through these hooks:

- `item_count(host)`, `selection(host)`, `set_selection(host, index)`
- `toggle_item(host, index)` — what `Enter` does to an item
- `scroll(host)`, `scroll_limit(host)`, `set_scroll(host, value)`

`set_selection` and `set_scroll` must clamp the value themselves (see
`panes/sources.py`).  The default `draw` keeps the selected row inside the
viewport.

## Geometry metadata

Class attributes tell the layout where the pane lives:

- `placeable` (default True): the name may appear in `tui_layout`.
- `parent`: for a *transient* pane only, the pane is not placed by the layout
  but carved out of the named pane's rectangle at run time, using `side`
  (`"bottom"`, `"top"`, `"left"`, `"right"`) and `split_ratio(host)` (default
  0.5).  `collapse` (default False) lets the child replace the parent when the
  parent is too small.  A placeable pane belongs in `tui_layout` instead: the
  tree owns its position and its weight.
- `transient` (default False): the pane is only carved while `visible(host)`
  returns True.
- `spawn_key` (default None): the key that toggles the pane.  The `[tui_panes]`
  table may also set `key = "..."` to override the class value at config time.
  While hidden the pane is pruned from the layout tree, its share goes to its
  siblings, and a split left with one child collapses into it.  A value of
  `"C-l"` means Ctrl-L.
- `splits_fullscreen` (default False): the pane may carve the expanded box when
  another pane is fullscreen.

Proportions are not pane metadata: they live in `tui_layout`.  `a | b` is a
`cols` split, `a / b` is a `rows` split, parentheses nest, and `name:N` sets the
weight inside the enclosing split.  To have a spawned box take half of another
box, put it under that box in the tree:

```toml
tui_layout = [
  "(sources / logs) | changes",
  "input",
]
```

`logs` stays hidden until Ctrl-L; while hidden `sources` fills the column, and
when shown the two share it 50/50.

## Capabilities

Two class flags let the host treat a box specially without knowing its name:

- `is_editor = True` marks the single box that edits text (the input line).  The
  host routes the editor, the hardware cursor and insert/normal mode to it.
  There must be exactly one editor pane; the layout rejects a config without it.
- `owns_merge_review = True` marks the box that renders and answers a pending
  merge review (`panes/changes.py`).

Do not invent new capability flags casually: if two boxes need a shared
behaviour, prefer a generic `PaneHost` service.

## Lifecycle hooks

These are optional no-ops that the host calls on every pane:

- `reset(host)` — clear view state for `/new` (default resets selection/scroll).
- `on_shown(host)` — called after the host toggles this pane into view.
- `on_log(host)` — called after the host appends lines to the shared log buffer.
- `on_asides(host)` — called after the host reloads the shared asides.

Use them to re-clamp a scroll or follow a live buffer without the host knowing
your pane exists (see `panes/logs.py` and `panes/extras.py`).

## The `PaneHost` surface

The `host` argument is a `PaneHost` (a runtime-checkable `Protocol`, implemented
by `tui.TuiApp`).  It is the complete surface a pane may touch.

**Shared session state** (written by the pipeline or the editor, read by panes):

```
config        focus        active       fullscreen   status
completion    merge_review settings     modal
source_paths  source_index source_scroll moc_hint
diff_entries  diff_index   diff_scroll  card_groups  card_index  card_scroll
asides        extras_scroll
log_lines     log_scroll
input_text    cursor       mode         editor
```

**Read-only properties:** `vim_movements`, `painter`, `running`,
`current_stage`; method `stage_text()`, `layout_rects()`, `pane_shown(name)`.

**Cross-box actions** (the generic services a pane may call):

```
set_focus(target)        activate_box()           move_focus(direction)
cycle_focus(delta)       toggle_fullscreen()      toggle_pane(name)
request_quit()           show_help()              open_settings()
answer_merge(response)   copy_text(text) -> bool  start_job()
submit_input()           start_completion()       refresh_completion()
cycle_completion(delta)  refresh_command_completion()
accept_command_completion()  accept_source_completion()
open_commands()          sync_editor_from_fields() pull_editor()
new_editor(...)          open_extras_export()
```

Services are generic on purpose.  There is no `copy_extras` or `remove_source`:
a pane copies via `host.copy_text(text)` and mutates the shared state it owns
directly (`panes/extras.py`, `panes/sources.py`).  If a future pane needs a new
cross-box action (for example an overlay), add it as a generic service here and
implement it once in the host — never as a per-pane branch.

## Drawing

The default `draw` calls `host.painter.draw_box(...)`, which draws the border,
title, footer and rows, soft-wraps long rows, keeps the selected row visible and
returns the scroll offset actually used.  Override `draw` only for a custom
paint (see `panes/changes.py` for the merge review, or `panes/input_pane.py`
for the mode badge).  Draw with `host.painter` and never touch the screen
directly.

## Key handling contract

`handle_key(host, key, *, active)` returns `True` when the pane consumed the
key.  The host routes a key to the focused pane first (`active=False` while the
box is merely focused, `active=True` once `Enter` selected it) and falls back to
the application-wide shortcuts only when the pane returns `False`.  Spawn keys
(`spawn_key`) are matched by the host before a pane sees them, so a box can be
toggled from anywhere.

## Checklist before shipping a pane

1. One module, one `Pane` subclass, one `name`; no `import tui`.
2. `title` and `rows` implemented; `has_selection() -> False` when read-only.
3. Selection/scroll hooks clamp their values.
4. Keys are owned via `handle_key`, calling `super()` for the list behaviour.
5. Cross-box actions go through `PaneHost`; no per-pane host methods.
6. `[tui_panes]` target is `"module:Class"` and `name` matches the key.
7. Placed in `tui_layout` (or carved via `parent` when transient); its weight
   and position live in the tree, not in the pane table.
8. A unit test in `tests/` covers the pane's rows and key handling against a
   fake host (see `tests/test_panes.py`).
