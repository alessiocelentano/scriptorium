"""TUI pane layout: a recursive split tree, its configuration and its geometry.

The layout is a tree.  A leaf names a pane; a split divides its rectangle along
one axis between its children, proportional to their weights.  ``rows`` stacks
children top to bottom and ``cols`` places them left to right.  A split may
contain further splits, so the topology can be arbitrarily deep.

The configuration mini-language mirrors the tree: ``a | b`` is a ``cols`` split,
``a / b`` is a ``rows`` split, parentheses nest, and ``name:weight`` sets the
share of the enclosing split (default 1).  A list of row strings stays the
common shorthand and means a ``rows`` split of ``cols`` rows.

Every name must be a registered *placeable* pane (see ``pane_registry``) and
must appear exactly once so focus navigation always has a destination.  The
input line is required because it is the only way to start a job; its box is
fixed at three terminal rows.

A pane whose spawn key has not been pressed is pruned from the tree before the
solver runs, so its share is redistributed to its siblings and a split left
with one child collapses into it.  The solver always fills the terminal
exactly; the weights decide the shares, not caps.

Panes that declare a ``parent`` are not part of the tree: they are carved out
of the named pane's rectangle at run time by :func:`split_rect`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .errors import ConfigurationError

PANE_SOURCES = "sources"
PANE_CHANGES = "changes"
PANE_FLASHCARDS = "flashcards"
PANE_EXTRAS = "extras"
PANE_LOGS = "logs"
PANE_INPUT = "input"
#: The panes the built-in default layout places.  A configuration can register
#: more (``[tui_panes]``); validation takes the name list from the caller.
PANES = (
    PANE_SOURCES,
    PANE_CHANGES,
    PANE_FLASHCARDS,
    PANE_EXTRAS,
    PANE_LOGS,
    PANE_INPUT,
)
DEFAULT_PANES = PANES

# A split child needs at least this many rows to show a border, one content line
# and a footer.
SPLIT_MIN_HEIGHT = 3

# A horizontal carve-out (a child pane on the left or right) needs this many
# columns to stay usable: two borders and two content cells.
MIN_SPLIT_WIDTH = 8

# A leaf placed by the tree needs two borders and two content cells to be
# readable at all.  The solver treats it as the floor of a ``cols`` split.
MIN_BOX_WIDTH = 4

# The two axes a split can use.  ``rows`` stacks its children, ``cols`` places
# them side by side with one column of gap between them.
AXIS_ROWS = "rows"
AXIS_COLS = "cols"

# Tokens of the layout mini-language: the structural punctuation, or a run of
# any other non-space character (a pane name or a weight).
_TOKEN_PATTERN = re.compile(r"[()|/:]|[^\s()|/:]+")


@dataclass(frozen=True)
class Leaf:
    """One pane placed by the layout tree."""

    pane: str


@dataclass(frozen=True)
class Split:
    """A rectangle divided along ``axis`` between weighted children."""

    axis: str
    children: tuple[tuple[LayoutNode, float], ...]


LayoutNode = Leaf | Split

# The default layout: sources above extras on the left, changes in the middle,
# flashcards above logs on the right, the input line at the bottom.  Extras and
# logs therefore take half of their column.  The logs box has a spawn key, so
# it starts hidden and flashcards fills the whole column until that key is
# pressed.
DEFAULT_LAYOUT: LayoutNode = Split(
    AXIS_ROWS,
    (
        (
            Split(
                AXIS_COLS,
                (
                    (Split(AXIS_ROWS, ((Leaf(PANE_SOURCES), 1), (Leaf(PANE_EXTRAS), 1))), 1),
                    (Leaf(PANE_CHANGES), 1),
                    (Split(AXIS_ROWS, ((Leaf(PANE_FLASHCARDS), 1), (Leaf(PANE_LOGS), 1))), 1),
                ),
            ),
            1,
        ),
        (Leaf(PANE_INPUT), 1),
    ),
)

# The input shows one content row; its box adds the top and bottom borders.
INPUT_CONTENT_HEIGHT = 1
INPUT_BOX_BORDERS = 2
INPUT_HEIGHT = INPUT_CONTENT_HEIGHT + INPUT_BOX_BORDERS


@dataclass(frozen=True)
class PaneRect:
    """A pane rectangle in terminal cells (``h``/``w`` are sizes, not edges)."""

    y: int
    x: int
    h: int
    w: int

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def parse_layout(
    raw: object,
    names: tuple[str, ...] = PANES,
) -> LayoutNode:
    """Validate and normalize a ``tui_layout`` value from TOML.

    ``None`` selects :data:`DEFAULT_LAYOUT`.  A string is one expression such
    as ``"(sources / extras) | changes"``.  A list is a series of rows: each
    entry is either a row expression or a sequence of pane names, and the rows
    are stacked top to bottom.  ``names`` is the list of placeable panes the
    registry offers; a name that is unknown, not placeable or used twice is a
    configuration error, so a typo never produces a half-working layout.

    A registered pane may be left out of the layout: the layout selects which
    boxes exist, so a configuration can hide a box without unregistering it.
    The input line is the exception -- it carries the only way to start a job,
    so a layout without it is rejected.
    """
    if isinstance(raw, (Leaf, Split)):
        layout = raw
    elif raw is None:
        layout = DEFAULT_LAYOUT
    elif isinstance(raw, str):
        layout = _parse_expression(raw)
    elif isinstance(raw, (list, tuple)):
        layout = _parse_rows(raw)
    else:
        raise ConfigurationError("tui_layout must be a string or a list of row strings")
    _validate_layout(layout, names)
    return layout


def _parse_rows(entries: tuple[object, ...] | list[object]) -> LayoutNode:
    rows: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            row = entry.strip()
        elif isinstance(entry, (list, tuple)):
            row = " | ".join(str(part).strip() for part in entry if str(part).strip())
        else:
            raise ConfigurationError("each tui_layout row must be a string")
        if not row:
            raise ConfigurationError("tui_layout rows cannot be empty")
        rows.append(row)
    if not rows:
        raise ConfigurationError("tui_layout cannot be empty")
    # Stack the rows; the parentheses keep a row's own operator local, so a row
    # may itself use '/' without mixing operators at the top level.
    return _parse_expression(" / ".join(f"({row})" for row in rows))


def _parse_expression(text: str) -> LayoutNode:
    tokens = _TOKEN_PATTERN.findall(text)
    if not tokens:
        raise ConfigurationError("tui_layout cannot be empty")
    return _Parser(tokens).parse()


class _Parser:
    """Recursive-descent parser for the layout mini-language."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> LayoutNode:
        node = self._expression()
        if self._peek() is not None:
            raise ConfigurationError(f"unexpected {self._peek()!r} in tui_layout")
        return node

    def _peek(self) -> str | None:
        return self._tokens[self._index] if self._index < len(self._tokens) else None

    def _advance(self) -> str | None:
        token = self._peek()
        if token is not None:
            self._index += 1
        return token

    def _expression(self) -> LayoutNode:
        children = [self._term()]
        axis: str | None = None
        while self._peek() in ("|", "/"):
            operator = self._advance()
            current = AXIS_COLS if operator == "|" else AXIS_ROWS
            if axis is None:
                axis = current
            elif current != axis:
                raise ConfigurationError(
                    "tui_layout cannot mix '|' and '/' at the same level; "
                    "wrap one side in parentheses"
                )
            children.append(self._term())
        if axis is None:
            return children[0][0]
        return Split(axis, tuple(children))

    def _term(self) -> tuple[LayoutNode, float]:
        token = self._advance()
        if token is None:
            raise ConfigurationError("tui_layout expression is incomplete")
        if token == "(":
            node = self._expression()
            if self._advance() != ")":
                raise ConfigurationError("tui_layout has an unclosed '('")
        elif token in (")", "|", "/", ":"):
            raise ConfigurationError(f"unexpected {token!r} in tui_layout")
        else:
            node = Leaf(token.strip().lower())
        weight = 1.0
        if self._peek() == ":":
            self._advance()
            weight = _parse_weight(self._advance())
        return node, weight


def _parse_weight(token: str | None) -> float:
    if token is None:
        raise ConfigurationError("tui_layout weight is missing after ':'")
    try:
        weight = float(token)
    except ValueError as exc:
        raise ConfigurationError(
            f"tui_layout weight must be a positive number, got {token!r}"
        ) from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ConfigurationError(f"tui_layout weight must be a positive number, got {token!r}")
    return weight


def _validate_layout(layout: LayoutNode, names: tuple[str, ...]) -> None:
    seen: list[str] = []

    def walk(node: LayoutNode) -> None:
        if isinstance(node, Leaf):
            pane = node.pane
            if pane not in names:
                raise ConfigurationError(
                    f"unknown tui_layout pane {pane!r}; expected one of {', '.join(names)}"
                )
            if pane in seen:
                raise ConfigurationError(f"tui_layout pane {pane!r} appears more than once")
            seen.append(pane)
            return
        for child, _weight in node.children:
            walk(child)

    walk(layout)
    missing = [pane for pane in (PANE_INPUT,) if pane in names and pane not in seen]
    if missing:
        raise ConfigurationError(f"tui_layout must include the {missing[0]} pane")


def layout_panes(layout: LayoutNode) -> tuple[str, ...]:
    """Pane names in reading order (depth first, children left to right)."""
    if isinstance(layout, Leaf):
        return (layout.pane,)
    return tuple(pane for child, _weight in layout.children for pane in layout_panes(child))


def layout_text(layout: LayoutNode) -> str:
    """Render a layout expression for status messages and diagnostics."""
    if isinstance(layout, Leaf):
        return layout.pane
    parts: list[str] = []
    for child, weight in layout.children:
        text = layout_text(child)
        if isinstance(child, Split):
            text = f"({text})"
        if weight != 1:
            text += f":{weight:g}"
        parts.append(text)
    operator = " | " if layout.axis == AXIS_COLS else " / "
    return operator.join(parts)


def pane_rects(
    layout: LayoutNode,
    rows: int,
    cols: int,
    hidden: frozenset[str] = frozenset(),
) -> dict[str, PaneRect]:
    """Compute every pane rectangle for a terminal of ``rows`` by ``cols``.

    The header occupies row 0.  Boxes use one column of left margin and end one
    column before the terminal edge.  The tree is pruned of hidden panes first,
    then every split divides its rectangle between its children in proportion
    to their weights.  The input box always keeps its three terminal rows.
    """
    if rows < 2 or cols < 4:
        return {}
    tree = prune_layout(layout, hidden)
    if tree is None:
        return {}
    rects: dict[str, PaneRect] = {}
    _place(tree, PaneRect(y=1, x=1, h=rows - 1, w=cols - 2), rects)
    return rects


def prune_layout(layout: LayoutNode, hidden: frozenset[str]) -> LayoutNode | None:
    """Remove hidden leaves, collapsing a split left with a single child."""
    if isinstance(layout, Leaf):
        return None if layout.pane in hidden else layout
    kept: list[tuple[LayoutNode, float]] = []
    for child, weight in layout.children:
        pruned = prune_layout(child, hidden)
        if pruned is not None:
            kept.append((pruned, weight))
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0][0]
    return Split(layout.axis, tuple(kept))


def _place(node: LayoutNode, rect: PaneRect, rects: dict[str, PaneRect]) -> None:
    if isinstance(node, Leaf):
        if rect.h >= 1 and rect.w >= 1:
            rects[node.pane] = rect
        return
    sizes = _split_sizes(node.children, rect, node.axis)
    if node.axis == AXIS_COLS:
        x = rect.x
        for (child, _weight), size in zip(node.children, sizes, strict=True):
            _place(child, PaneRect(y=rect.y, x=x, h=rect.h, w=size), rects)
            x += size + 1
    else:
        y = rect.y
        for (child, _weight), size in zip(node.children, sizes, strict=True):
            _place(child, PaneRect(y=y, x=rect.x, h=size, w=rect.w), rects)
            y += size


def _split_sizes(
    children: tuple[tuple[LayoutNode, float], ...],
    rect: PaneRect,
    axis: str,
) -> list[int]:
    gaps = len(children) - 1 if axis == AXIS_COLS else 0
    total = (rect.w if axis == AXIS_COLS else rect.h) - gaps
    weights = [weight for _child, weight in children]
    minimums = [_minimum_size(child, axis) for child, _weight in children]
    fixed = [_fixed_size(child, axis) for child, _weight in children]
    return _allocate(total, weights, minimums, fixed)


def _minimum_size(node: LayoutNode, axis: str) -> int:
    if isinstance(node, Leaf):
        if axis == AXIS_ROWS:
            return INPUT_HEIGHT if node.pane == PANE_INPUT else SPLIT_MIN_HEIGHT
        return MIN_BOX_WIDTH
    if node.axis != axis:
        return max(_minimum_size(child, axis) for child, _weight in node.children)
    gaps = len(node.children) - 1 if axis == AXIS_COLS else 0
    return gaps + sum(_minimum_size(child, axis) for child, _weight in node.children)


def _fixed_size(node: LayoutNode, axis: str) -> int | None:
    if axis == AXIS_ROWS and isinstance(node, Leaf) and node.pane == PANE_INPUT:
        return INPUT_HEIGHT
    return None


def _allocate(
    total: int,
    weights: list[float],
    minimums: list[int],
    fixed: list[int | None],
) -> list[int]:
    """Allocate exactly ``total`` cells across the siblings of one split.

    Fixed children (the input box) take their size first; the rest share the
    remainder in proportion to their weights, then any child below its minimum
    is raised at the expense of the others.
    """
    count = len(weights)
    if count == 0:
        return []
    if total <= 0:
        return [0 for _ in range(count)]
    sizes = [0 for _ in range(count)]
    fixed_indices = [index for index, value in enumerate(fixed) if value is not None]
    fixed_total = 0
    for index in fixed_indices:
        size = min(int(fixed[index] or 0), total)
        sizes[index] = size
        fixed_total += size
    if fixed_total > total:
        overflow = fixed_total - total
        for index in reversed(fixed_indices):
            cut = min(sizes[index], overflow)
            sizes[index] -= cut
            overflow -= cut
            if overflow == 0:
                break
        fixed_total = total
    flexible = [index for index, value in enumerate(fixed) if value is None]
    if not flexible:
        return sizes
    remaining = max(0, total - fixed_total)
    shares = _distribute(
        remaining,
        [weights[index] for index in flexible],
        [minimums[index] for index in flexible],
    )
    for index, share in zip(flexible, shares, strict=True):
        sizes[index] = share
    return sizes


def _distribute(total: int, weights: list[float], minimums: list[int]) -> list[int]:
    """Split ``total`` by weight, then enforce the per-child minimums."""
    count = len(weights)
    if count == 0 or total <= 0:
        return [0 for _ in range(count)]
    minimums = [max(0, min(int(value), total)) for value in minimums]
    floors = sum(minimums)
    if floors >= total:
        if floors == total:
            return minimums
        return _largest_remainder([total * value / floors for value in minimums], total)
    total_weight = sum(weights)
    if total_weight <= 0:
        quotas = [total / count for _ in range(count)]
    else:
        quotas = [total * weight / total_weight for weight in weights]
    sizes = _largest_remainder(quotas, total)
    deficit = 0
    for index, floor in enumerate(minimums):
        if sizes[index] < floor:
            deficit += floor - sizes[index]
            sizes[index] = floor
    while deficit > 0:
        donors = [index for index in range(count) if sizes[index] > minimums[index]]
        if not donors:
            break
        surplus = sum(sizes[index] - minimums[index] for index in donors)
        taken = 0
        for index in donors:
            if deficit <= 0:
                break
            share = max(1, round(deficit * (sizes[index] - minimums[index]) / surplus))
            take = min(share, sizes[index] - minimums[index], deficit)
            sizes[index] -= take
            deficit -= take
            taken += take
        if taken == 0:
            break
    return sizes


def _largest_remainder(quotas: list[float], total: int) -> list[int]:
    """Convert fractional quotas to integers that sum to ``total``."""
    sizes = [max(0, int(value)) for value in quotas]
    assigned = sum(sizes)
    if assigned < total:
        order = sorted(
            range(len(quotas)),
            key=lambda index: (-(quotas[index] - sizes[index]), index),
        )
        for index in order[: total - assigned]:
            sizes[index] += 1
    elif assigned > total:
        order = sorted(range(len(sizes)), key=lambda index: (-sizes[index], index))
        for index in order:
            if assigned <= total:
                break
            if sizes[index] > 0:
                sizes[index] -= 1
                assigned -= 1
    return sizes


def content_area(rects: dict[str, PaneRect], excluded: str = PANE_INPUT) -> PaneRect | None:
    """Bounding rectangle of every pane except ``excluded``.

    This is the space a full-screen pane may occupy: all the boxes above (or
    around) the input line, with no gap and no overlap.
    """
    boxes = [rect for name, rect in rects.items() if name != excluded]
    if not boxes:
        return None
    y = min(rect.y for rect in boxes)
    x = min(rect.x for rect in boxes)
    return PaneRect(
        y=y,
        x=x,
        h=max(rect.bottom for rect in boxes) - y,
        w=max(rect.right for rect in boxes) - x,
    )


def split_rect(
    rect: PaneRect,
    side: str = "bottom",
    ratio: float = 0.5,
    *,
    collapse: bool = False,
) -> tuple[PaneRect | None, PaneRect | None]:
    """Carve a child pane out of one edge of ``rect``.

    Returns ``(parent, child)``.  ``side`` is the edge the child takes
    (``bottom``/``top``/``left``/``right``) and ``ratio`` is the fraction of the
    parent it occupies.  When the parent is too small for two bordered boxes the
    child wins if ``collapse`` is set (the log viewer replaces the cards) and the
    parent wins otherwise (a short Sources column keeps only Sources); the other
    half of the result is then ``None``.
    """
    if side not in {"bottom", "top", "left", "right"}:
        raise ValueError(f"unknown split side {side!r}")
    horizontal = side in {"bottom", "top"}
    size = rect.h if horizontal else rect.w
    minimum = SPLIT_MIN_HEIGHT if horizontal else MIN_SPLIT_WIDTH
    if size < 2 * minimum:
        return (None, rect) if collapse else (rect, None)
    ratio = min(max(ratio, 0.1), 0.9)
    child_size = max(minimum, min(size - minimum, int(round(size * ratio))))
    parent_size = size - child_size
    if side == "bottom":
        return (
            PaneRect(y=rect.y, x=rect.x, h=parent_size, w=rect.w),
            PaneRect(y=rect.y + parent_size, x=rect.x, h=child_size, w=rect.w),
        )
    if side == "top":
        return (
            PaneRect(y=rect.y + child_size, x=rect.x, h=parent_size, w=rect.w),
            PaneRect(y=rect.y, x=rect.x, h=child_size, w=rect.w),
        )
    if side == "right":
        return (
            PaneRect(y=rect.y, x=rect.x, h=rect.h, w=parent_size),
            PaneRect(y=rect.y, x=rect.x + parent_size, h=rect.h, w=child_size),
        )
    return (
        PaneRect(y=rect.y, x=rect.x + child_size, h=rect.h, w=parent_size),
        PaneRect(y=rect.y, x=rect.x, h=rect.h, w=child_size),
    )


def neighbor(
    layout: LayoutNode,
    source: str,
    direction: str,
    rows: int,
    cols: int,
    point: tuple[float, float] | None = None,
) -> str | None:
    """Return the pane reached from ``source`` in ``direction`` (``h/j/k/l``).

    ``point`` is the remembered position inside ``source`` (the tiling-manager
    anchor); ``None`` uses the centre of the pane.  See :func:`neighbor_rects`.
    """
    return neighbor_rects(
        pane_rects(layout, rows, cols),
        source,
        direction,
        point=point,
    )


def neighbor_rects(
    rects: dict[str, PaneRect],
    source: str,
    direction: str,
    point: tuple[float, float] | None = None,
) -> str | None:
    """Return the rectangle reached from ``source`` in ``direction``.

    Movement follows the tiling-manager rule: only panes on the requested side
    are candidates, the closest edge wins, and ties are broken by the anchor
    position (which pane the remembered point falls in, then which is closest).
    This is what makes ``right`` followed by ``up`` land in the upper box that
    sits on the right, exactly like ``i3`` or ``tmux``.

    Callers that add transient panes (the inline log viewer) build the rectangle
    map themselves, so navigation sees exactly what the screen shows.
    """
    origin = rects.get(source)
    if origin is None:
        return None
    anchor = point if point is not None else (origin.center_x, origin.center_y)

    best: tuple[float, float, float, str] | None = None
    for name, rect in rects.items():
        if name == source:
            continue
        score = _directional_score(origin, rect, direction, anchor)
        if score is None:
            continue
        candidate = (score[0], score[1], score[2], name)
        if best is None or candidate < best:
            best = candidate
    return best[3] if best is not None else None


def clamp_point(point: tuple[float, float], rect: PaneRect) -> tuple[float, float]:
    """Clamp ``point`` so it falls inside ``rect`` (cells, not edges)."""
    x, y = point
    return (
        min(max(x, float(rect.x)), float(rect.right - 1)),
        min(max(y, float(rect.y)), float(rect.bottom - 1)),
    )


def edge_point(
    rect: PaneRect,
    direction: str,
    point: tuple[float, float],
) -> tuple[float, float]:
    """Move the anchor to the ``direction`` edge of ``rect``.

    Used when a direction has no neighbouring pane: the anchor still travels to
    that side, so the next perpendicular move picks the box on that side.
    """
    x, y = clamp_point(point, rect)
    if direction == "l":
        return (float(rect.right - 1), y)
    if direction == "h":
        return (float(rect.x), y)
    if direction == "j":
        return (x, float(rect.bottom - 1))
    if direction == "k":
        return (x, float(rect.y))
    raise ValueError(f"unknown layout direction {direction!r}")


def _directional_score(
    origin: PaneRect,
    rect: PaneRect,
    direction: str,
    anchor: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Score ``rect`` as a destination, or ``None`` when it is not that way.

    Returns ``(edge_gap, anchor_reach, anchor_centre_distance)``: the distance
    between the facing edges, how far the anchor is from the candidate's
    perpendicular span (zero when the anchor falls inside it), and the distance
    from the anchor to the candidate's centre.  Comparing them in that order is
    what makes the movement feel spatial instead of diagonal.
    """
    ax, ay = anchor
    if direction == "l":
        if rect.x < origin.right:
            return None
        gap = rect.x - origin.right
        low, high, coordinate, centre = rect.y, rect.bottom, ay, rect.center_y
    elif direction == "h":
        if rect.right > origin.x:
            return None
        gap = origin.x - rect.right
        low, high, coordinate, centre = rect.y, rect.bottom, ay, rect.center_y
    elif direction == "j":
        if rect.y < origin.bottom:
            return None
        gap = rect.y - origin.bottom
        low, high, coordinate, centre = rect.x, rect.right, ax, rect.center_x
    elif direction == "k":
        if rect.bottom > origin.y:
            return None
        gap = origin.y - rect.bottom
        low, high, coordinate, centre = rect.x, rect.right, ax, rect.center_x
    else:
        raise ValueError(f"unknown layout direction {direction!r}")

    if coordinate < low:
        reach = low - coordinate
    elif coordinate >= high:
        reach = coordinate - high
    else:
        reach = 0.0
    return (float(gap), float(reach), abs(centre - coordinate))
