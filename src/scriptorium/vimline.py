"""A single-line Vim editor for the TUI input fields.

The behaviour is ported from the ``pivim`` Pi extension
(https://github.com/alessiocelentano/pivim), reduced to a one-line buffer:
modal editing, counts, motions, operators, registers, undo/redo and ``.``.

The editor is deliberately independent from curses.  The TUI owns rendering and
asks the editor for ``text``, ``cursor`` and ``mode``; only a few keys that are
owned by the application (``Tab`` completion, ``Enter`` submit) are handled
before the key reaches this class.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - curses is always present on the supported platforms
    import curses as _curses
except ImportError:  # pragma: no cover
    _curses = None

MODE_INSERT = "insert"
MODE_NORMAL = "normal"

BACKSPACE_KEYS = {8, 127, "\b", "\x7f", getattr(_curses, "KEY_BACKSPACE", 263)}
KEY_LEFT = getattr(_curses, "KEY_LEFT", 260)
KEY_RIGHT = getattr(_curses, "KEY_RIGHT", 261)
KEY_HOME = getattr(_curses, "KEY_HOME", 262)
KEY_END = getattr(_curses, "KEY_END", 360)
KEY_DC = getattr(_curses, "KEY_DC", 330)
ENTER_KEYS = {10, 13, getattr(_curses, "KEY_ENTER", 343)}


def is_printable(key: object) -> bool:
    return isinstance(key, str) and len(key) == 1 and key >= " " and key != "\x7f"


def _is_word(char: str) -> bool:
    return char.isalnum() or char == "_"


def _is_nonblank(char: str) -> bool:
    return not char.isspace()


@dataclass
class KeyResult:
    """What happened after a key press."""

    handled: bool = False
    text_changed: bool = False
    mode_changed: bool = False


@dataclass
class LastChange:
    """A recorded change, replayed by ``.``."""

    kind: str
    count: int = 1
    motion: str = ""
    char: str = ""
    op: str = "d"


class VimLineEditor:
    """Vim modal editing on a single line of text."""

    def __init__(
        self,
        text: str = "",
        *,
        cursor: int | None = None,
        mode: str = MODE_INSERT,
        arrow_movements: bool = True,
    ) -> None:
        self.text = text
        self.arrow_movements = arrow_movements
        self.mode = mode if mode in {MODE_INSERT, MODE_NORMAL} else MODE_INSERT
        self.cursor = len(text) if cursor is None else cursor
        self.cursor = max(0, min(self.cursor, len(self.text)))
        if self.mode == MODE_NORMAL:
            self._clamp_normal()

        self._count = ""
        self._pending_op = ""
        self._pending_leader = ""
        self._register = ""
        self._register_linewise = False
        self._last_find: tuple[str, bool, bool] | None = None  # char, forward, till
        self._last_change: LastChange | None = None
        self._undo: list[tuple[str, int]] = []
        self._redo: list[tuple[str, int]] = []
        self._insert_snapshot_taken = False

    # -- public helpers ------------------------------------------------------ #

    def set_text(self, text: str, cursor: int | None = None) -> None:
        self.text = text
        self.cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))
        if self.mode == MODE_NORMAL:
            self._clamp_normal()
        self._reset_pending()

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        if mode == MODE_INSERT:
            self._insert_snapshot_taken = False
        else:
            self._clamp_normal()
        self.mode = mode
        self._reset_pending()

    def enter_insert(self, cursor: int | None = None) -> None:
        if cursor is not None:
            self.cursor = max(0, min(cursor, len(self.text)))
        self.mode = MODE_INSERT
        self._insert_snapshot_taken = False
        self._reset_pending()

    def exit_insert(self) -> None:
        if self.mode != MODE_INSERT:
            return
        if self.cursor > 0:
            self.cursor -= 1
        self.mode = MODE_NORMAL
        self._clamp_normal()
        self._insert_snapshot_taken = False

    # -- key entry point ----------------------------------------------------- #

    def handle_key(self, key: object) -> KeyResult:
        if self.mode == MODE_INSERT:
            return self._handle_insert(key)
        return self._handle_normal(key)

    # -- insert mode --------------------------------------------------------- #

    def _handle_insert(self, key: object) -> KeyResult:
        # Escape leaves insert mode and moves one character left, like Vim.
        if key == "\x1b":
            self.exit_insert()
            return KeyResult(handled=True, mode_changed=True)

        if key in BACKSPACE_KEYS:
            if self.cursor > 0:
                self._snapshot_insert()
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
                return KeyResult(handled=True, text_changed=True)
            return KeyResult(handled=True)

        if key == KEY_DC or key == 0x1B5B337E:  # Delete
            if self.cursor < len(self.text):
                self._snapshot_insert()
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
                return KeyResult(handled=True, text_changed=True)
            return KeyResult(handled=True)

        if key in (0x1B5B44, KEY_LEFT):
            if self.arrow_movements:
                self.cursor = max(0, self.cursor - 1)
            return KeyResult(handled=True)
        if key in (0x1B5B43, KEY_RIGHT):
            if self.arrow_movements:
                self.cursor = min(len(self.text), self.cursor + 1)
            return KeyResult(handled=True)
        if key in (0x1B5B48, KEY_HOME):
            self.cursor = 0
            return KeyResult(handled=True)
        if key in (0x1B5B46, KEY_END):
            self.cursor = len(self.text)
            return KeyResult(handled=True)

        if key == "\x01":  # Ctrl-A
            self.cursor = 0
            return KeyResult(handled=True)
        if key == "\x05":  # Ctrl-E
            self.cursor = len(self.text)
            return KeyResult(handled=True)
        if key == "\x15":  # Ctrl-U: delete to start
            if self.cursor > 0:
                self._snapshot_insert()
                self.text = self.text[self.cursor :]
                self.cursor = 0
                return KeyResult(handled=True, text_changed=True)
            return KeyResult(handled=True)
        if key == "\x17":  # Ctrl-W: delete the word before the cursor
            start = self._word_start_before(self.cursor)
            if start < self.cursor:
                self._snapshot_insert()
                self.text = self.text[:start] + self.text[self.cursor :]
                self.cursor = start
                return KeyResult(handled=True, text_changed=True)
            return KeyResult(handled=True)

        if is_printable(key):
            self._snapshot_insert()
            self.text = self.text[: self.cursor] + str(key) + self.text[self.cursor :]
            self.cursor += 1
            self._remember(LastChange(kind="insert", char=str(key)))
            return KeyResult(handled=True, text_changed=True)

        return KeyResult(handled=False)

    # -- normal mode --------------------------------------------------------- #

    def _handle_normal(self, key: object) -> KeyResult:
        if key == "\x1b":
            self._reset_pending()
            return KeyResult(handled=True)

        if self._pending_leader:
            return self._leader_key(key)
        if self._pending_op:
            return self._operator_key(key)

        if key == "\x12":  # Ctrl-R redo
            return self._redo_once()

        # Digits accumulate a count.  A bare 0 is the "start of line" motion.
        if is_printable(key) and str(key).isdigit():
            if key == "0" and not self._count:
                self.cursor = 0
                return KeyResult(handled=True)
            self._count += str(key)
            return KeyResult(handled=True)

        if key in {"f", "F", "t", "T"}:
            self._pending_leader = str(key)
            return KeyResult(handled=True)
        if key == "g":
            self._pending_leader = "g"
            return KeyResult(handled=True)
        if key == "r":
            self._pending_leader = "r"
            return KeyResult(handled=True)
        if key in {"d", "c", "y"}:
            self._pending_op = str(key)
            return KeyResult(handled=True)

        return self._normal_command(key)

    def _normal_command(self, key: object) -> KeyResult:
        if self.arrow_movements:
            if key == KEY_LEFT:
                key = "h"
            elif key == KEY_RIGHT:
                key = "l"
            elif key == KEY_HOME:
                key = "0"
            elif key == KEY_END:
                key = "$"
        elif key == KEY_HOME:
            key = "0"
        elif key == KEY_END:
            key = "$"
        if not is_printable(key):
            return KeyResult(handled=False)
        key = str(key)
        count = self._take_count()

        if key in {"h", "l", "j", "k"}:
            if key in {"j", "k"}:
                return KeyResult(handled=False)
            self._move_cursor("h" if key == "h" else "l", count)
            return KeyResult(handled=True)
        if key == "0":
            self.cursor = 0
            return KeyResult(handled=True)
        if key == "^":
            self.cursor = self._first_non_blank()
            return KeyResult(handled=True)
        if key == "$":
            self.cursor = max(0, len(self.text) - 1)
            return KeyResult(handled=True)
        if key in {"w", "W", "b", "B", "e", "E"}:
            self._move_cursor(key, count)
            return KeyResult(handled=True)
        if key == "G":
            self.cursor = max(0, len(self.text) - 1)
            return KeyResult(handled=True)
        if key == "|":
            self.cursor = max(0, min(len(self.text) - 1, count - 1))
            return KeyResult(handled=True)
        if key in {";", ","}:
            self._repeat_find(key == ";", count)
            return KeyResult(handled=True)

        if key == "x":
            return self._delete_chars(count, forward=True)
        if key == "X":
            return self._delete_chars(count, forward=False)
        if key == "D":
            return self._delete_to_end()
        if key == "Y":
            self._register = self.text[self.cursor :]
            self._register_linewise = False
            return KeyResult(handled=True)
        if key == "C":
            self._snapshot()
            self.text = self.text[: self.cursor]
            self._remember(LastChange(kind="delete_to_end"))
            self.enter_insert(self.cursor)
            self._insert_snapshot_taken = True
            return KeyResult(handled=True, text_changed=True, mode_changed=True)
        if key == "S":
            self._snapshot()
            self.text = ""
            self._remember(LastChange(kind="delete_line"))
            self.enter_insert(0)
            self._insert_snapshot_taken = True
            return KeyResult(handled=True, text_changed=True, mode_changed=True)

        if key == "p":
            return self._paste(after=True)
        if key == "P":
            return self._paste(after=False)
        if key == "~":
            return self._toggle_case(count)
        if key == "u":
            return self._undo_once()
        if key == ".":
            return self._repeat(count)

        if key in {"i", "I", "a", "A"}:
            target = self.cursor
            if key == "I":
                target = self._first_non_blank()
            elif key == "a":
                target = min(len(self.text), self.cursor + 1)
            elif key == "A":
                target = len(self.text)
            self.enter_insert(target)
            return KeyResult(handled=True, mode_changed=True)

        return KeyResult(handled=False)

    # -- operators and leaders ----------------------------------------------- #

    def _operator_key(self, key: object) -> KeyResult:
        op = self._pending_op
        self._pending_op = ""

        if not is_printable(key):
            return KeyResult(handled=False)
        key = str(key)

        # A count after the operator: d2w.  A bare 0 is the line-start motion.
        if key.isdigit():
            if key == "0" and not self._count:
                self._pending_op = op
                return self._apply_operator(op, "0")
            self._count += key
            self._pending_op = op
            return KeyResult(handled=True)

        # Doubling the operator is linewise: dd, cc, yy.
        if key == op:
            count = self._take_count()
            return self._apply_linewise(op, count)

        # D / C / Y behave as d$ / c$ / y$ even while an operator is pending.
        if key in {"D", "C", "Y"} and key.lower() == op:
            return self._apply_operator(op, "$", count=self._take_count())

        if key in {"f", "F", "t", "T"}:
            self._pending_op = op
            self._pending_leader = key
            return KeyResult(handled=True)

        if key in {"h", "l", "w", "W", "b", "B", "e", "E", "0", "^", "$", "G", "|"}:
            count = self._take_count()
            return self._apply_operator(op, key, count=count)

        # Unknown key cancels the pending operator.
        return KeyResult(handled=True)

    def _leader_key(self, key: object) -> KeyResult:
        leader = self._pending_leader
        self._pending_leader = ""

        if leader == "g":
            if key == "g":
                self.cursor = 0
            return KeyResult(handled=True)

        if leader == "r":
            if not is_printable(key):
                return KeyResult(handled=True)
            count = self._take_count()
            return self._replace_chars(count, str(key))

        if not is_printable(key):
            return KeyResult(handled=True)
        char = str(key)
        forward = leader in {"f", "t"}
        till = leader in {"t", "T"}
        self._last_find = (char, forward, till)
        count = self._take_count()

        if self._pending_op:
            op = self._pending_op
            self._pending_op = ""
            return self._apply_operator(op, leader, count=count, char=char)
        target = self._find(char, forward=forward, till=till, count=count)
        if target is not None:
            self.cursor = target
        return KeyResult(handled=True)

    # -- operator application ------------------------------------------------ #

    def _apply_operator(
        self,
        op: str,
        motion: str,
        *,
        count: int = 1,
        char: str = "",
    ) -> KeyResult:
        start = self.cursor
        target, kind = self._motion_target(motion, count, char)

        # Vim special case: cw behaves like ce when the cursor is on a word.
        if op == "c" and motion == "w" and kind == "exclusive" and start < len(self.text):
            if _is_word(self.text[start]):
                target, kind = self._motion_target("e", count)

        if kind == "linewise":
            lo, hi = 0, len(self.text)
        elif target >= start:
            lo = start
            hi = target + (1 if kind == "inclusive" else 0)
        else:
            lo = target
            hi = start + (1 if kind == "inclusive" else 0)
        lo = max(0, min(lo, len(self.text)))
        hi = max(lo, min(hi, len(self.text)))

        if op == "y":
            self._register = self.text[lo:hi]
            self._register_linewise = kind == "linewise"
            self.cursor = lo
            self._clamp_normal()
            return KeyResult(handled=True)

        self._snapshot()
        removed = self.text[lo:hi]
        self._register = removed
        self._register_linewise = kind == "linewise"
        self.text = self.text[:lo] + self.text[hi:]
        self.cursor = min(lo, max(0, len(self.text) - 1))
        self._remember(LastChange(kind="operator", motion=motion, char=char, count=count, op=op))
        if op == "d":
            return KeyResult(handled=True, text_changed=True)

        # c: delete the range and enter insert mode.
        self.enter_insert(lo)
        self._insert_snapshot_taken = True
        return KeyResult(handled=True, text_changed=True, mode_changed=True)

    def _apply_linewise(self, op: str, count: int) -> KeyResult:
        # A single-line buffer has exactly one line, so any count is the whole text.
        if op == "y":
            self._register = self.text
            self._register_linewise = True
            return KeyResult(handled=True)
        self._snapshot()
        self.text = ""
        if op == "d":
            self._remember(LastChange(kind="delete_line"))
            self.cursor = 0
            return KeyResult(handled=True, text_changed=True)
        self._remember(LastChange(kind="delete_line"))
        self.enter_insert(0)
        self._insert_snapshot_taken = True
        return KeyResult(handled=True, text_changed=True, mode_changed=True)

    # -- motions ------------------------------------------------------------- #

    def _motion_target(self, motion: str, count: int, char: str = "") -> tuple[int, str]:
        length = len(self.text)
        cursor = self.cursor
        if motion == "h":
            return max(0, cursor - count), "exclusive"
        if motion == "l":
            return min(max(0, length - 1), cursor + count), "exclusive"
        if motion in {"0", "^"}:
            return (0 if motion == "0" else self._first_non_blank()), "exclusive"
        if motion in {"$", "G"}:
            return max(0, length - 1), "inclusive"
        if motion == "|":
            return max(0, min(max(0, length - 1), count - 1)), "exclusive"
        if motion in {"w", "W"}:
            return self._word_forward(count, big=motion == "W"), "exclusive"
        if motion in {"b", "B"}:
            return self._word_backward(count, big=motion == "B"), "exclusive"
        if motion in {"e", "E"}:
            return self._word_end(count, big=motion == "E"), "inclusive"
        if motion in {"f", "F", "t", "T"}:
            forward = motion in {"f", "t"}
            till = motion in {"t", "T"}
            target = self._find(char, forward=forward, till=till, count=count)
            if target is None:
                return cursor, "exclusive"
            return target, "inclusive"
        return cursor, "exclusive"

    def _move_cursor(self, motion: str, count: int) -> None:
        target, _ = self._motion_target(motion, count)
        self.cursor = max(0, min(target, max(0, len(self.text) - 1)))

    def _first_non_blank(self) -> int:
        index = 0
        while index < len(self.text) and self.text[index] in {" ", "\t"}:
            index += 1
        return min(index, max(0, len(self.text) - 1))

    def _word_forward(self, count: int, *, big: bool) -> int:
        is_word = _is_nonblank if big else _is_word
        pos = self.cursor
        length = len(self.text)
        for _ in range(max(1, count)):
            if pos < length and is_word(self.text[pos]):
                while pos < length and is_word(self.text[pos]):
                    pos += 1
            while pos < length and not is_word(self.text[pos]):
                pos += 1
        return pos

    def _word_backward(self, count: int, *, big: bool) -> int:
        is_word = _is_nonblank if big else _is_word
        pos = self.cursor
        for _ in range(max(1, count)):
            if pos > 0:
                pos -= 1
            while pos > 0 and not is_word(self.text[pos]):
                pos -= 1
            while pos > 0 and is_word(self.text[pos - 1]):
                pos -= 1
        return max(0, pos)

    def _word_end(self, count: int, *, big: bool) -> int:
        is_word = _is_nonblank if big else _is_word
        pos = self.cursor
        length = len(self.text)
        for _ in range(max(1, count)):
            if pos < length:
                pos += 1
            while pos < length and not is_word(self.text[pos]):
                pos += 1
            while pos < length and is_word(self.text[pos]):
                pos += 1
            if pos > 0:
                pos -= 1
        return max(0, min(pos, max(0, length - 1)))

    def _word_start_before(self, cursor: int) -> int:
        index = cursor
        while index > 0 and self.text[index - 1].isspace():
            index -= 1
        while index > 0 and not self.text[index - 1].isspace():
            index -= 1
        return index

    def _find(
        self,
        char: str,
        *,
        forward: bool,
        till: bool,
        count: int,
    ) -> int | None:
        if not char:
            return None
        pos = self.cursor
        for _ in range(max(1, count)):
            if forward:
                pos += 1
                while pos < len(self.text) and self.text[pos] != char:
                    pos += 1
                if pos >= len(self.text):
                    return None
                if till:
                    pos -= 1
                    if pos == self.cursor:
                        return None
            else:
                pos -= 1
                while pos >= 0 and self.text[pos] != char:
                    pos -= 1
                if pos < 0:
                    return None
                if till:
                    pos += 1
                    if pos == self.cursor:
                        return None
        return pos

    def _repeat_find(self, forward: bool, count: int) -> None:
        if self._last_find is None:
            return
        char, last_forward, till = self._last_find
        direction = last_forward if forward else not last_forward
        target = self._find(char, forward=direction, till=till, count=count)
        if target is not None:
            self.cursor = target

    # -- edits --------------------------------------------------------------- #

    def _delete_chars(self, count: int, *, forward: bool) -> KeyResult:
        if not self.text:
            return KeyResult(handled=True)
        self._snapshot()
        if forward:
            start = self.cursor
            end = min(len(self.text), start + max(1, count))
            self._register = self.text[start:end]
        else:
            start = max(0, self.cursor - max(1, count))
            end = self.cursor
            self._register = self.text[start:end]
        self._register_linewise = False
        self.text = self.text[:start] + self.text[end:]
        self.cursor = min(start, max(0, len(self.text) - 1))
        self._remember(LastChange(kind="delete", count=count, char="f" if forward else "b"))
        return KeyResult(handled=True, text_changed=True)

    def _delete_to_end(self) -> KeyResult:
        if not is_text_to_delete(self.text, self.cursor):
            return KeyResult(handled=True)
        self._snapshot()
        self._register = self.text[self.cursor :]
        self._register_linewise = False
        self.text = self.text[: self.cursor]
        self.cursor = max(0, min(self.cursor, len(self.text) - 1))
        self._remember(LastChange(kind="delete_to_end"))
        return KeyResult(handled=True, text_changed=True)

    def _replace_chars(self, count: int, char: str) -> KeyResult:
        if self.cursor >= len(self.text):
            return KeyResult(handled=True)
        self._snapshot()
        end = min(len(self.text), self.cursor + max(1, count))
        self.text = self.text[: self.cursor] + char * (end - self.cursor) + self.text[end:]
        self.cursor = min(end - 1, max(0, len(self.text) - 1))
        self._remember(LastChange(kind="replace", count=count, char=char))
        return KeyResult(handled=True, text_changed=True)

    def _toggle_case(self, count: int) -> KeyResult:
        if self.cursor >= len(self.text):
            return KeyResult(handled=True)
        self._snapshot()
        end = min(len(self.text), self.cursor + max(1, count))
        chunk = self.text[self.cursor : end].swapcase()
        self.text = self.text[: self.cursor] + chunk + self.text[end:]
        self.cursor = min(end, max(0, len(self.text) - 1))
        self._remember(LastChange(kind="toggle", count=count))
        return KeyResult(handled=True, text_changed=True)

    def _paste(self, *, after: bool) -> KeyResult:
        if not self._register:
            return KeyResult(handled=True)
        self._snapshot()
        index = self.cursor + 1 if after else self.cursor
        index = max(0, min(index, len(self.text)))
        self.text = self.text[:index] + self._register + self.text[index:]
        self.cursor = index + len(self._register) - 1
        self._clamp_normal()
        self._remember(LastChange(kind="paste", char="p" if after else "P"))
        return KeyResult(handled=True, text_changed=True)

    # -- undo, redo and repeat ---------------------------------------------- #

    def _snapshot(self) -> None:
        self._undo.append((self.text, self.cursor))
        self._redo.clear()

    def _snapshot_insert(self) -> None:
        if not self._insert_snapshot_taken:
            self._snapshot()
            self._insert_snapshot_taken = True

    def _undo_once(self) -> KeyResult:
        if not self._undo:
            return KeyResult(handled=True)
        self._redo.append((self.text, self.cursor))
        self.text, self.cursor = self._undo.pop()
        self._clamp_normal()
        self._reset_pending()
        return KeyResult(handled=True, text_changed=True)

    def _redo_once(self) -> KeyResult:
        if not self._redo:
            return KeyResult(handled=True)
        self._undo.append((self.text, self.cursor))
        self.text, self.cursor = self._redo.pop()
        self._clamp_normal()
        self._reset_pending()
        return KeyResult(handled=True, text_changed=True)

    def _remember(self, change: LastChange) -> None:
        self._last_change = change

    def _repeat(self, count: int) -> KeyResult:
        change = self._last_change
        if change is None:
            return KeyResult(handled=True)
        if change.kind == "insert":
            self._snapshot()
            index = min(self.cursor, len(self.text))
            repeated = change.char * count
            self.text = self.text[:index] + repeated + self.text[index:]
            self.cursor = min(index + len(repeated), max(0, len(self.text) - 1))
            self._clamp_normal()
            return KeyResult(handled=True, text_changed=True)
        if change.kind == "delete":
            return self._delete_chars(change.count * count, forward=change.char != "b")
        if change.kind == "delete_to_end":
            return self._delete_to_end()
        if change.kind == "delete_line":
            self._snapshot()
            self.text = ""
            self.cursor = 0
            return KeyResult(handled=True, text_changed=True)
        if change.kind == "replace":
            return self._replace_chars(change.count * count, change.char)
        if change.kind == "toggle":
            return self._toggle_case(change.count * count)
        if change.kind == "paste":
            return self._paste(after=change.char == "p")
        if change.kind == "operator":
            return self._apply_operator(
                change.op,
                change.motion,
                count=change.count * count,
                char=change.char,
            )
        return KeyResult(handled=True)

    # -- bookkeeping --------------------------------------------------------- #

    def _take_count(self) -> int:
        value = int(self._count) if self._count else 1
        self._count = ""
        return max(1, value)

    def _clamp_normal(self) -> None:
        if not self.text:
            self.cursor = 0
            return
        self.cursor = max(0, min(self.cursor, len(self.text) - 1))

    def _reset_pending(self) -> None:
        self._count = ""
        self._pending_op = ""
        self._pending_leader = ""


def is_text_to_delete(text: str, cursor: int) -> bool:
    """Whether ``D``/``C`` would remove anything at ``cursor``."""
    return bool(text) and cursor < len(text)
