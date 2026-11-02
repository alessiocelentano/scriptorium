import unittest

from scriptorium.vimline import MODE_INSERT, MODE_NORMAL, VimLineEditor


def press(editor: VimLineEditor, sequence: str | list[str]) -> VimLineEditor:
    for key in sequence:
        editor.handle_key(key)
    return editor


class InsertModeTests(unittest.TestCase):
    def test_typing_and_backspace(self):
        editor = VimLineEditor("")
        press(editor, "abc")
        self.assertEqual(editor.text, "abc")
        self.assertEqual(editor.cursor, 3)
        editor.handle_key("\x7f")
        self.assertEqual(editor.text, "ab")
        self.assertEqual(editor.cursor, 2)

    def test_escape_enters_normal_and_moves_left(self):
        editor = VimLineEditor("abc")
        result = editor.handle_key("\x1b")
        self.assertTrue(result.mode_changed)
        self.assertEqual(editor.mode, MODE_NORMAL)
        self.assertEqual(editor.cursor, 2)

    def test_home_end_and_delete_keys(self):
        editor = VimLineEditor("abcd", cursor=2)
        editor.handle_key("\x01")
        self.assertEqual(editor.cursor, 0)
        editor.handle_key("\x05")
        self.assertEqual(editor.cursor, 4)
        editor.handle_key(260)  # KEY_LEFT
        self.assertEqual(editor.cursor, 3)
        editor.handle_key(330)  # KEY_DC
        self.assertEqual(editor.text, "abc")

    def test_ctrl_u_and_ctrl_w(self):
        editor = VimLineEditor("one two", cursor=7)
        editor.handle_key("\x17")
        self.assertEqual(editor.text, "one ")
        self.assertEqual(editor.cursor, 4)
        editor.handle_key("\x15")
        self.assertEqual(editor.text, "")
        self.assertEqual(editor.cursor, 0)


class NormalModeTests(unittest.TestCase):
    def setUp(self):
        self.editor = VimLineEditor("hello world foo", mode=MODE_NORMAL)
        self.editor.cursor = 0

    def test_horizontal_and_line_motions(self):
        press(self.editor, "ll")
        self.assertEqual(self.editor.cursor, 2)
        press(self.editor, "$")
        self.assertEqual(self.editor.cursor, len(self.editor.text) - 1)
        press(self.editor, "0")
        self.assertEqual(self.editor.cursor, 0)
        press(self.editor, "^")
        self.assertEqual(self.editor.cursor, 0)

    def test_word_motions(self):
        press(self.editor, "w")
        self.assertEqual(self.editor.cursor, 6)
        press(self.editor, "e")
        self.assertEqual(self.editor.cursor, 10)
        press(self.editor, "b")
        self.assertEqual(self.editor.cursor, 6)

    def test_counts(self):
        press(self.editor, "2w")
        self.assertEqual(self.editor.cursor, 12)
        press(self.editor, "0")
        press(self.editor, "2x")
        self.assertEqual(self.editor.text, "llo world foo")

    def test_delete_word_operator(self):
        press(self.editor, "dw")
        self.assertEqual(self.editor.text, "world foo")
        self.assertEqual(self.editor.cursor, 0)

    def test_count_after_operator(self):
        press(self.editor, "d2w")
        self.assertEqual(self.editor.text, "foo")

    def test_change_word_enters_insert(self):
        self.editor.cursor = 6
        press(self.editor, "cwbye")
        self.assertEqual(self.editor.text, "hello bye foo")
        self.assertEqual(self.editor.mode, MODE_INSERT)
        self.assertEqual(self.editor.cursor, 9)

    def test_change_to_end_of_line(self):
        self.editor.cursor = 6
        press(self.editor, "D")
        self.assertEqual(self.editor.text, "hello ")
        press(self.editor, "0")
        press(self.editor, "C")
        self.assertEqual(self.editor.text, "")
        self.assertEqual(self.editor.mode, MODE_INSERT)

    def test_yank_and_paste(self):
        self.editor.cursor = 0
        press(self.editor, "ylp")
        self.assertEqual(self.editor.text, "hhello world foo")
        press(self.editor, "P")
        self.assertEqual(self.editor.text, "hhhello world foo")

    def test_replace_and_toggle_case(self):
        self.editor.cursor = 0
        press(self.editor, "3rz")
        self.assertEqual(self.editor.text, "zzzlo world foo")
        press(self.editor, "0~")
        self.assertEqual(self.editor.text, "Zzzlo world foo")

    def test_find_and_repeat(self):
        self.editor.cursor = 0
        press(self.editor, "fo")
        self.assertEqual(self.editor.cursor, 4)
        press(self.editor, ";")
        self.assertEqual(self.editor.cursor, 7)
        press(self.editor, ",")
        self.assertEqual(self.editor.cursor, 4)

    def test_delete_till_and_find(self):
        editor = VimLineEditor("abc def", cursor=0, mode=MODE_NORMAL)
        press(editor, "dfd")
        self.assertEqual(editor.text, "ef")

        editor = VimLineEditor("abc def", cursor=0, mode=MODE_NORMAL)
        press(editor, "dtd")
        self.assertEqual(editor.text, "def")

    def test_enter_and_append(self):
        self.editor.cursor = 1
        press(self.editor, "iX")
        self.assertEqual(self.editor.text, "hXello world foo")
        self.editor.exit_insert()
        press(self.editor, "A!")
        self.assertEqual(self.editor.text, "hXello world foo!")
        self.assertEqual(self.editor.cursor, len(self.editor.text))

    def test_undo_and_redo(self):
        press(self.editor, "x")
        self.assertEqual(self.editor.text, "ello world foo")
        press(self.editor, "u")
        self.assertEqual(self.editor.text, "hello world foo")
        press(self.editor, "\x12")
        self.assertEqual(self.editor.text, "ello world foo")

    def test_dot_repeats_last_change(self):
        press(self.editor, "x")
        press(self.editor, ".")
        self.assertEqual(self.editor.text, "llo world foo")

    def test_linewise_delete(self):
        press(self.editor, "dd")
        self.assertEqual(self.editor.text, "")

    def test_escape_cancels_a_pending_operator(self):
        press(self.editor, "d\x1b")
        press(self.editor, "x")
        self.assertEqual(self.editor.text, "ello world foo")


class ArrowMovementTests(unittest.TestCase):
    def test_insert_mode_ignores_left_and_right_when_arrows_are_disabled(self):
        editor = VimLineEditor("abcd", cursor=2, arrow_movements=False)
        editor.handle_key(260)  # KEY_LEFT
        self.assertEqual(editor.cursor, 2)
        editor.handle_key(261)  # KEY_RIGHT
        self.assertEqual(editor.cursor, 2)

    def test_normal_mode_ignores_left_and_right_when_arrows_are_disabled(self):
        editor = VimLineEditor("abcd", cursor=2, mode=MODE_NORMAL, arrow_movements=False)
        editor.handle_key(260)  # KEY_LEFT
        self.assertEqual(editor.cursor, 2)
        editor.handle_key(261)  # KEY_RIGHT
        self.assertEqual(editor.cursor, 2)
        editor.handle_key(262)  # KEY_HOME is not an arrow and still works
        self.assertEqual(editor.cursor, 0)

    def test_standard_mode_keeps_arrow_movements(self):
        editor = VimLineEditor("abcd", cursor=2, arrow_movements=True)
        editor.handle_key(260)  # KEY_LEFT
        self.assertEqual(editor.cursor, 1)
        editor.handle_key(261)  # KEY_RIGHT
        self.assertEqual(editor.cursor, 2)


if __name__ == "__main__":
    unittest.main()
