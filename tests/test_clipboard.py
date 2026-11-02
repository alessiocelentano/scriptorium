import unittest
from types import SimpleNamespace
from unittest import mock

from scriptorium import clipboard


class ClipboardTests(unittest.TestCase):
    def test_osc52_sequence_encodes_the_text(self):
        self.assertEqual(clipboard._osc52_sequence("hi"), b"\x1b]52;c;aGk=\x07")

    def test_copy_uses_the_first_available_command(self):
        with (
            mock.patch.object(clipboard, "_candidate_commands", return_value=[["pbcopy"]]),
            mock.patch.object(clipboard.shutil, "which", return_value="/bin/pbcopy"),
            mock.patch.object(
                clipboard.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as run,
        ):
            self.assertTrue(clipboard.copy_to_clipboard("hello"))
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["pbcopy"])
        self.assertEqual(run.call_args.kwargs["input"], b"hello")

    def test_copy_falls_back_to_the_next_command(self):
        commands = [["pbcopy"], ["xclip", "-selection", "clipboard"]]
        with (
            mock.patch.object(clipboard, "_candidate_commands", return_value=commands),
            mock.patch.object(clipboard.shutil, "which", return_value="/bin/tool"),
            mock.patch.object(
                clipboard.subprocess,
                "run",
                side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)],
            ) as run,
        ):
            self.assertTrue(clipboard.copy_to_clipboard("hello"))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0], commands[1])

    def test_copy_uses_osc52_when_no_command_is_installed(self):
        with (
            mock.patch.object(clipboard, "_candidate_commands", return_value=[["pbcopy"]]),
            mock.patch.object(clipboard.shutil, "which", return_value=None),
            mock.patch.object(clipboard, "_write_osc52", return_value=True) as osc52,
        ):
            self.assertTrue(clipboard.copy_to_clipboard("hello"))
        osc52.assert_called_once_with("hello")

    def test_copy_reports_failure_when_no_mechanism_works(self):
        with (
            mock.patch.object(clipboard, "_candidate_commands", return_value=[]),
            mock.patch.object(clipboard, "_write_osc52", return_value=False),
        ):
            self.assertFalse(clipboard.copy_to_clipboard("hello"))


if __name__ == "__main__":
    unittest.main()
