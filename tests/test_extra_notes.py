import tempfile
import unittest
from pathlib import Path

from scriptorium.extra_notes import format_extra_notes, write_extra_notes
from scriptorium.models import Aside


class ExtraNotesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_format_preserves_kind_source_and_text(self):
        asides = [
            Aside(kind="exam", text="The exam is oral.", source="lecture.mp4 00:00:10"),
            Aside(kind="link", text="See the course page."),
        ]
        text = format_extra_notes(asides)
        self.assertIn("- **[exam]** The exam is oral. _lecture.mp4 00:00:10_", text)
        self.assertIn("- **[link]** See the course page.", text)

    def test_format_without_asides_is_empty(self):
        self.assertEqual(format_extra_notes([]), "")

    def test_write_creates_the_file_and_appends_on_the_second_call(self):
        target = self.root / "extras.md"
        first = [Aside(kind="exam", text="First note.")]
        second = [Aside(kind="link", text="Second note.")]
        self.assertFalse(write_extra_notes(target, first))
        self.assertIn("First note.", target.read_text(encoding="utf-8"))
        self.assertTrue(write_extra_notes(target, second))
        text = target.read_text(encoding="utf-8")
        self.assertIn("First note.", text)
        self.assertIn("Second note.", text)
        self.assertLess(text.index("First note."), text.index("Second note."))

    def test_write_requires_at_least_one_aside(self):
        with self.assertRaises(ValueError):
            write_extra_notes(self.root / "extras.md", [])


if __name__ == "__main__":
    unittest.main()
