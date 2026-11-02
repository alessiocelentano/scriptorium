import tempfile
import unittest
from pathlib import Path

from scriptorium.errors import VaultError
from scriptorium.vault import Vault, only_additions, parse_tags


class VaultTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "University").mkdir()
        self.vault = Vault(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")


class InsertTests(VaultTestCase):
    def test_insert_under_heading_preserves_bytes(self):
        original = (
            "---\n"
            "tags:\n"
            "  - cs/ai\n"
            "---\n"
            "\n"
            "# Note\n"
            "\n"
            "## Section A\n"
            "\n"
            "old line\n"
            "\n"
            "## Section B\n"
            "\n"
            "tail\n"
        )
        self.write("N.md", original)
        result = self.vault.insert_under_heading("N.md", "Section A", "new line")
        self.assertEqual(
            result,
            original.replace("old line\n\n## Section B", "old line\n\nnew line\n\n## Section B"),
        )

    def test_insert_under_heading_missing(self):
        self.write("N.md", "## A\n")
        with self.assertRaises(VaultError):
            self.vault.insert_under_heading("N.md", "B", "x")

    def test_insert_tight_for_moc(self):
        self.write(
            "MOC.md",
            "## Table of Contents\n\n- [[A]]\n\n## Useful Links\n",
        )
        result = self.vault.insert_under_heading(
            "MOC.md", "Table of Contents", "- [[B]]", tight=True
        )
        self.assertEqual(
            result,
            "## Table of Contents\n\n- [[A]]\n- [[B]]\n\n## Useful Links\n",
        )

    def test_append_at_end(self):
        self.write("N.md", "body")
        result = self.vault.append_at_end("N.md", "\n---\n\n## Flashcards")
        self.assertEqual(result, "body\n\n---\n\n## Flashcards\n")


class WriteGuardTests(VaultTestCase):
    def test_refuses_and_restores_non_insertion(self):
        self.write("N.md", "a\nb\n")
        with self.assertRaises(VaultError):
            self.vault.apply_change("N.md", "a\nc\n")
        self.assertEqual(self.read("N.md"), "a\nb\n")

    def test_allows_insertion(self):
        self.write("N.md", "a\nb\n")
        record = self.vault.apply_change("N.md", "a\nb\nc\n")
        self.assertEqual(record.added_lines, 1)
        self.assertIn("+c", record.diff)
        self.assertNotIn("-a", record.diff)
        self.assertEqual(self.read("N.md"), "a\nb\nc\n")

    def test_write_new_returns_a_full_diff(self):
        record = self.vault.write_new("Fresh.md", "hello\n")
        self.assertIn("--- a/Fresh.md", record.diff)
        self.assertIn("+hello", record.diff)

    def test_write_new_refuses_overwrite(self):
        self.write("N.md", "a\n")
        with self.assertRaises(VaultError):
            self.vault.write_new("N.md", "b\n")

    def test_only_additions(self):
        self.assertTrue(only_additions("a\n", "a\nb\n"))
        self.assertFalse(only_additions("a\n", "b\n"))
        self.assertFalse(only_additions("a", "a\n"))


class InventoryTests(VaultTestCase):
    def test_parse_tags(self):
        text = "---\ntags:\n  - cs/ai/foundationsofai\n  - language/en\n---\n\nbody\n"
        self.assertEqual(parse_tags(text), ["cs/ai/foundationsofai", "language/en"])

    def test_inventory_skips_ignored_directories(self):
        self.write("Note.md", "body\n")
        self.write("University/Course MOC.md", "body\n")
        self.write(".trash/Old.md", "body\n")
        self.write("Attachments/Stuff.md", "body\n")
        paths = [note.path for note in self.vault.inventory()]
        self.assertEqual(paths, ["Note.md", "University/Course MOC.md"])

    def test_match_moc(self):
        self.write("University/Foundations of Artificial Intelligence MOC.md", "x\n")
        self.write("University/Operations Research MOC.md", "x\n")
        matches = self.vault.match_moc("Foundations of Artificial Intelligence")
        self.assertEqual(
            matches[0].path,
            "University/Foundations of Artificial Intelligence MOC.md",
        )

    def test_find_mocs(self):
        self.write("University/Course MOC.md", "x\n")
        self.write("University/Other.md", "x\n")
        self.assertEqual(self.vault.find_mocs(), ["University/Course MOC.md"])

    def test_find_mocs_scans_the_given_roots_recursively(self):
        self.write("Courses/Deep/Foundations MOC.md", "x\n")
        self.write("Other/Ignored.md", "x\n")
        self.assertEqual(
            self.vault.find_mocs([self.root / "Courses"]),
            ["Courses/Deep/Foundations MOC.md"],
        )

    def test_find_mocs_ignores_roots_outside_the_vault(self):
        self.write("University/Course MOC.md", "x\n")
        self.assertEqual(self.vault.find_mocs([Path("/")]), [])

    def test_match_moc_uses_body_content_as_well_as_the_title(self):
        self.write(
            "University/Course A MOC.md",
            "convolutional networks backpropagation gradients\n",
        )
        self.write("University/Course B MOC.md", "cooking recipes\n")
        matches = self.vault.match_moc(
            "convolutional networks backpropagation gradients"
        )
        self.assertEqual(matches[0].path, "University/Course A MOC.md")

    def test_sanitize_links(self):
        body, dropped = self.vault.sanitize_links(
            "See [[Existing]] and [[Missing|missing thing]].",
            {"Existing"},
        )
        self.assertEqual(body, "See [[Existing]] and missing thing.")
        self.assertEqual(dropped, ["Missing"])


if __name__ == "__main__":
    unittest.main()
