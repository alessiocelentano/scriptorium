import tempfile
import unittest
from pathlib import Path

from scriptorium.errors import VaultError
from scriptorium.sandbox import SandboxVault
from scriptorium.vault import Vault


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault_dir = self.root / "vault"
        self.vault_dir.mkdir()
        (self.vault_dir / "Existing.md").write_text("## Section\n\nold\n", encoding="utf-8")
        self.vault = Vault(self.vault_dir)
        self.sandbox = SandboxVault.create(self.vault)

    def tearDown(self):
        self.sandbox.discard()
        self._tmp.cleanup()

    def test_reads_fall_through_to_the_real_vault(self):
        self.assertTrue(self.sandbox.exists("Existing.md"))
        self.assertIn("old", self.sandbox.read("Existing.md"))

    def test_new_notes_stay_in_the_sandbox(self):
        self.sandbox.write_new("Fresh.md", "## A\n\nbody\n")
        self.assertTrue((self.sandbox.directory / "Fresh.md").is_file())
        self.assertFalse((self.vault_dir / "Fresh.md").exists())
        self.assertTrue(self.sandbox.exists("Fresh.md"))
        self.assertEqual(self.sandbox.read("Fresh.md"), "## A\n\nbody\n")

    def test_inventory_merges_the_overlay(self):
        self.sandbox.write_new("Fresh.md", "---\ntags:\n  - cs/x\n---\n\n## A\n")
        paths = [note.path for note in self.sandbox.inventory()]
        self.assertEqual(paths, ["Existing.md", "Fresh.md"])
        fresh = self.sandbox.note_info("Fresh.md")
        assert fresh is not None
        self.assertEqual(fresh.tags, ("cs/x",))
        self.assertEqual(fresh.headings, ("A",))

    def test_insert_only_rule_is_enforced(self):
        original = self.sandbox.read("Existing.md")
        with self.assertRaises(VaultError):
            self.sandbox.apply_change("Existing.md", "## Section\n\nreplaced\n")
        self.assertEqual(self.sandbox.read("Existing.md"), original)
        self.assertIn("old", (self.vault_dir / "Existing.md").read_text())
        overlay = self.sandbox.directory / "Existing.md"
        if overlay.exists():
            self.assertEqual(overlay.read_text(encoding="utf-8"), original)

    def test_insertion_is_allowed_and_diffed(self):
        record = self.sandbox.apply_change(
            "Existing.md", "## Section\n\nold\n\nnew line\n", action="merge"
        )
        self.assertEqual(record.action, "merge")
        self.assertIn("+new line", record.diff)
        self.assertIn("new line", self.sandbox.read("Existing.md"))
        self.assertNotIn("new line", (self.vault_dir / "Existing.md").read_text())

    def test_discard_removes_everything(self):
        directory = self.sandbox.directory
        self.sandbox.write_new("Fresh.md", "x\n")
        self.sandbox.discard()
        self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
