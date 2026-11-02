import tempfile
import unittest
from pathlib import Path

from scriptorium.apply import apply_plan, create_note_text, moc_entries
from scriptorium.log import Logger
from scriptorium.models import Flashcard, Plan, PlannedNote
from scriptorium.ui import ConsoleUI
from scriptorium.vault import Vault


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = Vault(self.root)
        (self.root / "University").mkdir()
        (self.root / "University" / "Course MOC.md").write_text(
            "## Table of Contents\n\n- [[Existing]]\n\n## Useful Links\n",
            encoding="utf-8",
        )
        (self.root / "Existing.md").write_text(
            "---\ntags:\n  - cs/ai\n---\n\n## Section\n\nold line\n",
            encoding="utf-8",
        )
        self.plan = Plan(
            course="Course",
            moc_path="University/Course MOC.md",
            notes=[
                PlannedNote(
                    action="create",
                    path="New Note.md",
                    title="New Note",
                    tags=["cs/ai"],
                    body="## Fresh\n\nNew body.",
                    flashcards=[Flashcard(front="What is new?", back="A body.")],
                ),
                PlannedNote(
                    action="append",
                    path="Existing.md",
                    title="Existing",
                    anchor="Section",
                    body="merged line",
                ),
            ],
        )
        self.logger = Logger()

    def tearDown(self):
        self._tmp.cleanup()

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_create_is_automatic_and_merge_is_skipped(self):
        ui = ConsoleUI(yes=True, accept_merges=False, interactive=False)
        result = apply_plan(self.plan, self.vault, ui, self.logger, accept_merges=False)
        self.assertEqual([record.path for record in result.created], ["New Note.md"])
        self.assertEqual(result.skipped_merges, ["Existing.md"])
        self.assertEqual(
            self.read("Existing.md"),
            "---\ntags:\n  - cs/ai\n---\n\n## Section\n\nold line\n",
        )
        new_text = self.read("New Note.md")
        self.assertTrue(new_text.startswith("---\ntags:\n  - cs/ai\n---\n\n## Fresh"))
        moc = self.read("University/Course MOC.md")
        self.assertIn("- [[New Note]]", moc)
        self.assertIn("\t- Fresh", moc)
        self.assertEqual(len(result.moc), 1)

    def test_accept_merges_applies_insertion(self):
        ui = ConsoleUI(yes=True, accept_merges=True, interactive=False)
        result = apply_plan(self.plan, self.vault, ui, self.logger, accept_merges=True)
        self.assertEqual([record.path for record in result.merged], ["Existing.md"])
        text = self.read("Existing.md")
        self.assertIn("old line\n\nmerged line", text)
        self.assertTrue(text.startswith("---\ntags:\n  - cs/ai\n---\n"))

    def test_merge_action_replaces_the_section(self):
        self.plan.notes[1].action = "merge"
        self.plan.notes[1].body = "## Section\n\nrewritten line\n"
        ui = ConsoleUI(yes=True, accept_merges=True, interactive=False)
        result = apply_plan(self.plan, self.vault, ui, self.logger, accept_merges=True)
        text = self.read("Existing.md")
        self.assertIn("rewritten line", text)
        self.assertNotIn("old line", text)
        self.assertEqual(result.merged[0].removed_lines, 1)
        self.assertEqual(result.merged[0].action, "merge")

    def test_merge_without_a_heading_keeps_the_section_title(self):
        self.plan.notes[1].action = "merge"
        self.plan.notes[1].body = "rewritten line without heading"
        ui = ConsoleUI(yes=True, accept_merges=True, interactive=False)
        apply_plan(self.plan, self.vault, ui, self.logger, accept_merges=True)
        text = self.read("Existing.md")
        self.assertIn("## Section", text)
        self.assertIn("rewritten line without heading", text)
        self.assertNotIn("old line", text)

    def test_merge_needs_review_by_default(self):
        self.plan.notes[1].action = "merge"
        self.plan.notes[1].body = "## Section\n\nrewritten line\n"
        ui = ConsoleUI(yes=False, accept_merges=False, interactive=False)
        result = apply_plan(self.plan, self.vault, ui, self.logger, accept_merges=False)
        self.assertEqual(result.skipped_merges, ["Existing.md"])
        self.assertIn("old line", self.read("Existing.md"))

    def test_manual_policy_writes_a_conflict_file_and_leaves_the_note(self):
        self.plan.notes[1].action = "merge"
        self.plan.notes[1].body = "## Section\n\nrewritten line\n"
        conflicts = self.root / "conflicts"
        ui = ConsoleUI(yes=False, accept_merges=False, interactive=False)
        result = apply_plan(
            self.plan,
            self.vault,
            ui,
            self.logger,
            accept_merges=False,
            conflict_dir=conflicts,
            merge_policy="manual",
        )
        self.assertEqual(len(result.conflicts), 1)
        conflict = Path(result.conflicts[0])
        text = conflict.read_text(encoding="utf-8")
        self.assertIn("<<<<<<< existing", text)
        self.assertIn("old line", text)
        self.assertIn("=======", text)
        self.assertIn("rewritten line", text)
        self.assertIn(">>>>>>> scriptorium", text)
        self.assertIn("old line", self.read("Existing.md"))
        self.assertTrue(any("Manual merge" in todo.text for todo in result.todos))

    def test_refuses_create_inside_university(self):
        plan = Plan(
            course="Course",
            moc_path="University/Course MOC.md",
            notes=[
                PlannedNote(
                    action="create",
                    path="University/Bad.md",
                    title="Bad",
                    body="## X\n\ntext",
                )
            ],
        )
        result = apply_plan(
            plan,
            self.vault,
            ConsoleUI(yes=True, interactive=False),
            self.logger,
            accept_merges=False,
        )
        self.assertEqual(result.refused, ["University/Bad.md"])
        self.assertFalse((self.root / "University" / "Bad.md").exists())


class HelperTests(unittest.TestCase):
    def test_create_note_text(self):
        note = PlannedNote(
            action="create", path="N.md", title="N", tags=["cs/a", "cs/b"], body="Body."
        )
        self.assertEqual(
            create_note_text(note),
            "---\ntags:\n  - cs/a\n  - cs/b\n---\n\nBody.\n",
        )

    def test_moc_entries_nesting(self):
        text = "## A\n\n### B\n\n## Flashcards\n"
        self.assertEqual(moc_entries("Title", text), "- [[Title]]\n\t- A\n\t\t- B")


if __name__ == "__main__":
    unittest.main()
