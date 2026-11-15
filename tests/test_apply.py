import tempfile
import unittest
from pathlib import Path

from scriptorium.apply import apply_plan, create_note_text, moc_entries
from scriptorium.log import Logger
from scriptorium.models import Flashcard, Plan, PlannedNote
from scriptorium.sandbox import SandboxVault
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

    def test_manual_policy_writes_markers_into_the_note(self):
        self.plan.notes[1].action = "merge"
        self.plan.notes[1].body = "## Section\n\nrewritten line\n"
        conflicts = self.root / "conflicts"
        original = self.read("Existing.md")
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
        conflict = result.conflicts[0]
        self.assertEqual(conflict.path, "Existing.md")
        self.assertEqual(conflict.action, "conflict")
        self.assertIn("<<<<<<< existing", conflict.diff)

        note = self.read("Existing.md")
        prefix = original.split("## Section", 1)[0]
        self.assertTrue(note.startswith(prefix), "frontmatter must stay byte-identical")
        self.assertIn("## Section", note)
        self.assertIn("<<<<<<< existing", note)
        self.assertIn("old line", note)
        self.assertIn("=======", note)
        self.assertIn("rewritten line", note)
        self.assertIn(">>>>>>> scriptorium", note)
        self.assertLess(note.index("old line"), note.index("======="))
        self.assertLess(note.index("======="), note.index("rewritten line"))

        copies = list(conflicts.glob("*.merge.conflict.md"))
        self.assertEqual(len(copies), 1)
        copy = copies[0].read_text(encoding="utf-8")
        self.assertIn("<<<<<<< existing", copy)
        self.assertIn("old line", copy)
        self.assertIn("rewritten line", copy)
        self.assertTrue(any("resolve the conflict markers" in todo.text for todo in result.todos))

    def test_second_manual_merge_skips_when_markers_already_present(self):
        note = PlannedNote(
            action="merge",
            path="Existing.md",
            title="Existing",
            anchor="Section",
            body="## Section\n\nrewritten line\n",
        )
        plan = Plan(course="Course", moc_path=None, notes=[note])
        conflicts = self.root / "conflicts"
        ui = ConsoleUI(yes=False, accept_merges=False, interactive=False)
        apply_plan(
            plan,
            self.vault,
            ui,
            self.logger,
            accept_merges=False,
            conflict_dir=conflicts,
            merge_policy="manual",
        )
        before = self.read("Existing.md")
        result = apply_plan(
            plan,
            self.vault,
            ui,
            self.logger,
            accept_merges=False,
            conflict_dir=conflicts,
            merge_policy="manual",
        )
        self.assertEqual(self.read("Existing.md"), before)
        self.assertIn("Existing.md", result.skipped_merges)
        self.assertEqual(result.conflicts, [])
        self.assertIn("unresolved_conflict_markers", [w.code for w in result.warnings])

    def test_manual_merge_skips_when_markers_are_in_another_section(self):
        (self.root / "Existing.md").write_text(
            "---\ntags:\n  - cs/ai\n---\n\n## Section\n\nold line\n\n## Other\n\n"
            "<<<<<<< existing\nkeep\n=======\nother\n>>>>>>> scriptorium\n",
            encoding="utf-8",
        )
        note = PlannedNote(
            action="merge",
            path="Existing.md",
            title="Existing",
            anchor="Section",
            body="## Section\n\nrewritten line\n",
        )
        plan = Plan(course="Course", moc_path=None, notes=[note])
        before = self.read("Existing.md")
        ui = ConsoleUI(yes=False, accept_merges=False, interactive=False)
        result = apply_plan(
            plan,
            self.vault,
            ui,
            self.logger,
            accept_merges=False,
            conflict_dir=self.root / "conflicts",
            merge_policy="manual",
        )
        self.assertEqual(self.read("Existing.md"), before)
        self.assertIn("Existing.md", result.skipped_merges)
        self.assertEqual(result.conflicts, [])
        self.assertIn("unresolved_conflict_markers", [w.code for w in result.warnings])

    def test_manual_merge_dry_run_writes_sandbox_only(self):
        self.plan.notes[1].action = "merge"
        self.plan.notes[1].body = "## Section\n\nrewritten line\n"
        original = self.read("Existing.md")
        sandbox = SandboxVault.create(self.vault)
        self.addCleanup(sandbox.discard)
        conflicts = self.root / "conflicts"
        ui = ConsoleUI(yes=False, accept_merges=False, interactive=False)
        apply_plan(
            self.plan,
            sandbox,
            ui,
            self.logger,
            accept_merges=False,
            conflict_dir=conflicts,
            merge_policy="manual",
        )
        self.assertEqual(self.read("Existing.md"), original)
        overlay = (sandbox.directory / "Existing.md").read_text(encoding="utf-8")
        self.assertIn("<<<<<<< existing", overlay)
        self.assertIn("old line", overlay)
        self.assertIn("rewritten line", overlay)
        self.assertIn(">>>>>>> scriptorium", overlay)
        self.assertTrue(list(conflicts.glob("*.merge.conflict.md")))

    def test_manual_policy_does_not_turn_an_append_into_markers(self):
        conflicts = self.root / "conflicts"
        original = self.read("Existing.md")
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
        self.assertEqual(result.conflicts, [])
        self.assertIn("Existing.md", result.skipped_merges)
        self.assertEqual(self.read("Existing.md"), original)
        self.assertEqual(list(conflicts.glob("*.merge.conflict.md")), [])

    def test_manual_merge_keeps_the_full_section_body(self):
        long_body = "old " * 50_000 + "TAIL-END"
        (self.root / "Existing.md").write_text(
            "---\ntags:\n  - cs/ai\n---\n\n## Section\n\n" + long_body + "\n\n## Next\n\ntail\n",
            encoding="utf-8",
        )
        note = PlannedNote(
            action="merge",
            path="Existing.md",
            title="Existing",
            anchor="Section",
            body="## Section\n\nrewritten line\n",
        )
        plan = Plan(course="Course", moc_path=None, notes=[note])
        conflicts = self.root / "conflicts"
        ui = ConsoleUI(yes=False, accept_merges=False, interactive=False)
        apply_plan(
            plan,
            self.vault,
            ui,
            self.logger,
            accept_merges=False,
            conflict_dir=conflicts,
            merge_policy="manual",
        )
        self.assertIn("TAIL-END", self.read("Existing.md"))
        copy = next(conflicts.glob("*.merge.conflict.md")).read_text(encoding="utf-8")
        self.assertIn("TAIL-END", copy)

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
