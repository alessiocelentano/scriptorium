"""Run history: the snapshot round-trips the text each box showed and its status."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scriptorium.history import (
    HistoryEntry,
    HistoryStore,
    record_run,
    state_status,
    status_text,
)
from scriptorium.models import Aside, Flashcard, Plan, PlannedNote
from scriptorium.state import GLOBAL_STAGES, GROUP_STAGES, GroupState, JobState


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        self.job_dir = self.root / "jobs" / "job-1"
        self.job_dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def write_artifacts(self) -> None:
        (self.job_dir / "apply.json").write_text(
            json.dumps(
                {
                    "created": [
                        {
                            "path": "New.md",
                            "action": "create",
                            "added_lines": 2,
                            "removed_lines": 0,
                            "diff": "--- /dev/null\n+++ b/New.md\n@@ -0,0 +1,2 @@\n+one\n+two",
                        }
                    ],
                    "merged": [],
                    "moc": [],
                }
            ),
            encoding="utf-8",
        )
        plan = Plan(
            course="Test Course",
            notes=[
                PlannedNote(
                    action="create",
                    path="New.md",
                    title="New",
                    flashcards=[Flashcard(front="What is X?", back="Y.")],
                )
            ],
            asides=[Aside(kind="exam", text="The exam is oral.", source="00:00:02")],
        )
        (self.job_dir / "05.plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
        (self.job_dir / "cards.json").write_text(
            json.dumps(
                [
                    {
                        "path": "New.md",
                        "requested": 1,
                        "written": 1,
                        "skipped": 0,
                        "writer": "internal",
                    }
                ]
            ),
            encoding="utf-8",
        )

    def test_record_run_round_trips_every_box(self):
        self.write_artifacts()
        record_run(
            self.home / "history",
            self.job_dir,
            job_id="job-1",
            course="Test Course",
            sources=["/lectures/lecture.mp4"],
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:05:00+00:00",
        )

        entries = HistoryStore(self.home / "history").entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.course, "Test Course")
        self.assertEqual(entry.finished_at, "2026-01-01T10:05:00+00:00")
        self.assertEqual(entry.sources, ["/lectures/lecture.mp4"])
        self.assertEqual(len(entry.changes), 1)
        self.assertEqual(entry.changes[0].path, "New.md")
        self.assertIn("+one", entry.changes[0].diff)
        self.assertEqual(entry.card_count, 1)
        self.assertEqual(entry.card_groups[0].cards[0].front, "What is X?")
        self.assertEqual(entry.extras[0].text, "The exam is oral.")

    def test_saving_the_same_job_twice_rewrites_one_file(self):
        store = HistoryStore(self.home / "history")
        entry = HistoryEntry(
            id="job-1",
            job_id="job-1",
            course="Course",
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:00:00+00:00",
        )
        store.save(entry)
        store.save(entry)
        self.assertEqual(len(list(store.directory.glob("*.json"))), 1)

    def test_corrupt_snapshots_are_skipped(self):
        self.write_artifacts()
        store = HistoryStore(self.home / "history")
        store.directory.mkdir(parents=True)
        (store.directory / "broken.json").write_text("{", encoding="utf-8")
        record_run(
            self.home / "history",
            self.job_dir,
            job_id="job-2",
            course="Course",
            sources=[],
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:05:00+00:00",
        )
        entries = HistoryStore(self.home / "history").entries()
        self.assertEqual([entry.job_id for entry in entries], ["job-2"])

    def test_a_failed_run_carries_its_stage_and_stays_resumable(self):
        entry = HistoryEntry(
            id="job-1",
            job_id="job-1",
            course="Course",
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:01:00+00:00",
            status="failed",
            error="compose: the model returned invalid JSON",
        )
        payload = json.loads(json.dumps(entry.to_dict()))
        loaded = HistoryEntry.from_dict(payload)
        self.assertEqual(loaded.status, "failed")
        self.assertEqual(loaded.status_label, "failed at compose")
        self.assertTrue(loaded.resumable)
        self.assertIn("failed at compose", loaded.label)

    def test_a_running_run_is_not_completed_but_stays_resumable(self):
        entry = HistoryEntry(
            id="job-1",
            job_id="job-1",
            course="Course",
            created_at="2026-01-01T10:00:00+00:00",
            finished_at="2026-01-01T10:00:00+00:00",
            status="running",
        )
        self.assertEqual(entry.status_label, "in progress")
        self.assertTrue(entry.resumable)

    def test_old_snapshots_without_a_status_count_as_completed(self):
        payload = {
            "id": "job-1",
            "job_id": "job-1",
            "course": "Course",
            "created_at": "2026-01-01T10:00:00+00:00",
            "finished_at": "2026-01-01T10:05:00+00:00",
        }
        entry = HistoryEntry.from_dict(payload)
        self.assertEqual(entry.status, "completed")
        self.assertFalse(entry.resumable)

    def test_newest_finished_run_comes_first(self):
        store = HistoryStore(self.home / "history")
        runs = (
            ("older", "2026-01-01T10:00:00+00:00"),
            ("newer", "2026-01-02T10:00:00+00:00"),
        )
        for job_id, finished in runs:
            store.save(
                HistoryEntry(
                    id=job_id,
                    job_id=job_id,
                    course=job_id,
                    created_at=finished,
                    finished_at=finished,
                )
            )
        self.assertEqual([entry.job_id for entry in store.entries()], ["newer", "older"])


class RunStatusTests(unittest.TestCase):
    def test_status_text_names_each_state(self):
        self.assertEqual(status_text("completed"), "completed")
        self.assertEqual(status_text("running"), "in progress")
        self.assertEqual(status_text("dry_run"), "dry run · ready to apply")
        self.assertEqual(status_text("failed", "outline: boom"), "failed at outline")
        self.assertEqual(status_text("failed"), "failed")
        self.assertEqual(status_text("incomplete"), "incomplete")

    def test_state_status_reads_the_job_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            state = JobState(
                job_id="job-1",
                video=str(Path(directory) / "lecture.mp4"),
                course="Course",
            )
            self.assertEqual(state_status(state), ("incomplete", ""))

            state.error = "compose: boom"
            self.assertEqual(state_status(state), ("failed", "compose: boom"))

            for stage in GLOBAL_STAGES:
                state.set_status(stage, "done")
            group = GroupState(id="g1", topic="Topic")
            for stage in GROUP_STAGES:
                group.set_status(stage, "done")
            state.groups = [group]
            self.assertEqual(state_status(state), ("completed", ""))

            state.options["dry_run"] = True
            self.assertEqual(state_status(state), ("dry_run", ""))


if __name__ == "__main__":
    unittest.main()
