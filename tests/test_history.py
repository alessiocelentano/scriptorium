"""Completed-run history: the snapshot round-trips the text each box showed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scriptorium.history import HistoryEntry, HistoryStore, record_completed_run
from scriptorium.models import Aside, Flashcard, Plan, PlannedNote


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

    def test_record_completed_run_round_trips_every_box(self):
        self.write_artifacts()
        record_completed_run(
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
        record_completed_run(
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


if __name__ == "__main__":
    unittest.main()
