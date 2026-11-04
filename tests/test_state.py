import tempfile
import unittest
from pathlib import Path

from scriptorium.errors import StageError
from scriptorium.state import (
    GLOBAL_STAGES,
    GROUP_STAGES,
    GroupState,
    JobState,
    JobStore,
    Stage,
)


class StateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.store = JobStore(self.home)
        self.video = self.home / "lecture.mp4"
        self.video.write_bytes(b"fake")

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_save_load(self):
        state = self.store.create([self.video], "Course")
        state.set_status(Stage.SOURCES, "done")
        state.set_status(Stage.PREPARE, "done")
        state.set_status(Stage.AUDIO, "done")
        self.store.save(state)
        loaded = self.store.load(state.job_id)
        self.assertEqual(loaded.course, "Course")
        self.assertEqual(loaded.status(Stage.AUDIO), "done")
        self.assertIs(loaded.next_stage(), Stage.TRANSCRIBE)

    def test_recent(self):
        first = self.store.create([self.video], "A")
        second = self.store.create([self.video], "B")
        ids = [state.job_id for state in self.store.recent()]
        self.assertIn(first.job_id, ids)
        self.assertIn(second.job_id, ids)

    def test_lock_blocks_second_process(self):
        state = self.store.create([self.video], "Course")
        with self.store.lock(state.job_id):
            with self.assertRaises(StageError):
                with self.store.lock(state.job_id):
                    pass
        # after release the lock is usable again
        with self.store.lock(state.job_id):
            pass

    def test_invalid_job_id(self):
        with self.assertRaises(StageError):
            self.store.load("../escape")

    def test_missing_job(self):
        with self.assertRaises(StageError):
            self.store.load("does-not-exist")

    def _finished(self, job_id="20260101-120000-alpha"):
        state = JobState(job_id=job_id, video=str(self.video), course="A")
        for stage in GLOBAL_STAGES:
            state.set_status(stage, "done")
        group = GroupState(id="g1", topic="Topic")
        for stage in GROUP_STAGES:
            group.set_status(stage, "done")
        state.groups = [group]
        self.store.save(state)
        return state

    def test_resolve_accepts_the_exact_id_and_a_unique_prefix(self):
        state = self._finished("20260101-120000-alpha")
        self.assertEqual(self.store.resolve(state.job_id).job_id, state.job_id)
        self.assertEqual(self.store.resolve("20260101-1200").job_id, state.job_id)

    def test_resolve_reports_an_ambiguous_prefix(self):
        self._finished("20260101-120000-alpha")
        self._finished("20260101-120000-beta")
        with self.assertRaises(StageError) as caught:
            self.store.resolve("20260101-120000")
        self.assertIn("ambiguous", str(caught.exception))
        self.assertIn("alpha", str(caught.exception))

    def test_resolve_reports_an_unknown_id(self):
        with self.assertRaises(StageError) as caught:
            self.store.resolve("20260101-000000-missing")
        self.assertIn("job not found", str(caught.exception))

    def test_resumable_skips_completed_jobs_but_keeps_dry_runs(self):
        finished = self._finished("20260101-120000-alpha")
        dry = self._finished("20260102-120000-beta")
        dry.options["dry_run"] = True
        self.store.save(dry)
        unfinished = self.store.create([self.video], "C")

        ids = [state.job_id for state in self.store.resumable()]
        self.assertNotIn(finished.job_id, ids)
        self.assertIn(dry.job_id, ids)
        self.assertIn(unfinished.job_id, ids)


if __name__ == "__main__":
    unittest.main()
