import tempfile
import unittest
from pathlib import Path

from scriptorium.errors import StageError
from scriptorium.state import JobStore, Stage


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


if __name__ == "__main__":
    unittest.main()
