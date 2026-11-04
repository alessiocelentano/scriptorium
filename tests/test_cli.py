"""The CLI wiring: the resume subcommand and its default job selection."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.cli import build_parser, main
from scriptorium.config import Config
from scriptorium.log import Logger
from scriptorium.pipeline import Pipeline
from scriptorium.state import GLOBAL_STAGES, GROUP_STAGES, GroupState, JobStore, Stage
from scriptorium.ui import ConsoleUI


class ResumeCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.source = self.root / "lecture.mp4"
        self.source.write_bytes(b"x")
        self.config = Config(notes_root=self.vault, artifacts_dir=self.root / "home")
        self.store = JobStore(self.config.artifacts_dir)
        self.ui = ConsoleUI(yes=True, accept_merges=True, interactive=False)
        self.logger = Logger()
        self._key = mock.patch.dict(os.environ, {"SCRIPTORIUM_LLM_API_KEY": "test-key"})
        self._key.start()

    def tearDown(self):
        self._key.stop()
        self._tmp.cleanup()

    def _job(self, *, finished=False, error=None, dry_run=False):
        state = self.store.create([self.source], "Test Course")
        state.set_status(Stage.SOURCES, "done")
        state.error = error
        state.options["dry_run"] = dry_run
        if finished:
            for stage in GLOBAL_STAGES:
                state.set_status(stage, "done")
            group = GroupState(id="g1", topic="Topic")
            for stage in GROUP_STAGES:
                group.set_status(stage, "done")
            state.groups = [group]
        self.store.save(state)
        return state

    def _run(self, argv):
        context = mock.patch(
            "scriptorium.cli._pipeline_context",
            return_value=(self.config, self.ui, self.logger),
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with context, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_the_parser_accepts_resume_with_and_without_an_id(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["resume"]).command, "resume")
        self.assertIsNone(parser.parse_args(["resume"]).job_id)
        self.assertEqual(parser.parse_args(["resume", "job-1"]).job_id, "job-1")

    def test_resume_without_an_id_picks_the_most_recent_unfinished_job(self):
        self._job(finished=True)
        state = self._job(error="outline: boom")

        with mock.patch.object(Pipeline, "resume", return_value=state) as resume:
            code, stdout, _ = self._run(["resume"])

        self.assertEqual(code, 0)
        resume.assert_called_once()
        self.assertEqual(resume.call_args.args[0], state.job_id)
        self.assertIn(f"resuming {state.job_id}", stdout)
        self.assertIn("failed at outline", stdout)

    def test_resume_without_an_id_and_nothing_unfinished_is_an_error(self):
        self._job(finished=True)
        code, _, stderr = self._run(["resume"])
        self.assertEqual(code, 1)
        self.assertIn("no unfinished job to resume", stderr)
        self.assertIn("scriptorium run", stderr)

    def test_resume_accepts_a_unique_job_id_prefix(self):
        state = self._job(error="outline: boom")
        prefix = state.job_id.rsplit("-", 1)[0]

        with mock.patch.object(Pipeline, "resume", return_value=state) as resume:
            code, _, _ = self._run(["resume", prefix])

        self.assertEqual(code, 0)
        self.assertEqual(resume.call_args.args[0], state.job_id)

    def test_resume_reports_an_unknown_id(self):
        code, _, stderr = self._run(["resume", "20260101-000000-missing"])
        self.assertEqual(code, 1)
        self.assertIn("job not found", stderr)

    def test_resume_reports_an_ambiguous_prefix(self):
        first = self._job(error="outline: boom")
        second = self.store.create([self.source], "Other Course")
        self.store.save(second)
        prefix = first.job_id.rsplit("-", 1)[0]
        self.assertTrue(second.job_id.startswith(prefix))

        code, _, stderr = self._run(["resume", prefix])
        self.assertEqual(code, 1)
        self.assertIn("ambiguous", stderr)

    def test_resume_on_a_completed_job_is_a_no_op(self):
        state = self._job(finished=True)

        with mock.patch.object(Pipeline, "resume") as resume:
            code, stdout, _ = self._run(["resume", state.job_id])

        self.assertEqual(code, 0)
        resume.assert_not_called()
        self.assertIn("already complete. Nothing to resume", stdout)

    def test_resume_applies_a_dry_run(self):
        state = self._job(finished=True, dry_run=True)

        with mock.patch.object(Pipeline, "resume", return_value=state) as resume:
            code, stdout, _ = self._run(["resume"])

        self.assertEqual(code, 0)
        resume.assert_called_once()
        self.assertIn("dry run · ready to apply", stdout)


if __name__ == "__main__":
    unittest.main()
