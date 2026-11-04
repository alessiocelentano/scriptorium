import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.asr import ApiTranscriber, LocalTranscriber
from scriptorium.audio import MediaInfo
from scriptorium.config import Config
from scriptorium.errors import ConfigurationError, LLMError, StageError, TranscribeError
from scriptorium.log import Logger
from scriptorium.models import Plan, Segment, Transcript
from scriptorium.pdf_images import ExtractedImage, ImageHarvest
from scriptorium.pipeline import Pipeline, RunOptions
from scriptorium.sources import ReadResult
from scriptorium.state import Stage
from scriptorium.ui import ConsoleUI


def extract_response():
    return {
        "units": [
            {
                "kind": "definition",
                "title": "Heuristic Function",
                "content": "A heuristic estimates the cost to the goal.",
                "source": "00:00:00",
                "terms": ["heuristic"],
                "confidence": 0.9,
            }
        ],
        "asides": [
            {
                "kind": "exam",
                "text": "The exam is oral and covers chapters 1-4.",
                "source": "00:00:02",
            }
        ],
        "topics": ["Heuristics"],
    }


def group_response():
    return {
        "groups": [
            {
                "id": "g1",
                "topic": "Heuristics",
                "source_ids": ["s01"],
                "rationale": "single source",
                "confidence": 0.9,
            }
        ],
        "todos": [],
    }


def outline_response():
    return {
        "notes": [
            {
                "action": "create",
                "path": "Test Note.md",
                "title": "Test Note",
                "tags": ["cs/test"],
                "unit_ids": ["u1"],
                "confidence": 0.9,
            }
        ],
        "todos": [],
    }


def compose_response():
    return {
        "body": "## Fresh\n\nSee [[Existing]].\n",
        "flashcards": [
            {
                "front": "What is a heuristic function?",
                "back": "An estimate of the cost to the goal.",
                "kind": "single",
            }
        ],
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault_dir = self.root / "vault"
        self.home = self.root / "home"
        (self.vault_dir / "University").mkdir(parents=True)
        (self.vault_dir / "University" / "Test Course MOC.md").write_text(
            "## Table of Contents\n", encoding="utf-8"
        )
        (self.vault_dir / "Existing.md").write_text(
            "---\ntags:\n  - cs/test\n---\n\n## Section\n\nold\n", encoding="utf-8"
        )
        self.video = self.root / "lecture.mp4"
        self.video.write_bytes(b"fake video")
        self._old_key = os.environ.get("SCRIPTORIUM_LLM_API_KEY")
        os.environ["SCRIPTORIUM_LLM_API_KEY"] = "test-key"
        self.config = Config(
            notes_root=self.vault_dir,
            artifacts_dir=self.home,
        )
        self.ui = ConsoleUI(yes=True, accept_merges=True, interactive=False)
        self.logger = Logger()
        self.prompts: list[tuple[str, str]] = []
        self.group_payload = group_response()

    def tearDown(self):
        self._tmp.cleanup()
        if self._old_key is None:
            os.environ.pop("SCRIPTORIUM_LLM_API_KEY", None)
        else:
            os.environ["SCRIPTORIUM_LLM_API_KEY"] = self._old_key

    def test_local_backend_builds_the_whisper_transcriber(self):
        transcriber = Pipeline(self.config, self.ui, self.logger)._transcriber()
        self.assertIsInstance(transcriber, LocalTranscriber)

    def test_api_backend_builds_the_api_transcriber(self):
        config = self.config.with_overrides(
            asr_api_base_url="https://api.openai.com/v1", asr_api_model="whisper-1"
        )
        with mock.patch.dict(os.environ, {"SCRIPTORIUM_ASR_API_KEY": "test-key"}):
            transcriber = Pipeline(config, self.ui, self.logger)._transcriber()
        self.assertIsInstance(transcriber, ApiTranscriber)
        self.assertEqual(transcriber.settings.api_key, "test-key")

    def fake_to_mp3(self, video, destination, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake mp3")
        return MediaInfo(duration=1.0, has_audio=True, has_video=False)

    def fake_transcribe(self, audio, prompt=None):
        return Transcript(
            language="en",
            duration=2.0,
            segments=[Segment(start=0, end=2, text="A heuristic estimates the cost.")],
        )

    def fake_chat(self, system, user, *, phase="llm", temperature=0.2, max_tokens=8192):
        self.prompts.append((phase, user))
        if phase == "group":
            return self.group_payload
        return {
            "extract": extract_response,
            "outline": outline_response,
            "compose": compose_response,
        }[phase]()

    def run_pipeline(self, options, sources=None):
        with (
            mock.patch("scriptorium.pipeline.to_mp3", self.fake_to_mp3),
            mock.patch("scriptorium.asr.LocalTranscriber.transcribe", self.fake_transcribe),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", self.fake_chat),
        ):
            pipeline = Pipeline(self.config, self.ui, self.logger)
            return pipeline.run(sources or [self.video], options)

    def test_full_run_creates_note_cards_and_report(self):
        state = self.run_pipeline(RunOptions(course="Test Course", accept_merges=True))
        self.assertTrue(state.completed())
        self.assertIsNone(state.error)

        note = (self.vault_dir / "Test Note.md").read_text(encoding="utf-8")
        self.assertIn("---\ntags:\n  - cs/test\n---", note)
        self.assertIn("<!--DRAFT: What is a heuristic function?", note)

        moc = (self.vault_dir / "University" / "Test Course MOC.md").read_text(encoding="utf-8")
        self.assertIn("- [[Test Note]]", moc)

        job_dir = self.home / "jobs" / state.job_id
        self.assertTrue((job_dir / "02.transcript.txt").is_file())
        self.assertTrue((job_dir / "05.plan.json").is_file())
        plan = Plan.model_validate_json((job_dir / "05.plan.json").read_text())
        self.assertEqual(len(plan.asides), 1)
        self.assertEqual(plan.asides[0].kind, "exam")
        self.assertIn("lecture.mp4", plan.asides[0].source)
        self.assertTrue((job_dir / "report.md").is_file())
        self.assertTrue((self.home / "history" / f"{state.job_id}.json").is_file())
        self.assertFalse((job_dir / "sources" / "s01" / "audio.mp3").exists())
        self.assertFalse((job_dir / "03.extract.json").exists())
        self.assertTrue(self.video.exists(), "sources are never deleted")

    def test_dry_run_writes_nothing_and_discards_the_sandbox(self):
        existing = self.vault_dir / "Test Note.md"
        existing.write_text(
            "---\ntags:\n  - cs/test\n---\n\n## Old\n\nold body\n", encoding="utf-8"
        )
        before = existing.read_bytes()

        state = self.run_pipeline(RunOptions(course="Test Course", dry_run=True))

        self.assertEqual(existing.read_bytes(), before, "dry run must not touch the vault")
        self.assertEqual(state.groups[0].status("apply"), "skipped")
        job_dir = self.home / "jobs" / state.job_id
        applied = json.loads((job_dir / "apply.json").read_text(encoding="utf-8"))
        self.assertTrue(applied["merged"] or applied["created"])
        report = (job_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("Dry run", report)
        history = json.loads(
            (self.home / "history" / f"{state.job_id}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(history["status"], "dry_run")
        leftovers = list(Path(tempfile.gettempdir()).glob("scriptorium-dry-run-*"))
        self.assertEqual(leftovers, [], "the sandbox must be deleted at the end")

    def test_sources_are_processed_in_topic_groups(self):
        notes = self.root / "notes.md"
        notes.write_text("Notes about heuristic search.", encoding="utf-8")
        self.group_payload = {
            "groups": [
                {"id": "g1", "topic": "Lecture Foundations", "source_ids": ["s01"]},
                {"id": "g2", "topic": "Search Notes", "source_ids": ["s02"]},
            ],
            "todos": [],
        }

        state = self.run_pipeline(
            RunOptions(course="Test Course", accept_merges=True),
            [self.video, notes],
        )

        self.assertEqual(
            [group.topic for group in state.groups],
            ["Lecture Foundations", "Search Notes"],
        )
        job_dir = self.home / "jobs" / state.job_id
        self.assertTrue((job_dir / "apply.g1.json").is_file())
        self.assertTrue((job_dir / "apply.g2.json").is_file())
        merged = Plan.model_validate_json((job_dir / "05.plan.json").read_text())
        self.assertEqual(len(merged.notes), 2)
        outlines = [prompt for phase, prompt in self.prompts if phase == "outline"]
        self.assertEqual(len(outlines), 2)
        self.assertIn("Test Note.md", outlines[1])
        report = (job_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Topic groups", report)
        self.assertIn("Search Notes", report)

    def test_extract_waits_for_every_transcript(self):
        second = self.root / "second.mp4"
        second.write_bytes(b"fake video")
        self.group_payload = {
            "groups": [{"id": "g1", "topic": "Everything", "source_ids": ["s01", "s02"]}],
            "todos": [],
        }
        transcripts_at_extract: list[set[str]] = []
        original = self.fake_chat

        def spy(_client, system, user, *, phase="llm", temperature=0.2, max_tokens=8192):
            if phase == "extract":
                # The barrier: no text model call may see a partial set.
                job = next((self.home / "jobs").iterdir())
                transcripts_at_extract.append(
                    {path.parent.name for path in job.glob("sources/*/transcript.json")}
                )
            return original(
                system, user, phase=phase, temperature=temperature, max_tokens=max_tokens
            )

        with (
            mock.patch("scriptorium.pipeline.to_mp3", self.fake_to_mp3),
            mock.patch("scriptorium.asr.LocalTranscriber.transcribe", self.fake_transcribe),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", spy),
        ):
            state = Pipeline(self.config, self.ui, self.logger).run(
                [self.video, second], RunOptions(course="Test Course", accept_merges=True)
            )

        self.assertTrue(state.completed())
        # One extract call per chunk: every one of them must already see both.
        self.assertTrue(transcripts_at_extract)
        for seen in transcripts_at_extract:
            self.assertEqual(seen, {"s01", "s02"})

    def test_a_failed_source_keeps_the_others_and_resume_finishes_it(self):
        second = self.root / "second.mp4"
        second.write_bytes(b"fake video")
        self.group_payload = {
            "groups": [{"id": "g1", "topic": "Everything", "source_ids": ["s01", "s02"]}],
            "todos": [],
        }

        def one_source_down(_transcriber, audio, prompt=None):
            if audio.parent.name == "s02":
                raise TranscribeError("HTTP 500: upstream is down")
            return self.fake_transcribe(audio, prompt=prompt)

        pipeline = Pipeline(self.config, self.ui, self.logger)
        with (
            mock.patch("scriptorium.pipeline.to_mp3", self.fake_to_mp3),
            mock.patch("scriptorium.asr.LocalTranscriber.transcribe", one_source_down),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", self.fake_chat),
            self.assertRaises(StageError) as caught,
        ):
            pipeline.run([self.video, second], RunOptions(course="Test Course"))

        self.assertIn("second.mp4", str(caught.exception))
        job = next((self.home / "jobs").iterdir())
        self.assertTrue((job / "sources" / "s01" / "transcript.json").is_file())
        self.assertFalse((job / "sources" / "s02" / "transcript.json").exists())

        asked: list[str] = []

        def working(_transcriber, audio, prompt=None):
            asked.append(audio.parent.name)
            return self.fake_transcribe(audio, prompt=prompt)

        with (
            mock.patch("scriptorium.pipeline.to_mp3", self.fake_to_mp3),
            mock.patch("scriptorium.asr.LocalTranscriber.transcribe", working),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", self.fake_chat),
        ):
            state = pipeline.resume(job.name, RunOptions(accept_merges=True))

        self.assertEqual(asked, ["s02"], "the finished transcript must not be redone")
        self.assertTrue(state.completed())

    def test_grouping_failure_falls_back_to_one_group(self):
        notes = self.root / "notes.md"
        notes.write_text("Notes about heuristic search.", encoding="utf-8")
        self.group_payload = None

        state = self.run_pipeline(
            RunOptions(course="Test Course", accept_merges=True),
            [self.video, notes],
        )

        self.assertEqual(len(state.groups), 1)
        self.assertEqual(state.groups[0].source_ids, ["s01", "s02"])
        codes = [warning["code"] for warning in state.warnings]
        self.assertIn("grouping_fallback", codes)
        self.assertTrue(state.completed())

    def test_missing_moc_is_created_and_indexed(self):
        (self.vault_dir / "University" / "Test Course MOC.md").unlink()

        state = self.run_pipeline(RunOptions(course="New Course", accept_merges=True))

        self.assertEqual(state.moc, "University/New Course MOC.md")
        moc = (self.vault_dir / "University" / "New Course MOC.md").read_text(encoding="utf-8")
        self.assertIn("# New Course MOC", moc)
        self.assertIn("- [[Test Note]]", moc)
        codes = [warning["code"] for warning in state.warnings]
        self.assertIn("moc_not_found", codes)
        self.assertIn("moc_created", codes)

    def test_document_only_run_produces_a_note(self):
        slides = self.root / "slides.md"
        slides.write_text("Slides about heuristic functions.", encoding="utf-8")

        state = self.run_pipeline(RunOptions(course="Test Course", accept_merges=True), [slides])

        self.assertTrue(state.completed())
        self.assertEqual(state.sources, [str(slides)])
        job_dir = self.home / "jobs" / state.job_id
        self.assertIn("Slides about heuristic", (job_dir / "sources" / "s01.txt").read_text())
        self.assertFalse((job_dir / "sources" / "s01" / "audio.mp3").exists())
        self.assertTrue((self.vault_dir / "Test Note.md").is_file())

    def test_every_source_is_read_and_kept(self):
        notes = self.root / "notes.md"
        notes.write_text("Notes about heuristic search.", encoding="utf-8")

        state = self.run_pipeline(
            RunOptions(course="Test Course", accept_merges=True), [self.video, notes]
        )

        job_dir = self.home / "jobs" / state.job_id
        self.assertIn("heuristic search", (job_dir / "sources" / "s02.txt").read_text())
        self.assertIn("heuristic estimates", (job_dir / "sources" / "s01.txt").read_text())
        self.assertTrue(self.video.is_file(), "sources are never deleted")
        self.assertTrue(notes.is_file(), "sources are never deleted")

    def test_unreadable_source_produces_a_warning(self):
        empty = self.root / "empty.md"
        empty.write_text("   \n", encoding="utf-8")

        state = self.run_pipeline(
            RunOptions(course="Test Course", accept_merges=True), [self.video, empty]
        )

        codes = [warning["code"] for warning in state.warnings]
        self.assertIn("source_text_empty", codes)
        report = (self.home / "jobs" / state.job_id / "report.md").read_text()
        self.assertIn("## Warnings", report)

    def test_pdf_figures_are_embedded_and_published(self):
        slides = self.root / "slides.pdf"
        slides.write_bytes(b"%PDF-1.4")
        figure = self.root / "figure.png"
        figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        image = ExtractedImage(
            id="s01-1-1",
            page=1,
            name="s01-p01-01.png",
            vault_path="Attachments/scriptorium/slides/s01-p01-01.png",
            file=str(figure),
            width=400,
            height=300,
            caption="Figure 1: Heuristic search",
        )
        prompts: dict[str, str] = {}

        def chat(_client, system, user, *, phase="llm", temperature=0.2, max_tokens=8192):
            prompts[phase] = user
            if phase == "extract":
                payload = extract_response()
                payload["units"][0]["images"] = ["s01-p01-01.png"]
                return payload
            if phase == "group":
                return group_response()
            if phase == "outline":
                return outline_response()
            return {
                "body": "## Fresh\n\n![[Attachments/scriptorium/slides/s01-p01-01.png]]\n",
                "flashcards": [],
            }

        config = self.config.with_overrides(image_input="disabled")
        with (
            mock.patch(
                "scriptorium.pipeline.read_source",
                return_value=ReadResult(text="[page 1]\nSlides about heuristic search."),
            ),
            mock.patch(
                "scriptorium.pipeline.extract_pdf_images",
                return_value=ImageHarvest(images=[image]),
            ),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", chat),
        ):
            pipeline = Pipeline(config, self.ui, self.logger)
            state = pipeline.run([slides], RunOptions(course="Test Course", accept_merges=True))

        self.assertTrue(state.completed())
        self.assertIn("s01-p01-01.png", prompts["extract"])
        body = (self.vault_dir / "Test Note.md").read_text(encoding="utf-8")
        self.assertIn("![[Attachments/scriptorium/slides/s01-p01-01.png]]", body)
        self.assertTrue((self.vault_dir / image.vault_path).is_file())
        codes = [warning["code"] for warning in state.warnings]
        self.assertIn("image_input_unsupported", codes)

    def test_dry_run_then_resume(self):
        options = RunOptions(course="Test Course", dry_run=True, accept_merges=True)
        state = self.run_pipeline(options)
        self.assertFalse((self.vault_dir / "Test Note.md").exists())
        self.assertEqual(state.groups[0].status("apply"), "skipped")
        self.assertTrue((self.home / "jobs" / state.job_id / "report.md").is_file())

        with (
            mock.patch("scriptorium.pipeline.to_mp3", self.fake_to_mp3),
            mock.patch("scriptorium.asr.LocalTranscriber.transcribe", self.fake_transcribe),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", self.fake_chat),
        ):
            pipeline = Pipeline(self.config, self.ui, self.logger)
            resumed = pipeline.resume(state.job_id, RunOptions(accept_merges=True))

        self.assertTrue(resumed.completed())
        self.assertFalse(resumed.options.get("dry_run"))
        self.assertTrue((self.vault_dir / "Test Note.md").is_file())

    def failing_chat(self, system, user, *, phase="llm", **kwargs):
        if phase == "outline":
            raise LLMError("model offline")
        return self.fake_chat(system, user, phase=phase, **kwargs)

    def test_a_failed_run_is_recorded_and_resumable(self):
        pipeline = Pipeline(self.config, self.ui, self.logger)
        with (
            mock.patch("scriptorium.pipeline.to_mp3", self.fake_to_mp3),
            mock.patch("scriptorium.asr.LocalTranscriber.transcribe", self.fake_transcribe),
            mock.patch("scriptorium.llm.client.DeepSeekClient.chat_json", self.failing_chat),
            self.assertRaises(LLMError),
        ):
            pipeline.run([self.video], RunOptions(course="Test Course"))

        state = pipeline.store.recent()[0]
        self.assertEqual(state.groups[0].status(Stage.OUTLINE), "failed")
        payload = json.loads(
            (self.home / "history" / f"{state.job_id}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["status"], "failed")
        self.assertIn("outline", payload["error"])
        self.assertEqual(
            [candidate.job_id for candidate in pipeline.store.resumable()], [state.job_id]
        )

    def test_resume_rejects_a_missing_source(self):
        state = self.run_pipeline(RunOptions(course="Test Course", dry_run=True))
        self.video.unlink()

        with self.assertRaises(ConfigurationError):
            Pipeline(self.config, self.ui, self.logger).resume(state.job_id, RunOptions())


if __name__ == "__main__":
    unittest.main()
