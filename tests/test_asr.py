import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.asr import (
    MAX_IN_FLIGHT,
    USER_AGENT,
    ApiTranscriber,
    AsrSettings,
    LocalTranscriber,
    _multipart_body,
    _RateLimiter,
    _segments_from_payload,
)
from scriptorium.audio import MediaInfo
from scriptorium.errors import TranscribeError
from scriptorium.log import Logger
from scriptorium.models import Segment, Transcript


class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


class ApiTranscriberTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.audio = Path(self._tmp.name) / "lecture.mp3"
        self.audio.write_bytes(b"fake mp3 bytes")
        self.logger = Logger()

    def tearDown(self):
        self._tmp.cleanup()

    def settings(self, **overrides: object) -> AsrSettings:
        values: dict[str, object] = {
            "backend": "api",
            "api_model": "whisper-1",
            "api_base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
            "language": "en",
        }
        values.update(overrides)
        return AsrSettings(**values)  # type: ignore[arg-type]

    def test_verbose_response_becomes_timestamped_segments(self):
        payload = {
            "language": "english",
            "duration": 12.0,
            "segments": [
                {"start": 0.0, "end": 5.5, "text": " First part."},
                {"start": 5.5, "end": 12.0, "text": "Second part."},
            ],
        }
        with (
            mock.patch("scriptorium.asr.probe", return_value=MediaInfo(12.0, True, False)),
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)),
        ):
            transcript = ApiTranscriber(self.settings(), self.logger).transcribe(self.audio)

        self.assertEqual([(s.start, s.end) for s in transcript.segments], [(0.0, 5.5), (5.5, 12.0)])
        self.assertEqual(transcript.text, "First part.\nSecond part.")
        self.assertEqual(transcript.duration, 12.0)

    def test_plain_json_response_falls_back_to_one_segment(self):
        payload = {"text": "Whole lecture.", "duration": 30.0}
        with (
            mock.patch("scriptorium.asr.probe", return_value=MediaInfo(30.0, True, False)),
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)),
        ):
            transcript = ApiTranscriber(self.settings(), self.logger).transcribe(self.audio)

        self.assertEqual(len(transcript.segments), 1)
        self.assertEqual(transcript.segments[0].text, "Whole lecture.")
        self.assertEqual(transcript.segments[0].end, 30.0)

    def test_openai_request_sends_an_explicit_user_agent(self):
        # urllib's default signature is blocked by Cloudflare (error 1010).
        payload = {"text": "Lecture."}
        with (
            mock.patch("scriptorium.asr.probe", return_value=MediaInfo(10.0, True, False)),
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as opened,
        ):
            ApiTranscriber(self.settings(), self.logger).transcribe(self.audio)

        request = opened.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertNotIn("Python-urllib", request.get_header("User-agent"))

    def test_huggingface_request_sends_an_explicit_user_agent(self):
        payload = {"text": "Hugging Face lecture."}
        settings = self.settings(api_protocol="huggingface")
        with (
            mock.patch("scriptorium.asr.probe", return_value=MediaInfo(10.0, True, False)),
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as opened,
        ):
            ApiTranscriber(settings, self.logger).transcribe(self.audio)

        request = opened.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)

    def test_huggingface_protocol_posts_raw_audio_to_the_model_route(self):
        payload = {"text": "Hugging Face lecture."}
        settings = self.settings(
            api_protocol="huggingface",
            api_base_url="https://router.huggingface.co/hf-inference/models",
            api_model="openai/whisper-large-v3",
        )
        with (
            mock.patch("scriptorium.asr.probe", return_value=MediaInfo(10.0, True, False)),
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as opened,
        ):
            transcript = ApiTranscriber(settings, self.logger).transcribe(self.audio)

        request = opened.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3",
        )
        self.assertEqual(request.data, b"fake mp3 bytes")
        self.assertEqual(request.get_header("Content-type"), "audio/mpeg")
        self.assertEqual(transcript.segments[0].text, "Hugging Face lecture.")

    def test_huggingface_chunks_become_timestamped_segments(self):
        payload = {
            "text": "First. Second.",
            "chunks": [
                {"text": " First.", "timestamp": [0.0, 2.0]},
                {"text": " Second.", "timestamp": [2.0, 5.0]},
            ],
        }
        with (
            mock.patch("scriptorium.asr.probe", return_value=MediaInfo(5.0, True, False)),
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)),
        ):
            transcript = ApiTranscriber(self.settings(), self.logger).transcribe(self.audio)

        self.assertEqual([(s.start, s.end) for s in transcript.segments], [(0.0, 2.0), (2.0, 5.0)])

    def test_missing_api_key_is_rejected_before_any_request(self):
        transcriber = ApiTranscriber(self.settings(api_key=""), self.logger)
        with self.assertRaises(TranscribeError):
            transcriber.transcribe(self.audio)

    def test_upload_above_limit_is_split_into_chunks(self):
        # A file larger than the configured limit forces the ffmpeg path.
        self.audio.write_bytes(b"x" * 2048)
        transcriber = ApiTranscriber(self.settings(api_max_bytes=1024), self.logger)
        chunks = [Path(self._tmp.name) / "chunk-000.mp3"]
        with (
            mock.patch("scriptorium.asr.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch("scriptorium.asr._media_duration", return_value=60.0),
            mock.patch("scriptorium.asr.subprocess.run") as run,
        ):
            run.return_value = mock.Mock(returncode=0, stderr="")
            with mock.patch("scriptorium.asr.Path.glob", return_value=chunks):
                result = transcriber._chunks(self.audio, Path(self._tmp.name))

        self.assertEqual(result, chunks)
        command = run.call_args.args[0]
        self.assertIn("-f", command)
        self.assertIn("segment", command)


class FakeClock:
    """A clock whose ``sleep`` advances time instead of waiting."""

    def __init__(self):
        self.now = 0.0
        self.slept = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds


class RateLimiterTests(unittest.TestCase):
    def limiter(self, rpm: int) -> tuple[_RateLimiter, FakeClock]:
        clock = FakeClock()
        return _RateLimiter(rpm, clock=clock.clock, sleep=clock.sleep), clock

    def test_a_burst_up_to_the_allowance_leaves_at_once(self):
        limiter, clock = self.limiter(200)
        for _ in range(200):
            limiter.acquire()
        self.assertEqual(clock.slept, 0.0)

    def test_requests_over_the_allowance_wait_one_token_at_a_time(self):
        limiter, clock = self.limiter(60)  # one token per second
        for _ in range(62):
            limiter.acquire()
        self.assertAlmostEqual(clock.slept, 2.0, places=6)

    def test_zero_rpm_never_waits(self):
        limiter, clock = self.limiter(0)
        for _ in range(100):
            limiter.acquire()
        self.assertEqual(clock.slept, 0.0)

    def test_estimate_reports_only_the_waiting_part(self):
        limiter, _clock = self.limiter(60)
        self.assertEqual(limiter.estimate(30), 0.0)
        self.assertAlmostEqual(limiter.estimate(63), 3.0, places=6)


class ApiBatchTests(unittest.TestCase):
    """One queue for every source and every chunk of a split recording."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.logger = Logger()

    def tearDown(self):
        self._tmp.cleanup()

    def source(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"fake mp3 bytes")
        return path

    def settings(self, **overrides: object) -> AsrSettings:
        values: dict[str, object] = {
            "backend": "api",
            "api_model": "whisper-1",
            "api_base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
            "language": "en",
        }
        values.update(overrides)
        return AsrSettings(**values)  # type: ignore[arg-type]

    @staticmethod
    def three_chunks(_transcriber, audio: Path, _directory: Path) -> list[Path]:
        """Pretend every source is large enough to be split in three."""
        return [audio.with_name(f"{audio.stem}-{index}.mp3") for index in range(1, 4)]

    @staticmethod
    def answer(audio: Path) -> dict:
        """A payload naming the chunk it answers and lasting ten seconds."""
        stem = audio.stem
        return {
            "language": "en",
            "segments": [{"start": 0.0, "end": 4.0, "text": stem.replace("-", " ")}],
        }

    def batch(self, audios, request, **overrides):
        with (
            mock.patch.object(ApiTranscriber, "_chunks", ApiBatchTests.three_chunks),
            mock.patch("scriptorium.asr._media_duration", return_value=10.0),
            mock.patch.object(ApiTranscriber, "_request", side_effect=request),
        ):
            return ApiTranscriber(self.settings(**overrides), self.logger).transcribe_many(audios)

    def test_chunks_arriving_out_of_order_are_reassembled_per_source(self):
        first, second = self.source("a.mp3"), self.source("b.mp3")

        def request(audio: Path, _prompt: str | None = None) -> dict:
            # The last chunk of each source answers first, so the completion
            # order never matches the source order.
            index = int(audio.stem.rsplit("-", 1)[1])
            time.sleep(0.01 * (4 - index))
            return self.answer(audio)

        batch = self.batch([("s01", first), ("s02", second)], request)

        self.assertTrue(batch.ok())
        self.assertEqual(list(batch.transcripts), ["s01", "s02"])
        a = batch.transcripts["s01"]
        self.assertEqual(a.text, "a 1\na 2\na 3")
        self.assertEqual([segment.start for segment in a.segments], [0.0, 10.0, 20.0])
        self.assertEqual([segment.end for segment in a.segments], [4.0, 14.0, 24.0])
        self.assertEqual(a.duration, 30.0)
        self.assertEqual(batch.transcripts["s02"].text, "b 1\nb 2\nb 3")

    def test_one_failed_chunk_loses_only_its_source(self):
        first, second = self.source("a.mp3"), self.source("b.mp3")

        def request(audio: Path, _prompt: str | None = None) -> dict:
            if audio.stem == "b-2":
                raise TranscribeError("HTTP 500: upstream is down")
            return self.answer(audio)

        batch = self.batch([("s01", first), ("s02", second)], request)

        self.assertFalse(batch.ok())
        self.assertEqual(list(batch.failures), ["s02"])
        self.assertIn("HTTP 500", batch.failures["s02"])
        self.assertEqual(list(batch.transcripts), ["s01"])

    def test_the_pool_never_opens_more_uploads_than_the_cap(self):
        audio = self.source("long.mp3")
        lock = threading.Lock()
        live = 0
        peak = 0

        def request(_audio: Path, _prompt: str | None = None) -> dict:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.005)
            with lock:
                live -= 1
            return {"text": "words", "duration": 1.0}

        def many_chunks(_transcriber, _audio: Path, _directory: Path) -> list[Path]:
            return [audio.with_name(f"long-{index}.mp3") for index in range(1, 41)]

        with (
            mock.patch.object(ApiTranscriber, "_chunks", many_chunks),
            mock.patch("scriptorium.asr._media_duration", return_value=1.0),
            mock.patch.object(ApiTranscriber, "_request", side_effect=request),
        ):
            batch = ApiTranscriber(self.settings(), self.logger).transcribe_many([("s01", audio)])

        self.assertTrue(batch.ok())
        self.assertLessEqual(peak, MAX_IN_FLIGHT)
        self.assertGreater(peak, 1, "the uploads must actually overlap")

    def test_progress_reports_every_source_once(self):
        first, second = self.source("a.mp3"), self.source("b.mp3")
        events: list[tuple[str, str, int, int]] = []

        def on_progress(source_id: str, *, state: str, done: int, total: int) -> None:
            events.append((source_id, state, done, total))

        with (
            mock.patch.object(ApiTranscriber, "_chunks", ApiBatchTests.three_chunks),
            mock.patch("scriptorium.asr._media_duration", return_value=10.0),
            mock.patch.object(
                ApiTranscriber, "_request", side_effect=lambda audio, _prompt: self.answer(audio)
            ),
        ):
            ApiTranscriber(self.settings(), self.logger).transcribe_many(
                [("s01", first), ("s02", second)], on_progress=on_progress
            )

        final = [event for event in events if event[1] in {"done", "failed"}]
        self.assertEqual(sorted(event[0] for event in final), ["s01", "s02"])
        for _source_id, state, done, total in final:
            self.assertEqual(state, "done")
            self.assertEqual((done, total), (3, 3))

    def test_an_audio_that_cannot_be_split_loses_only_itself(self):
        first, second = self.source("a.mp3"), self.source("b.mp3")

        def chunks(_transcriber, audio: Path, directory: Path) -> list[Path]:
            if audio.stem == "a":
                raise TranscribeError("cannot split a.mp3 for the API: no ffmpeg")
            return ApiBatchTests.three_chunks(None, audio, directory)

        with (
            mock.patch.object(ApiTranscriber, "_chunks", chunks),
            mock.patch("scriptorium.asr._media_duration", return_value=10.0),
            mock.patch.object(
                ApiTranscriber, "_request", side_effect=lambda audio, _prompt: self.answer(audio)
            ),
        ):
            batch = ApiTranscriber(self.settings(), self.logger).transcribe_many(
                [("s01", first), ("s02", second)]
            )

        self.assertEqual(list(batch.failures), ["s01"])
        self.assertEqual(list(batch.transcripts), ["s02"])


class LocalBatchTests(unittest.TestCase):
    def test_local_batches_stay_sequential(self):
        live = 0
        peak = 0
        order: list[str] = []
        lock = threading.Lock()

        def transcribe(_self, audio: Path, prompt: str | None = None) -> Transcript:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
                order.append(audio.name)
            with lock:
                live -= 1
            return Transcript(
                language="en",
                duration=1.0,
                segments=[Segment(start=0.0, end=1.0, text=audio.name)],
            )

        with mock.patch.object(LocalTranscriber, "transcribe", transcribe):
            batch = LocalTranscriber(AsrSettings(), Logger()).transcribe_many(
                [("s01", Path("a.mp3")), ("s02", Path("b.mp3"))]
            )

        self.assertTrue(batch.ok())
        self.assertEqual(peak, 1, "the local model must not run two sources at once")
        self.assertEqual(order, ["a.mp3", "b.mp3"])
        self.assertEqual(list(batch.transcripts), ["s01", "s02"])


class LocalTranscriberTests(unittest.TestCase):
    def test_non_whisper_model_retries_without_whisper_only_options(self):
        class FakePipe:
            def __call__(self, _audio, **kwargs):
                if "generate_kwargs" in kwargs:
                    raise ValueError("language is not a valid generation option")
                return {"chunks": [{"text": "Hello.", "timestamp": (0.0, 1.0)}]}

        settings = AsrSettings(model="facebook/wav2vec2-base-960h", language="en")
        transcriber = LocalTranscriber(settings, Logger(verbose=False))
        transcriber._pipe = FakePipe()
        transcript = transcriber.transcribe(Path("lecture.mp3"))
        self.assertEqual(transcript.text, "Hello.")


class PayloadParsingTests(unittest.TestCase):
    def test_empty_segment_text_is_dropped(self):
        segments = _segments_from_payload(
            {
                "segments": [
                    {"start": 0, "end": 1, "text": " "},
                    {"start": 1, "end": 2, "text": "ok"},
                ]
            }
        )
        self.assertEqual([s.text for s in segments], ["ok"])

    def test_missing_segments_uses_duration(self):
        segments = _segments_from_payload({"text": "hello", "duration": 9.0})
        self.assertEqual(segments[0].end, 9.0)


class MultipartTests(unittest.TestCase):
    def test_body_contains_fields_and_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.mp3"
            audio.write_bytes(b"abc")
            body, content_type = _multipart_body({"model": "whisper-1"}, "file", audio)

        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="model"', body)
        self.assertIn(b"whisper-1", body)
        self.assertIn(b'name="file"; filename="a.mp3"', body)
        self.assertIn(b"Content-Type: audio/mpeg", body)
        self.assertIn(b"abc", body)


if __name__ == "__main__":
    unittest.main()
