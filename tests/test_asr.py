import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.asr import (
    USER_AGENT,
    ApiTranscriber,
    AsrSettings,
    LocalTranscriber,
    _multipart_body,
    _segments_from_payload,
)
from scriptorium.audio import MediaInfo
from scriptorium.errors import TranscribeError
from scriptorium.log import Logger


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
