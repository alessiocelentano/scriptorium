"""Speech-to-text: a local transformers pipeline or an OpenAI-compatible API."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .audio import probe
from .config import (
    ASR_PROTOCOL_HUGGINGFACE,
    ASR_PROTOCOL_OPENAI,
    DEFAULT_ASR_API_TIMEOUT,
    DEFAULT_ASR_MODEL,
)
from .errors import DependencyError, TranscribeError
from .log import Logger, Timer
from .models import Segment, Transcript

# Some providers sit behind a Cloudflare browser-integrity check that blocks
# urllib's default ``Python-urllib/x.y`` signature with HTTP 403 (error 1010).
USER_AGENT = f"scriptorium/{__version__}"

# OpenAI rejects uploads above 25 MB; keep a margin for multipart overhead.
API_UPLOAD_LIMIT_BYTES = 24 * 1024 * 1024
API_CHUNK_SECONDS = 1200.0

# Hard ceiling on concurrent uploads.  ``asr_api_rpm`` paces the requests; this
# only stops a generous allowance from opening hundreds of sockets at once.
MAX_IN_FLIGHT = 16


@dataclass
class AsrSettings:
    model: str = DEFAULT_ASR_MODEL
    language: str = "en"
    device: str = "auto"
    batch_size: int = 8
    chunk_length_s: int = 30
    stride_length_s: int = 5
    backend: str = "local"
    api_model: str = ""
    api_base_url: str = ""
    api_protocol: str = ASR_PROTOCOL_OPENAI
    api_key: str = ""
    api_timeout: float = DEFAULT_ASR_API_TIMEOUT
    api_max_bytes: int = API_UPLOAD_LIMIT_BYTES
    # Requests per minute toward the API; zero sends every request as soon as a
    # worker is free.
    api_rpm: int = 0


@dataclass
class TranscribeBatch:
    """Transcripts by source id, plus the sources that could not be transcribed."""

    transcripts: dict[str, Transcript] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def ok(self) -> bool:
        return not self.failures


class _RateLimiter:
    """Token bucket pacing the requests sent to the transcription API.

    ``rpm`` tokens refill over a minute, so a burst up to ``rpm`` requests goes
    out immediately and the rest wait one token at a time.  Zero disables the
    pacing: the caller is then bounded only by its worker count.
    """

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = max(0, int(rpm))
        self._clock = clock
        self._sleep = sleep
        self._interval = 60.0 / self.rpm if self.rpm else 0.0
        self._tokens = float(self.rpm)
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Take one request slot, sleeping until a token is available."""
        while True:
            with self._lock:
                if not self.rpm:
                    return
                now = self._clock()
                self._tokens = min(
                    float(self.rpm),
                    self._tokens + max(0.0, now - self._updated) / self._interval,
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) * self._interval
            self._sleep(wait)

    def estimate(self, requests: int) -> float:
        """Seconds a burst of ``requests`` spends waiting, ignoring the network."""
        if not self.rpm or requests <= self.rpm:
            return 0.0
        return (requests - self.rpm) * self._interval


@dataclass(frozen=True)
class _ChunkRequest:
    """One upload: a whole audio file, or one piece of a split one."""

    source_id: str
    name: str
    index: int
    total: int
    path: Path
    duration: float


@dataclass(frozen=True)
class _ChunkResult:
    """The segments returned for one :class:`_ChunkRequest`."""

    request: _ChunkRequest
    segments: list[Segment]
    language: str


#: ``source id, state, chunks done, chunks total``.
ProgressCallback = Callable[..., None]


class LocalTranscriber:
    """Lazily loads a local speech-to-text model, so the CLI works without torch."""

    def __init__(self, settings: AsrSettings, logger: Logger):
        self.settings = settings
        self.logger = logger
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"

    # -- loading ------------------------------------------------------------- #

    def _load(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DependencyError(
                "speech recognition requires torch and transformers. "
                "Install them with: python3 -m pip install -e '.[asr]'"
            ) from exc

        self._torch = torch
        if self.settings.device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = self.settings.device

        dtype = torch.float16 if self._device in {"mps", "cuda"} else torch.float32
        self.logger.info(
            f"loading {self.settings.model} on {self._device} ({str(dtype).split('.')[-1]})"
        )
        try:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.settings.model,
                device=self._device,
                torch_dtype=dtype,
            )
        except TypeError:
            # Older transformers versions do not accept torch_dtype here.
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.settings.model,
                device=self._device,
            )
        except Exception as exc:  # pragma: no cover - model download/runtime
            raise TranscribeError(f"cannot load ASR model {self.settings.model}: {exc}") from exc
        return self._pipe

    # -- transcription ------------------------------------------------------- #

    def transcribe(self, audio: Path, *, prompt: str | None = None) -> Transcript:
        pipe = self._load()
        timer = Timer()
        self.logger.step(f"transcribing {audio.name} on {self._device}")

        kwargs: dict[str, Any] = {
            "chunk_length_s": self.settings.chunk_length_s,
            "stride_length_s": self.settings.stride_length_s,
            "batch_size": self.settings.batch_size,
            "return_timestamps": True,
        }
        generation: dict[str, Any] = {
            "language": self.settings.language or None,
            "task": "transcribe",
        }
        prompt_ids = self._prompt_ids(pipe, prompt)

        # Whisper accepts language/task and prompt ids; other speech-to-text
        # models may reject one or both, so fall back to the plain call instead
        # of failing the transcript.
        attempts: list[dict[str, Any]] = []
        if prompt_ids is not None:
            attempts.append({"prompt_ids": prompt_ids, "generate_kwargs": generation})
        attempts.append({"generate_kwargs": generation})
        attempts.append({})

        result: Any = None
        last_error: Exception | None = None
        for options in attempts:
            try:
                result = pipe(str(audio), **options, **kwargs)
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
                self.logger.warn(f"retrying without unsupported ASR options: {exc}")
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                raise TranscribeError(f"transcription failed: {exc}") from exc
        if result is None:
            raise TranscribeError(f"transcription failed: {last_error}")

        transcript = self._to_transcript(result)
        self.logger.success(
            f"transcribed {len(transcript.segments)} segments in {timer.human()} "
            f"({len(transcript.text.split())} words)"
        )
        self.logger.event(
            "asr",
            model=self.settings.model,
            device=self._device,
            segments=len(transcript.segments),
            words=len(transcript.text.split()),
            seconds=round(timer.elapsed, 1),
        )
        return transcript

    def transcribe_many(
        self,
        audios: Sequence[tuple[str, Path]],
        *,
        prompt: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TranscribeBatch:
        """Transcribe the sources one after the other on the local model.

        A local pipeline keeps the whole model in one process and its own batch
        already fills the device, so extra workers would only fight for the
        same GPU or CPU.  Only the API backend overlaps its requests.
        """
        batch = TranscribeBatch()
        for source_id, audio in audios:
            if on_progress is not None:
                on_progress(source_id, state="running", done=0, total=1)
            try:
                batch.transcripts[source_id] = self.transcribe(audio, prompt=prompt)
            except Exception as exc:
                batch.failures[source_id] = str(exc)
                if on_progress is not None:
                    on_progress(source_id, state="failed", done=0, total=1)
                continue
            if on_progress is not None:
                on_progress(source_id, state="done", done=1, total=1)
        return batch

    def _prompt_ids(self, pipe: Any, prompt: str | None) -> Any | None:
        if not prompt:
            return None
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "get_prompt_ids"):
            return None
        try:
            ids = tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        except Exception:
            return None
        if self._torch is not None and self._device != "cpu":
            try:
                ids = ids.to(self._device)
            except Exception:
                pass
        return ids

    def _to_transcript(self, result: Any) -> Transcript:
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            raise TranscribeError("unexpected ASR output shape")

        raw_chunks = result.get("chunks") or []
        segments: list[Segment] = []
        if raw_chunks:
            previous_end = 0.0
            for chunk in raw_chunks:
                text = str(chunk.get("text") or "").strip()
                if not text:
                    continue
                stamp = chunk.get("timestamp") or (None, None)
                start = _as_float(stamp[0], previous_end)
                end = _as_float(stamp[1], start)
                segments.append(Segment(start=start, end=max(start, end), text=text))
                previous_end = max(start, end)
        else:
            text = str(result.get("text") or "").strip()
            if text:
                segments.append(Segment(start=0.0, end=0.0, text=text))

        if not segments:
            raise TranscribeError("the ASR model returned an empty transcript")

        duration = segments[-1].end
        return Transcript(
            language=self.settings.language,
            duration=duration,
            segments=segments,
        )


class ApiTranscriber:
    """Transcribe through an OpenAI-compatible ``/audio/transcriptions`` API.

    Long audio is split into overlapping-free MP3 chunks with ffmpeg so each
    upload stays below the provider limit; chunk timestamps are shifted back to
    the original timeline.  Providers that support ``verbose_json`` return
    segments, the others fall back to a single whole-file segment.
    """

    def __init__(self, settings: AsrSettings, logger: Logger):
        self.settings = settings
        self.logger = logger

    def transcribe(self, audio: Path, *, prompt: str | None = None) -> Transcript:
        """Transcribe one audio file; :meth:`transcribe_many` handles several."""
        batch = self.transcribe_many([("", audio)], prompt=prompt)
        if "" in batch.transcripts:
            return batch.transcripts[""]
        raise TranscribeError(batch.failures.get("", "transcription failed"))

    def transcribe_many(
        self,
        audios: Sequence[tuple[str, Path]],
        *,
        prompt: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TranscribeBatch:
        """Transcribe several audios with one shared request budget.

        Every upload enters a single queue: the pieces of one source and the
        pieces of the next are sent together and reassembled by source id
        afterwards, so a slow episode never blocks the one after it.  The
        ``api_rpm`` token bucket paces the queue as a whole; ``MAX_IN_FLIGHT``
        caps the sockets.  A failed source keeps its name in ``failures`` and
        never stops the others.
        """
        if not audios:
            return TranscribeBatch()
        if not self.settings.api_key:
            raise TranscribeError(
                "api transcription needs SCRIPTORIUM_ASR_API_KEY or OPENAI_API_KEY"
            )
        timer = Timer()
        limiter = _RateLimiter(self.settings.api_rpm)
        self.logger.step(f"transcribing {len(audios)} source(s) via {self.settings.api_base_url}")
        with tempfile.TemporaryDirectory(prefix="scriptorium-asr-") as temporary:
            requests, batch = self._plan(audios, Path(temporary))
            self._log_plan(len(requests), len(audios), limiter)
            results = self._dispatch(requests, prompt, limiter, on_progress, batch)
        self._assemble(audios, results, batch)

        if batch.transcripts:
            segments = sum(len(item.segments) for item in batch.transcripts.values())
            words = sum(len(item.text.split()) for item in batch.transcripts.values())
            self.logger.success(
                f"transcribed {len(batch.transcripts)} source(s) in {timer.human()} "
                f"({segments} segments, {words} words)"
            )
        self.logger.event(
            "asr",
            backend="api",
            model=self.settings.api_model,
            sources=len(batch.transcripts),
            failed=len(batch.failures),
            requests=len(requests),
            rpm=self.settings.api_rpm,
            words=sum(len(item.text.split()) for item in batch.transcripts.values()),
            seconds=round(timer.elapsed, 1),
        )
        return batch

    # -- planning ------------------------------------------------------------ #

    def _plan(
        self, audios: Sequence[tuple[str, Path]], root: Path
    ) -> tuple[list[_ChunkRequest], TranscribeBatch]:
        """Split every audio that exceeds the upload limit and list the uploads."""
        requests: list[_ChunkRequest] = []
        batch = TranscribeBatch()
        for position, (source_id, audio) in enumerate(audios):
            directory = root / f"{position:03d}"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                chunks = self._chunks(audio, directory)
            except (TranscribeError, OSError) as exc:
                batch.failures[source_id] = str(exc)
                self.logger.error(f"{audio.name}: {exc}")
                continue
            for index, chunk in enumerate(chunks, start=1):
                requests.append(
                    _ChunkRequest(
                        source_id=source_id,
                        name=audio.name,
                        index=index,
                        total=len(chunks),
                        path=chunk,
                        duration=_media_duration(chunk),
                    )
                )
        return requests, batch

    def _log_plan(self, requests: int, sources: int, limiter: _RateLimiter) -> None:
        detail = f"{sources} source(s), {requests} request(s)"
        if self.settings.api_rpm:
            detail += f", {self.settings.api_rpm} rpm"
        waiting = limiter.estimate(requests)
        if waiting >= 1:
            detail += f", about {int(waiting)}s of rate-limit waiting"
        self.logger.info(detail)

    # -- dispatch ------------------------------------------------------------ #

    def _dispatch(
        self,
        requests: Sequence[_ChunkRequest],
        prompt: str | None,
        limiter: _RateLimiter,
        on_progress: ProgressCallback | None,
        batch: TranscribeBatch,
    ) -> list[_ChunkResult]:
        """Send every upload, paced by ``limiter``, and collect the answers."""
        if not requests:
            return []
        results: list[_ChunkResult] = []
        done: dict[str, int] = {}
        workers = max(1, min(len(requests), MAX_IN_FLIGHT))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scriptorium-asr") as pool:
            futures = {
                pool.submit(self._upload, request, prompt, limiter): request for request in requests
            }
            for future in as_completed(futures):
                request = futures[future]
                try:
                    segments, language = future.result()
                except Exception as exc:
                    # One upload can fail without losing the other sources.
                    batch.failures.setdefault(request.source_id, str(exc))
                    self.logger.error(f"{request.name}: {exc}")
                    if on_progress is not None:
                        on_progress(
                            request.source_id,
                            state="failed",
                            done=done.get(request.source_id, 0),
                            total=request.total,
                        )
                    continue
                results.append(_ChunkResult(request=request, segments=segments, language=language))
                done[request.source_id] = done.get(request.source_id, 0) + 1
                if on_progress is not None:
                    finished = done[request.source_id] == request.total
                    on_progress(
                        request.source_id,
                        state="done" if finished else "running",
                        done=done[request.source_id],
                        total=request.total,
                    )
        return results

    def _upload(
        self, request: _ChunkRequest, prompt: str | None, limiter: _RateLimiter
    ) -> tuple[list[Segment], str]:
        """Wait for a request slot, upload one chunk, return its segments."""
        limiter.acquire()
        if request.total > 1:
            self.logger.info(f"{request.name}: chunk {request.index}/{request.total}")
        payload = self._request(request.path, prompt)
        return _segments_from_payload(payload), str(payload.get("language") or "")

    # -- reassembly ---------------------------------------------------------- #

    def _assemble(
        self,
        audios: Sequence[tuple[str, Path]],
        results: Sequence[_ChunkResult],
        batch: TranscribeBatch,
    ) -> None:
        """Rebuild one transcript per source, in chunk order, with shifted times."""
        grouped: dict[str, list[_ChunkResult]] = {}
        for result in results:
            grouped.setdefault(result.request.source_id, []).append(result)
        for source_id, _audio in audios:
            if source_id in batch.failures:
                continue
            chunks = sorted(grouped.get(source_id, []), key=lambda item: item.request.index)
            if not chunks:
                continue
            language = next(
                (item.language for item in chunks if item.language), self.settings.language
            )
            try:
                batch.transcripts[source_id] = _join_chunks(chunks, language)
            except TranscribeError as exc:
                batch.failures[source_id] = str(exc)

    # -- request ------------------------------------------------------------- #

    def _request(self, audio: Path, prompt: str | None) -> dict[str, Any]:
        if self.settings.api_protocol == ASR_PROTOCOL_HUGGINGFACE:
            return self._request_huggingface(audio)
        return self._request_openai(audio, prompt)

    def _request_huggingface(self, audio: Path) -> dict[str, Any]:
        """Hugging Face router and Inference Endpoints take a raw audio body."""
        try:
            data = audio.read_bytes()
        except OSError as exc:
            raise TranscribeError(f"cannot read audio {audio}: {exc}") from exc
        url = self.settings.api_base_url
        if self.settings.api_model:
            url = f"{url}/{self.settings.api_model}"
        last_error = ""
        for attempt in range(4):
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "audio/mpeg",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.api_timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return _json_payload(raw)
            except urllib.error.HTTPError as exc:
                detail = _short(exc.read().decode("utf-8", errors="replace"))
                last_error = f"HTTP {exc.code}: {detail}"
                # 503 is the usual "model is loading" answer of the router.
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < 3:
                    delay = 2**attempt
                    self.logger.warn(
                        f"Hugging Face transcription error ({exc.code}), retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise TranscribeError(f"Hugging Face transcription failed: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
                if attempt < 3:
                    delay = 2**attempt
                    self.logger.warn(
                        f"Hugging Face transcription request failed ({last_error}), "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise TranscribeError(
                    f"Hugging Face transcription request failed: {last_error}"
                ) from exc
        raise TranscribeError(f"Hugging Face transcription failed: {last_error}")

    def _request_openai(self, audio: Path, prompt: str | None) -> dict[str, Any]:
        verbose = True
        last_error = ""
        for attempt in range(4):
            body, content_type = _multipart_body(
                self._fields(prompt, verbose=verbose), "file", audio
            )
            request = urllib.request.Request(
                f"{self.settings.api_base_url}/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": content_type,
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.api_timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return _json_payload(raw)
            except urllib.error.HTTPError as exc:
                detail = _short(exc.read().decode("utf-8", errors="replace"))
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and verbose:
                    self.logger.info("provider rejected verbose_json; retrying without timestamps")
                    verbose = False
                    continue
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < 3:
                    delay = 2**attempt
                    self.logger.warn(f"transcription API error ({exc.code}), retrying in {delay}s")
                    time.sleep(delay)
                    continue
                raise TranscribeError(f"transcription API failed: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
                if attempt < 3:
                    delay = 2**attempt
                    self.logger.warn(
                        f"transcription request failed ({last_error}), retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise TranscribeError(f"transcription request failed: {last_error}") from exc
        raise TranscribeError(f"transcription API failed: {last_error}")

    def _fields(self, prompt: str | None, *, verbose: bool) -> dict[str, str]:
        fields = {
            "model": self.settings.api_model,
            "response_format": "verbose_json" if verbose else "json",
        }
        if verbose:
            fields["timestamp_granularities[]"] = "segment"
        if self.settings.language:
            fields["language"] = self.settings.language
        if prompt:
            fields["prompt"] = prompt
        return fields

    # -- chunking ------------------------------------------------------------ #

    def _chunks(self, audio: Path, directory: Path) -> list[Path]:
        size = audio.stat().st_size if audio.is_file() else 0
        limit = self.settings.api_max_bytes
        if limit <= 0 or size <= limit:
            return [audio]

        if shutil.which("ffmpeg") is None:
            raise TranscribeError(
                f"{audio.name} is {size // (1024 * 1024)} MB, above the API upload "
                "limit, and ffmpeg is not available to split it. Install ffmpeg or "
                "clear asr_api_base_url to run the local speech-to-text model"
            )

        duration = _media_duration(audio)
        seconds = API_CHUNK_SECONDS
        if duration > 0:
            seconds = max(60.0, duration * (limit / size) * 0.9)
        pattern = directory / "chunk-%03d.mp3"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio),
            "-f",
            "segment",
            "-segment_time",
            f"{seconds:.0f}",
            "-c",
            "copy",
            str(pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        chunks = sorted(directory.glob("chunk-*.mp3"))
        if result.returncode != 0 or not chunks:
            detail = " ".join(result.stderr.split())[-300:]
            raise TranscribeError(f"cannot split {audio.name} for the API: {detail}")
        return chunks


def _media_duration(path: Path) -> float:
    try:
        return max(0.0, probe(path).duration)
    except Exception:
        return 0.0


def _json_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranscribeError(f"the ASR API returned invalid JSON: {_short(raw)}") from exc
    if not isinstance(payload, dict):
        raise TranscribeError("unexpected ASR API response shape")
    return payload


def _segments_from_payload(payload: dict[str, Any]) -> list[Segment]:
    segments: list[Segment] = []
    raw = payload.get("segments")
    if not isinstance(raw, list):
        # Hugging Face calls its timed chunks "chunks" with [start, end].
        raw = payload.get("chunks")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            stamp = item.get("timestamp")
            if isinstance(stamp, list) and len(stamp) >= 2:
                start = _as_float(stamp[0], 0.0)
                end = _as_float(stamp[1], start)
            else:
                start = _as_float(item.get("start"), 0.0)
                end = _as_float(item.get("end"), start)
            segments.append(Segment(start=start, end=max(start, end), text=text))
    if not segments:
        text = str(payload.get("text") or "").strip()
        if text:
            duration = _as_float(payload.get("duration"), 0.0)
            segments.append(Segment(start=0.0, end=duration, text=text))
    return segments


def _join_chunks(chunks: Sequence[_ChunkResult], language: str) -> Transcript:
    """Stitch the chunks of one source back into a single timeline.

    Chunks are joined in index order, not in the order their answers arrived,
    so the transcript is identical to a sequential run.  A chunk whose probe
    returned no duration advances the timeline by its last segment end, the
    same fallback the sequential implementation used.
    """
    segments: list[Segment] = []
    offset = 0.0
    duration = 0.0
    for chunk in chunks:
        for segment in chunk.segments:
            segments.append(
                Segment(
                    start=segment.start + offset,
                    end=segment.end + offset,
                    text=segment.text,
                )
            )
        offset += chunk.request.duration or (chunk.segments[-1].end if chunk.segments else 0.0)
        duration = max(duration, offset)
    if not segments:
        raise TranscribeError("the ASR API returned an empty transcript")
    return Transcript(language=language, duration=duration, segments=segments)


def _multipart_body(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = "----scriptorium" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TranscribeError(f"cannot read audio {path}: {exc}") from exc
    mime = "audio/mpeg" if path.suffix.lower() == ".mp3" else "application/octet-stream"
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _short(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


_WIKILINK_TITLE_RE = re.compile(r"\[\[([^\]|#]+)")


def build_bias_prompt(
    course: str,
    moc_text: str = "",
    glossary_text: str = "",
    extra_terms: list[str] | None = None,
    max_chars: int = 900,
) -> str:
    """Build a short prompt with course vocabulary for prompt biasing."""
    terms: list[str] = []
    for match in _WIKILINK_TITLE_RE.findall(moc_text):
        title = match.strip()
        if title and not title.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            terms.append(title)
    terms.extend(extra_terms or [])

    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(term)

    parts = [f"The lecture belongs to the course {course}."]
    if unique:
        parts.append("Key terms: " + ", ".join(unique[:40]) + ".")
    glossary = " ".join(glossary_text.split())
    if glossary:
        parts.append("Vocabulary: " + glossary[:400])

    prompt = " ".join(parts)
    return prompt[:max_chars].strip()
