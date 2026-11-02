"""DeepSeek chat-completions client with JSON mode and retries."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, llm_api_key
from ..errors import LLMError
from ..log import Logger

# 1x1 transparent PNG, enough to prove the provider accepts image content.
PROBE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    per_phase: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int, phase: str) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        bucket = self.per_phase.setdefault(phase, {"prompt": 0, "completion": 0, "calls": 0})
        bucket["prompt"] += prompt
        bucket["completion"] += completion
        bucket["calls"] += 1


class DeepSeekClient:
    def __init__(
        self,
        config: Config,
        logger: Logger,
        *,
        api_key: str | None = None,
        max_retries: int = 4,
    ):
        self.config = config
        self.logger = logger
        self.api_key = api_key if api_key is not None else llm_api_key(config)
        self.max_retries = max_retries
        self.usage = Usage()

    # -- public -------------------------------------------------------------- #

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        phase: str = "llm",
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.config.llm_json_mode == "object":
            payload["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        response = self._post(payload, phase=phase)
        content = self._content(response)
        usage = response.get("usage") or {}
        self.usage.add(
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            phase,
        )
        self.logger.event(
            "llm_call",
            phase=phase,
            model=self.config.llm_model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            seconds=round(time.monotonic() - started, 1),
        )
        try:
            return parse_json_object(content)
        except LLMError as exc:
            raise LLMError(f"{phase}: {exc}") from exc

    def chat_image(
        self,
        prompt: str,
        image: Path,
        *,
        phase: str = "image",
        max_tokens: int = 2048,
    ) -> str:
        """Return the plain-text answer for one image; JSON mode is off."""
        try:
            data = base64.b64encode(image.read_bytes()).decode("ascii")
        except OSError as exc:
            raise LLMError(f"{phase}: cannot read image {image}: {exc}") from exc
        mime = "image/jpeg" if image.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        started = time.monotonic()
        response = self._post(payload, phase=phase)
        content = self._content(response)
        usage = response.get("usage") or {}
        self.usage.add(
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            phase,
        )
        self.logger.event(
            "llm_call",
            phase=phase,
            model=self.config.llm_model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            seconds=round(time.monotonic() - started, 1),
        )
        return content

    # -- model capabilities -------------------------------------------------- #

    def image_input_supported(self) -> bool | None:
        """Whether the configured model accepts image input.

        ``image_input = "enabled"``/``"disabled"`` in the settings answer
        directly.  ``"auto"`` probes once per model and base URL and caches the
        answer under ``$SCRIPTORIUM_ARTIFACTS_DIR/cache``.  ``None`` means the probe could
        not decide, which callers turn into a warning instead of an error.
        """
        mode = self.config.image_input
        if mode == "enabled":
            return True
        if mode == "disabled":
            return False
        cache = self._load_capabilities()
        key = self._capability_key()
        if key in cache:
            return bool(cache[key])
        supported = self._probe_image_support()
        if supported is not None:
            cache[key] = supported
            self._save_capabilities(cache)
        return supported

    def _capability_key(self) -> str:
        return f"{self.config.llm_base_url}|{self.config.llm_model}"

    def _capability_path(self) -> Path:
        return Path(self.config.artifacts_dir) / "cache" / "capabilities.json"

    def _load_capabilities(self) -> dict[str, object]:
        path = self._capability_path()
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_capabilities(self, payload: dict[str, object]) -> None:
        path = self._capability_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".tmp-{int(time.time() * 1000)}")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            self.logger.debug(f"could not cache model capabilities: {exc}")

    def _probe_image_support(self) -> bool | None:
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                'Reply with the JSON object {"ok": true} if you can read the image.'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{PROBE_PNG_BASE64}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 16,
            "stream": False,
        }
        url = f"{self.config.llm_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=min(30.0, self.config.llm_timeout)
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
            json.loads(body)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 404, 415, 422}:
                self.logger.debug(f"image probe rejected with HTTP {exc.code}")
                return False
            self.logger.warn(f"image probe inconclusive (HTTP {exc.code})")
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warn(f"image probe failed: {exc}")
            return None

    # -- internals ----------------------------------------------------------- #

    def _post(self, payload: dict[str, Any], *, phase: str) -> dict[str, Any]:
        url = f"{self.config.llm_base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: str = ""
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.llm_timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                return json.loads(body)
            except urllib.error.HTTPError as exc:
                detail = _short(exc.read().decode("utf-8", errors="replace"))
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    delay = 2**attempt
                    self.logger.warn(
                        f"{phase}: transient API error ({exc.code}), retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"{phase}: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt < self.max_retries - 1:
                    delay = 2**attempt
                    self.logger.warn(
                        f"{phase}: API request failed ({last_error}), retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"{phase}: request failed: {last_error}") from exc
        raise LLMError(f"{phase}: request failed: {last_error}")

    @staticmethod
    def _content(response: dict[str, Any]) -> str:
        try:
            choices = response["choices"]
            message = choices[0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected API response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("the API returned an empty message")
        return content


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating fences and prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(_balanced_object(text))

    if not isinstance(value, dict):
        raise LLMError("expected a JSON object")
    return value


def _balanced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise LLMError(f"no JSON object in model output: {_short(text)}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise LLMError(f"unterminated JSON object in model output: {_short(text)}")


def _short(text: str, limit: int = 400) -> str:
    text = " ".join((text or "").split())
    return text[:limit]
