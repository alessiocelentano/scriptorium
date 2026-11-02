"""Read PDF pages that have no usable text layer.

A scanned page is rendered to a PNG with ``pdftoppm`` and then read by one of
two backends, both chosen in the settings:

- ``vision``: one chat call to the configured model (the same provider used by
  the pipeline, which must accept image input);
- ``ocr``: local Tesseract, no network and no API cost, but plain typed scans
  only: formulas, diagrams and multi-column layouts are lost.

The module returns text, never raising: a missing binary, a rejected image or a
failed conversion becomes a warning the run carries to the report.  Page text is
cached per job, so resuming a run does not pay for the same page twice.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from .config import SCANNED_PDF_OCR, Config
from .errors import LLMError
from .llm.client import DeepSeekClient
from .log import Logger
from .prompting import load_prompt, render
from .sources import PageText
from .warnings import RunWarning, WarningCode

RENDER_DPI = 200
RENDER_TIMEOUT = 120
OCR_TIMEOUT = 120
BLANK_PAGE_SENTINEL = "blank"


class ScannedPdfReader:
    """Callable ``(pdf, page) -> PageText`` used by ``sources.read_source``."""

    def __init__(
        self,
        config: Config,
        logger: Logger,
        *,
        cache_dir: Path,
        client: DeepSeekClient | None = None,
        client_factory: Callable[[], DeepSeekClient] | None = None,
    ):
        self.config = config
        self.logger = logger
        self.cache_dir = cache_dir
        self.client = client
        self.client_factory = client_factory
        self._template = load_prompt("transcribe_page.md")
        self._read_pages = 0

    def __call__(self, pdf: Path, page: int) -> PageText:
        cached = self._cached(page)
        if cached is not None:
            return PageText(cached)

        if self._read_pages >= self.config.max_scanned_pages:
            return PageText(warning=self._capped_warning(pdf))

        png = self._render(pdf, page)
        if isinstance(png, RunWarning):
            return PageText(warning=png)

        self._read_pages += 1
        if self.config.scanned_pdf == SCANNED_PDF_OCR:
            text, warning = self._ocr(pdf, png, page)
        else:
            text, warning = self._vision(pdf, png, page)

        text = self._clean(text)
        if text:
            self._store(page, text)
            self.logger.info(f"{pdf.name}: page {page} read with {self.config.scanned_pdf}")
        return PageText(text, warning)

    # -- cache --------------------------------------------------------------- #

    def _cache_path(self, page: int) -> Path:
        return self.cache_dir / f"{page:03d}.txt"

    def _cached(self, page: int) -> str | None:
        path = self._cache_path(page)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _store(self, page: int, text: str) -> None:
        path = self._cache_path(page)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(text.strip() + "\n", encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            self.logger.debug(f"could not cache page {page}: {exc}")

    # -- rendering ----------------------------------------------------------- #

    def _render(self, pdf: Path, page: int) -> Path | RunWarning:
        png = self.cache_dir / f"{page:03d}.png"
        if png.is_file() and png.stat().st_size > 0:
            return png
        binary = shutil.which("pdftoppm")
        if not binary:
            return self._unavailable(
                pdf,
                "pdftoppm is not on PATH; install poppler to render scanned pages.",
            )
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    binary,
                    "-png",
                    "-r",
                    str(RENDER_DPI),
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-singlefile",
                    str(pdf),
                    str(png.with_suffix("")),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=RENDER_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return self._failed(pdf, page, f"cannot render the page: {exc}")
        if completed.returncode != 0 or not png.is_file():
            detail = (completed.stderr or "pdftoppm produced no image").strip()
            return self._failed(pdf, page, f"cannot render the page: {detail}")
        return png

    # -- backends ------------------------------------------------------------ #

    def _ocr(self, pdf: Path, png: Path, page: int) -> tuple[str, RunWarning | None]:
        binary = shutil.which("tesseract")
        if not binary:
            return "", self._unavailable(
                pdf,
                "tesseract is not on PATH; install it (with the language data) "
                'or set scanned_pdf = "vision".',
            )
        # Leptonica (behind tesseract) refuses paths under a symlinked prefix
        # such as /tmp on macOS, so hand it the canonical path.
        command = [binary, str(png.resolve()), "stdout"]
        if self.config.ocr_languages:
            command.extend(["-l", self.config.ocr_languages])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=OCR_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return "", self._failed(pdf, page, f"tesseract failed: {exc}")
        if completed.returncode != 0:
            detail = (completed.stderr or "tesseract failed").strip()
            return "", self._failed(pdf, page, f"tesseract failed: {detail}")
        return completed.stdout, None

    def _vision(self, pdf: Path, png: Path, page: int) -> tuple[str, RunWarning | None]:
        client = self._resolve_client()
        if client is None:
            return "", self._unavailable(pdf, "no model client is available for scanning.")
        supported = client.image_input_supported()
        if supported is False:
            return "", self._unavailable(
                pdf,
                'the configured model does not read images; set scanned_pdf = "ocr" '
                'or image_input = "enabled" with a vision model.',
            )
        prompt = render(self._template, {"page": page})
        try:
            return client.chat_image(prompt, png, phase="page_scan"), None
        except LLMError as exc:
            return "", self._failed(pdf, page, str(exc))

    # -- helpers ------------------------------------------------------------- #

    def _resolve_client(self) -> DeepSeekClient | None:
        """Build the model client only when a page actually needs vision."""
        if self.client is None and self.client_factory is not None:
            self.client = self.client_factory()
        return self.client

    def _clean(self, text: str) -> str:
        value = (text or "").strip()
        if value.casefold() == BLANK_PAGE_SENTINEL:
            return ""
        return value

    def _capped_warning(self, pdf: Path) -> RunWarning:
        return RunWarning(
            code=WarningCode.SCANNED_PDF_CAPPED,
            message=(
                f"{pdf.name}: stopped reading scanned pages after "
                f"{self.config.max_scanned_pages}; later pages keep no text. "
                "Raise max_scanned_pages to read them."
            ),
            source=str(pdf),
        )

    def _unavailable(self, pdf: Path, message: str) -> RunWarning:
        return RunWarning(
            code=WarningCode.SCANNED_PDF_UNAVAILABLE,
            message=f"{pdf.name}: {message}",
            source=str(pdf),
        )

    def _failed(self, pdf: Path, page: int, detail: str) -> RunWarning:
        where = f"page {page}" if page else "a page"
        return RunWarning(
            code=WarningCode.SCANNED_PDF_FALLBACK_FAILED,
            message=f"{pdf.name}: could not read {where} ({detail}).",
            source=str(pdf),
            detail=detail,
        )


