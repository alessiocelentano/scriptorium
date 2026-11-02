import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.config import Config
from scriptorium.errors import LLMError
from scriptorium.log import Logger
from scriptorium.scanned import ScannedPdfReader
from scriptorium.warnings import WarningCode

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048


class FakeClient:
    def __init__(
        self,
        text: str | Exception = "Vision page text",
        supported: bool | None = True,
    ):
        self.text = text
        self.supported = supported
        self.calls: list[Path] = []

    def image_input_supported(self) -> bool | None:
        return self.supported

    def chat_image(self, prompt: str, image: Path, *, phase: str = "image") -> str:
        self.calls.append(image)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class ScannedReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.pdf = self.root / "scans.pdf"
        self.pdf.write_bytes(b"%PDF-1.4")
        self.cache = self.root / "pages"

    def tearDown(self):
        self._tmp.cleanup()

    def reader(self, *, client=None, **overrides) -> ScannedPdfReader:
        values: dict[str, object] = {
            "notes_root": self.vault,
            "artifacts_dir": self.root / "home",
            "scanned_pdf": "vision",
        }
        values.update(overrides)
        config = Config(**values)  # type: ignore[arg-type]
        return ScannedPdfReader(
            config,
            Logger(sink=lambda _line: None),
            cache_dir=self.cache,
            client=client,
        )

    def warning_code(self, warning):
        self.assertIsNotNone(warning)
        return warning.code

    def png(self, page: int) -> Path:
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"{page:03d}.png"
        path.write_bytes(PNG)
        return path

    # -- cache --------------------------------------------------------------- #

    def test_cached_page_is_returned_without_rendering(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / "001.txt").write_text("Cached page text\n", encoding="utf-8")
        client = FakeClient()
        with mock.patch("scriptorium.scanned.subprocess.run") as run:
            result = self.reader(client=client)(self.pdf, 1)
        self.assertEqual(result.text, "Cached page text")
        run.assert_not_called()
        self.assertEqual(client.calls, [])

    def test_recovered_text_is_cached(self):
        self.png(1)
        reader = self.reader(client=FakeClient("Recovered"))
        reader(self.pdf, 1)
        self.assertEqual((self.cache / "001.txt").read_text(encoding="utf-8"), "Recovered\n")

    # -- OCR ----------------------------------------------------------------- #

    def test_ocr_reads_the_rendered_page(self):
        self.png(1)
        completed = mock.Mock(returncode=0, stdout="OCR page text", stderr="")
        client = FakeClient()
        reader = self.reader(scanned_pdf="ocr", ocr_languages="eng+ita", client=client)
        with (
            mock.patch("scriptorium.scanned.shutil.which", return_value="/usr/bin/tesseract"),
            mock.patch("scriptorium.scanned.subprocess.run", return_value=completed) as run,
        ):
            result = reader(self.pdf, 1)
        self.assertEqual(result.text, "OCR page text")
        self.assertEqual(client.calls, [])
        self.assertIn(str((self.cache / "001.png").resolve()), run.call_args.args[0])
        self.assertIn("-l", run.call_args.args[0])
        self.assertIn("eng+ita", run.call_args.args[0])

    def test_missing_tesseract_is_a_warning(self):
        self.png(1)
        with mock.patch("scriptorium.scanned.shutil.which", return_value=None):
            result = self.reader(scanned_pdf="ocr")(self.pdf, 1)
        self.assertEqual(result.text, "")
        self.assertEqual(self.warning_code(result.warning), WarningCode.SCANNED_PDF_UNAVAILABLE)

    # -- vision -------------------------------------------------------------- #

    def test_vision_sends_the_page_image(self):
        self.png(1)
        client = FakeClient("Vision text")
        result = self.reader(client=client)(self.pdf, 1)
        self.assertEqual(result.text, "Vision text")
        self.assertEqual(len(client.calls), 1)

    def test_vision_rejected_by_the_model_is_a_warning(self):
        self.png(1)
        client = FakeClient(supported=False)
        result = self.reader(client=client)(self.pdf, 1)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.warning_code(result.warning), WarningCode.SCANNED_PDF_UNAVAILABLE)

    def test_vision_failure_is_a_warning(self):
        self.png(1)
        client = FakeClient(LLMError("boom"))
        result = self.reader(client=client)(self.pdf, 1)
        self.assertEqual(result.text, "")
        self.assertEqual(self.warning_code(result.warning), WarningCode.SCANNED_PDF_FALLBACK_FAILED)

    def test_blank_page_sentinel_is_treated_as_empty(self):
        self.png(1)
        result = self.reader(client=FakeClient("blank"))(self.pdf, 1)
        self.assertEqual(result.text, "")
        self.assertIsNone(result.warning)
        self.assertFalse((self.cache / "001.txt").exists())

    # -- limits and rendering ------------------------------------------------ #

    def test_page_cap_skips_later_pages(self):
        self.png(1)
        self.png(2)
        client = FakeClient()
        reader = self.reader(client=client, max_scanned_pages=1)
        self.assertEqual(reader(self.pdf, 1).text, "Vision page text")
        capped = reader(self.pdf, 2)
        self.assertEqual(capped.text, "")
        self.assertEqual(self.warning_code(capped.warning), WarningCode.SCANNED_PDF_CAPPED)
        self.assertEqual(len(client.calls), 1)

    def test_missing_pdftoppm_is_a_warning(self):
        with mock.patch("scriptorium.scanned.shutil.which", return_value=None):
            result = self.reader(client=FakeClient())(self.pdf, 1)
        self.assertEqual(self.warning_code(result.warning), WarningCode.SCANNED_PDF_UNAVAILABLE)

    def test_render_failure_is_a_warning(self):
        completed = mock.Mock(returncode=1, stdout="", stderr="broken pdf")
        with (
            mock.patch("scriptorium.scanned.shutil.which", return_value="/usr/bin/pdftoppm"),
            mock.patch("scriptorium.scanned.subprocess.run", return_value=completed),
        ):
            result = self.reader(client=FakeClient())(self.pdf, 1)
        self.assertEqual(self.warning_code(result.warning), WarningCode.SCANNED_PDF_FALLBACK_FAILED)
        self.assertIsNotNone(result.warning)
        self.assertIn("broken pdf", result.warning.message if result.warning else "")


if __name__ == "__main__":
    unittest.main()
