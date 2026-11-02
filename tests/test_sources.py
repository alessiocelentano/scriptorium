import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scriptorium.sources import (
    PageText,
    SourceFile,
    classify,
    clean_course_name,
    list_sources,
    read_source,
)
from scriptorium.warnings import WarningCode

DOCX_XML = (
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p></w:body></w:document>"
)

PPTX_XML = (
    '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
    ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
    "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide title</a:t></a:r></a:p>"
    "<a:p><a:r><a:t>Slide bullet</a:t></a:r></a:p></p:txBody></p:sp>"
    "</p:spTree></p:cSld></p:sld>"
)

XLSX_SHARED = (
    '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    "<si><t>Alpha</t></si><si><t>Beta</t></si></sst>"
)

XLSX_SHEET = (
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row></sheetData>'
    "</worksheet>"
)


class SourceListTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_classify_separates_media_from_documents(self):
        self.assertEqual(classify(Path("lecture.mp4")), "media")
        self.assertEqual(classify(Path("slides.pdf")), "document")
        self.assertEqual(classify(Path("archive.bin")), "document")

    def test_list_sources_numbers_and_deduplicates(self):
        sources = list_sources([self.root / "a.mp4", self.root / "b.pdf", self.root / "a.mp4"])
        self.assertEqual([source.id for source in sources], ["s01", "s02"])
        self.assertEqual([source.kind for source in sources], ["media", "document"])

    def test_clean_course_name_strips_moc_and_separators(self):
        self.assertEqual(clean_course_name("Foundations_of_AI-MOC"), "Foundations of AI")


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def source(self, name: str) -> SourceFile:
        return SourceFile(id="s01", path=self.root / name, kind="document")

    def test_plain_text_is_read_verbatim(self):
        path = self.root / "notes.md"
        path.write_text("# Heading\n\nBody text.\n", encoding="utf-8")
        result = read_source(self.source("notes.md"))
        self.assertEqual(result.warnings, [])
        self.assertIn("Body text.", result.text)

    def test_missing_file_is_a_warning_not_an_error(self):
        result = read_source(self.source("missing.md"))
        self.assertEqual(result.text, "")
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_UNREADABLE)

    def test_empty_document_reports_no_text(self):
        (self.root / "empty.md").write_text("   \n", encoding="utf-8")
        result = read_source(self.source("empty.md"))
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_TEXT_EMPTY)

    def test_unknown_suffix_is_reported_without_reading(self):
        (self.root / "archive.bin").write_bytes(b"\x00\x01")
        result = read_source(self.source("archive.bin"))
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_UNSUPPORTED)

    def test_media_sources_are_not_read_as_documents(self):
        media = SourceFile(id="s01", path=self.root / "lecture.mp4", kind="media")
        self.assertEqual(read_source(media).text, "")

    def test_docx_paragraphs_are_extracted(self):
        path = self.root / "notes.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", DOCX_XML)
        result = read_source(self.source("notes.docx"))
        self.assertIn("First paragraph", result.text)
        self.assertIn("Second paragraph", result.text)

    def test_pptx_slides_are_extracted_in_order(self):
        path = self.root / "slides.pptx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("ppt/slides/slide10.xml", PPTX_XML)
            archive.writestr("ppt/slides/slide2.xml", PPTX_XML)
        result = read_source(self.source("slides.pptx"))
        self.assertIn("[slide 1]\nSlide title", result.text)
        self.assertIn("[slide 2]", result.text)

    def test_xlsx_rows_resolve_shared_strings(self):
        path = self.root / "data.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/sharedStrings.xml", XLSX_SHARED)
            archive.writestr("xl/worksheets/sheet1.xml", XLSX_SHEET)
        result = read_source(self.source("data.xlsx"))
        self.assertIn("Alpha\tBeta", result.text)

    def test_html_keeps_text_and_drops_scripts(self):
        path = self.root / "page.html"
        path.write_text(
            "<html><body><h1>Title</h1><p>Body</p>"
            "<script>ignore()</script><style>p{}</style></body></html>",
            encoding="utf-8",
        )
        result = read_source(self.source("page.html"))
        self.assertIn("Title", result.text)
        self.assertIn("Body", result.text)
        self.assertNotIn("ignore", result.text)

    def test_code_only_zip_is_reported_as_unreadable(self):
        path = self.root / "broken.docx"
        path.write_bytes(b"not a zip")
        result = read_source(self.source("broken.docx"))
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_UNREADABLE)


class ScannedPdfTests(unittest.TestCase):
    """A PDF whose pages lack a text layer, with an injected page reader."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "slides.pdf").write_bytes(b"%PDF-1.4")

    def tearDown(self):
        self._tmp.cleanup()

    def read(self, stdout: str, reader=None):
        pdf = SourceFile(id="s01", path=self.root / "slides.pdf", kind="document")
        completed = mock.Mock(returncode=0, stderr="", stdout=stdout)
        with (
            mock.patch("scriptorium.sources.shutil.which", return_value="/usr/bin/pdftotext"),
            mock.patch("scriptorium.sources.subprocess.run", return_value=completed),
        ):
            return read_source(pdf, page_reader=reader)

    def test_pages_keep_their_numbers_and_only_empty_pages_use_the_reader(self):
        calls: list[int] = []

        def reader(path, page):
            calls.append(page)
            return PageText("Recovered scan text")

        result = self.read(
            "Page one has plenty of text.\f\fPage three has plenty of text.\f", reader
        )
        self.assertEqual(calls, [2])
        self.assertIn("[page 1]\nPage one has plenty of text.", result.text)
        self.assertIn("[page 2]\nRecovered scan text", result.text)
        self.assertIn("[page 3]\nPage three has plenty of text.", result.text)

    def test_page_with_text_and_figures_is_not_sent_to_the_reader(self):
        def reader(path, page):
            raise AssertionError(f"page {page} should keep its text layer")

        result = self.read("A slide with a title and one figure.\f", reader)
        self.assertIn("A slide with a title and one figure.", result.text)

    def test_a_short_title_only_page_is_sent_to_the_reader(self):
        calls: list[int] = []

        def reader(path, page):
            calls.append(page)
            return PageText("Quiz\n\n1. What is a derivative?")

        result = self.read("Quiz\f", reader)
        self.assertEqual(calls, [1])
        self.assertIn("What is a derivative?", result.text)

    def test_reader_warning_is_carried_to_the_result(self):
        from scriptorium.warnings import RunWarning

        def reader(path, page):
            return PageText("", RunWarning(code=WarningCode.SCANNED_PDF_CAPPED, message="capped"))

        result = self.read("\f\f", reader)
        self.assertEqual(result.warnings[0].code, WarningCode.SCANNED_PDF_CAPPED)

    def test_pdf_without_any_text_and_no_reader_reports_no_text(self):
        result = self.read("\f\f")
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_TEXT_EMPTY)

    def test_pdf_without_any_text_after_the_reader_reports_no_text(self):
        result = self.read("\f\f", lambda path, page: PageText(""))
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_TEXT_EMPTY)

    def test_single_page_without_text_reports_no_text(self):
        result = self.read("")
        self.assertEqual(result.warnings[0].code, WarningCode.SOURCE_TEXT_EMPTY)


if __name__ == "__main__":
    unittest.main()
