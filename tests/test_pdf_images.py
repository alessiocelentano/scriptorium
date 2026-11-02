import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptorium.pdf_images import (
    ExtractedImage,
    _describe,
    _is_useful_image,
    _TextBlock,
    extract_pdf_images,
    images_for_pages,
    page_numbers,
    publish_images,
)
from scriptorium.warnings import WarningCode

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048


class PageMarkerTests(unittest.TestCase):
    def test_page_numbers_reads_sources_markers(self):
        text = "[page 1]\nintro\n\n[page 2]\n\n[page 3]\nbody"
        self.assertEqual(page_numbers(text), {1, 2, 3})

    def test_page_numbers_without_markers_is_empty(self):
        self.assertEqual(page_numbers("plain text"), set())

    def test_images_for_pages_keeps_only_matching_pages(self):
        def image(identifier: str, page: int) -> ExtractedImage:
            return ExtractedImage(
                id=identifier,
                page=page,
                name=f"{identifier}.png",
                vault_path=f"x/{identifier}.png",
                file="",
                width=10,
                height=10,
            )

        images = [image("a", 1), image("b", 4)]
        self.assertEqual([image.id for image in images_for_pages(images, {4})], ["b"])


class DescribeTests(unittest.TestCase):
    def test_caption_below_the_image_wins(self):
        blocks = [
            _TextBlock(top=10, left=0, width=100, height=10, text="Body text"),
            _TextBlock(top=300, left=0, width=100, height=10, text="Figure 2: Overview"),
            _TextBlock(top=600, left=0, width=100, height=10, text="Far away"),
        ]
        caption, nearby = _describe(blocks, top=100, left=0, height=150)
        self.assertEqual(caption, "Figure 2: Overview")
        self.assertIn("Body text", nearby)

    def test_no_caption_leaves_it_empty(self):
        blocks = [_TextBlock(top=10, left=0, width=100, height=10, text="Just prose")]
        caption, nearby = _describe(blocks, top=100, left=0, height=50)
        self.assertEqual(caption, "")
        self.assertIn("Just prose", nearby)


class ImageFilterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_tiny_images_are_skipped(self):
        path = self.root / "icon.png"
        path.write_bytes(PNG_HEADER)
        self.assertFalse(_is_useful_image(path, 16, 16))

    def test_large_images_are_kept(self):
        path = self.root / "figure.png"
        path.write_bytes(PNG_HEADER)
        self.assertTrue(_is_useful_image(path, 400, 300))


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.workdir = self.root / "job"

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_pdftohtml_is_a_warning(self):
        pdf = self.root / "slides.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        with mock.patch("scriptorium.pdf_images.shutil.which", return_value=None):
            harvest = extract_pdf_images(
                pdf,
                workdir=self.workdir,
                vault_root=self.vault,
                attachment_root=self.vault / "Attachments",
                source_id="s01",
                slug="slides",
            )
        self.assertEqual(harvest.images, [])
        self.assertEqual(harvest.warnings[0].code, WarningCode.IMAGE_EXTRACTION_FAILED)

    def test_exported_images_are_deduplicated_and_captioned(self):
        pdf = self.root / "slides.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        output = self.workdir / "pdf-s01"
        output.mkdir(parents=True)
        figure = PNG_HEADER + b"figure"
        (output / "document-1_1.png").write_bytes(figure)
        (output / "document-1_2.png").write_bytes(figure)  # smask duplicate
        (output / "document-1_3.png").write_bytes(b"")
        xml = (
            '<pdf2xml><page number="1">'
            '<text top="10" left="0" width="100" height="10">Intro text</text>'
            '<text top="300" left="0" width="100" height="10">Figure 1: Block diagram</text>'
            '<image top="100" left="50" width="400" height="150" src="document-1_1.png"/>'
            '<image top="100" left="50" width="400" height="150" src="document-1_2.png"/>'
            '<image top="500" left="50" width="20" height="20" src="document-1_3.png"/>'
            "</page></pdf2xml>"
        )
        (output / "document.xml").write_text(xml, encoding="utf-8")
        completed = mock.Mock(returncode=0, stderr="", stdout="")
        with (
            mock.patch("scriptorium.pdf_images.shutil.which", return_value="/usr/bin/pdftohtml"),
            mock.patch("scriptorium.pdf_images.subprocess.run", return_value=completed),
        ):
            harvest = extract_pdf_images(
                pdf,
                workdir=self.workdir,
                vault_root=self.vault,
                attachment_root=self.vault / "Attachments",
                source_id="s01",
                slug="slides",
            )
        self.assertEqual(len(harvest.images), 1)
        image = harvest.images[0]
        self.assertEqual(image.name, "s01-p01-01.png")
        self.assertEqual(image.caption, "Figure 1: Block diagram")
        self.assertEqual(image.vault_path, "Attachments/scriptorium/slides/s01-p01-01.png")


    def test_page_sized_image_is_not_published_as_a_figure(self):
        pdf = self.root / "scanned.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        output = self.workdir / "pdf-s01"
        output.mkdir(parents=True)
        (output / "document-1_1.png").write_bytes(PNG_HEADER + b"page")
        (output / "document-1_2.png").write_bytes(PNG_HEADER + b"figure")
        xml = (
            '<pdf2xml><page number="1" width="1000" height="1000">'
            '<text top="10" left="0" width="100" height="10">Slide title</text>'
            '<image top="0" left="0" width="980" height="980" src="document-1_1.png"/>'
            '<image top="200" left="100" width="400" height="150" src="document-1_2.png"/>'
            "</page></pdf2xml>"
        )
        (output / "document.xml").write_text(xml, encoding="utf-8")
        completed = mock.Mock(returncode=0, stderr="", stdout="")
        with (
            mock.patch("scriptorium.pdf_images.shutil.which", return_value="/usr/bin/pdftohtml"),
            mock.patch("scriptorium.pdf_images.subprocess.run", return_value=completed),
        ):
            harvest = extract_pdf_images(
                pdf,
                workdir=self.workdir,
                vault_root=self.vault,
                attachment_root=self.vault / "Attachments",
                source_id="s01",
                slug="scanned",
            )
        self.assertEqual([image.name for image in harvest.images], ["s01-p01-01.png"])


class PublishTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_publish_copies_into_the_vault_once(self):
        source = self.root / "figure.png"
        source.write_bytes(PNG_HEADER)
        image = ExtractedImage(
            id="s01-1-1",
            page=1,
            name="s01-p01-01.png",
            vault_path="Attachments/scriptorium/slides/s01-p01-01.png",
            file=str(source),
            width=400,
            height=300,
        )
        published = publish_images([image], self.vault)
        self.assertEqual(published, [image.vault_path])
        target = self.vault / image.vault_path
        self.assertEqual(target.read_bytes(), PNG_HEADER)

        source.write_bytes(PNG_HEADER + b"new")
        publish_images([image], self.vault)
        self.assertEqual(target.read_bytes(), PNG_HEADER, "existing attachment is kept")

    def test_round_trip_keeps_every_field(self):
        image = ExtractedImage(
            id="s01-1-1",
            page=3,
            name="s01-p03-01.png",
            vault_path="Attachments/a.png",
            file="/tmp/a.png",
            width=10,
            height=20,
            caption="Figure 1",
            nearby_text="near",
        )
        restored = ExtractedImage.from_dict(image.to_dict())
        self.assertEqual(restored, image)


if __name__ == "__main__":
    unittest.main()
