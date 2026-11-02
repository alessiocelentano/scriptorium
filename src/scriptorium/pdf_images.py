"""Extract figures from PDF sources and publish them into the vault.

The PDF is rendered to XML by ``pdftohtml`` so every image keeps its page and
bounding box.  Captions and nearby text give the language model enough context
to place a figure in the right note section; the model never sees the pixels
unless the configured provider accepts image input.

Extraction is deterministic and idempotent: images land in the job directory,
and only ``publish_images`` copies them inside the vault, at apply time.  A
dry run therefore never leaves attachments behind.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .warnings import RunWarning, WarningCode

CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|table|algorithm|listing|chart|diagram|graph|equation|scheme)"
    r"\s*[\dIVXivx]+",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
MIN_IMAGE_SIDE = 48
MIN_IMAGE_PIXELS = 48 * 48
# An image covering most of the page is the page raster (a scan or a full-page
# background), not a figure to publish inside the note.
PAGE_IMAGE_RATIO = 0.85
PDF_TIMEOUT = 180


@dataclass(frozen=True)
class ExtractedImage:
    """One raster figure harvested from a PDF source."""

    id: str
    page: int
    name: str
    vault_path: str
    file: str
    width: int
    height: int
    caption: str = ""
    nearby_text: str = ""

    @property
    def label(self) -> str:
        return f"{self.vault_path} (page {self.page})"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "page": self.page,
            "name": self.name,
            "vault_path": self.vault_path,
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "caption": self.caption,
            "nearby_text": self.nearby_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ExtractedImage:
        return cls(
            id=str(payload.get("id") or ""),
            page=int(str(payload.get("page") or 1)),
            name=str(payload.get("name") or ""),
            vault_path=str(payload.get("vault_path") or ""),
            file=str(payload.get("file") or ""),
            width=int(str(payload.get("width") or 0)),
            height=int(str(payload.get("height") or 0)),
            caption=str(payload.get("caption") or ""),
            nearby_text=str(payload.get("nearby_text") or ""),
        )


@dataclass
class ImageHarvest:
    images: list[ExtractedImage] = field(default_factory=list)
    warnings: list[RunWarning] = field(default_factory=list)


@dataclass(frozen=True)
class _TextBlock:
    top: int
    left: int
    width: int
    height: int
    text: str


def extract_pdf_images(
    pdf: Path,
    *,
    workdir: Path,
    vault_root: Path,
    attachment_root: Path,
    source_id: str,
    slug: str,
) -> ImageHarvest:
    """Harvest figures from one PDF into ``workdir``."""
    harvest = ImageHarvest()
    binary = shutil.which("pdftohtml")
    if not binary:
        harvest.warnings.append(
            RunWarning(
                code=WarningCode.IMAGE_EXTRACTION_FAILED,
                message=f"{pdf.name}: pdftohtml is not on PATH; figures were skipped.",
                source=str(pdf),
            )
        )
        return harvest

    output_dir = workdir / f"pdf-{source_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "document"
    try:
        completed = subprocess.run(
            [binary, "-xml", "-hidden", "-q", str(pdf), str(prefix)],
            capture_output=True,
            text=True,
            timeout=PDF_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        harvest.warnings.append(_extraction_warning(pdf, str(exc)))
        return harvest

    xml_path = prefix.with_suffix(".xml")
    if completed.returncode != 0 or not xml_path.is_file():
        detail = (completed.stderr or "pdftohtml produced no XML").strip()
        harvest.warnings.append(_extraction_warning(pdf, detail))
        return harvest

    try:
        root = ElementTree.fromstring(xml_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ElementTree.ParseError) as exc:
        harvest.warnings.append(_extraction_warning(pdf, str(exc)))
        return harvest

    seen_hashes: set[str] = set()
    for page in root.iter("page"):
        number = int(page.get("number") or 1)
        texts = _text_blocks(page)
        page_area = _page_area(page)
        counter = 0
        for image in page.iter("image"):
            src = output_dir / (image.get("src") or "")
            if not src.is_file():
                continue
            width = int(float(image.get("width") or 0))
            height = int(float(image.get("height") or 0))
            if not _is_useful_image(src, width, height):
                continue
            if _covers_page(width, height, page_area):
                continue
            digest = _digest(src)
            if digest is None or digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            counter += 1
            top = int(float(image.get("top") or 0))
            left = int(float(image.get("left") or 0))
            caption, nearby = _describe(texts, top=top, left=left, height=height)
            name = f"{source_id}-p{number:02d}-{counter:02d}.png"
            destination = attachment_root / "scriptorium" / slug / name
            vault_path = destination.relative_to(vault_root).as_posix()
            harvest.images.append(
                ExtractedImage(
                    id=f"{source_id}-{number}-{counter}",
                    page=number,
                    name=name,
                    vault_path=vault_path,
                    file=str(src),
                    width=width,
                    height=height,
                    caption=caption,
                    nearby_text=nearby,
                )
            )
    return harvest


def publish_images(images: list[ExtractedImage], vault_root: Path) -> list[str]:
    """Copy job-local images into the vault; returns the published paths."""
    published: list[str] = []
    for image in images:
        source = Path(image.file)
        if not source.is_file():
            continue
        destination = vault_root / image.vault_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        published.append(image.vault_path)
    return published


def images_for_pages(
    images: Sequence[ExtractedImage],
    pages: set[int],
) -> list[ExtractedImage]:
    return [image for image in images if image.page in pages]


def page_numbers(text: str) -> set[int]:
    """Pages referenced by the ``[page N]`` markers that ``sources.py`` writes."""
    return {int(match) for match in re.findall(r"\[page (\d+)\]", text)}


def _extraction_warning(pdf: Path, detail: str) -> RunWarning:
    return RunWarning(
        code=WarningCode.IMAGE_EXTRACTION_FAILED,
        message=f"{pdf.name}: could not extract figures ({detail}).",
        source=str(pdf),
        detail=detail,
    )


def _page_area(page: ElementTree.Element) -> int:
    """Page area in the XML coordinate space, zero when the attributes are absent."""
    try:
        return int(float(page.get("width") or 0)) * int(float(page.get("height") or 0))
    except ValueError:
        return 0


def _covers_page(width: int, height: int, page_area: int) -> bool:
    """A page-sized image is the page itself: it must not be published as a figure."""
    if page_area <= 0:
        return False
    return width * height >= PAGE_IMAGE_RATIO * page_area


def _is_useful_image(path: Path, width: int, height: int) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        return False
    if width * height < MIN_IMAGE_PIXELS:
        return False
    # Very large, very light images are usually page backgrounds.
    try:
        if path.stat().st_size < 1024:
            return False
    except OSError:
        return False
    return True


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _text_blocks(page: ElementTree.Element) -> list[_TextBlock]:
    blocks: list[_TextBlock] = []
    for element in page.iter("text"):
        text = " ".join((element.text or "").split())
        if not text:
            continue
        blocks.append(
            _TextBlock(
                top=int(float(element.get("top") or 0)),
                left=int(float(element.get("left") or 0)),
                width=int(float(element.get("width") or 0)),
                height=int(float(element.get("height") or 0)),
                text=text,
            )
        )
    blocks.sort(key=lambda block: (block.top, block.left))
    return blocks


def _describe(blocks: list[_TextBlock], *, top: int, left: int, height: int) -> tuple[str, str]:
    """Return (caption, nearby text) for one image using its geometry."""
    if not blocks:
        return "", ""
    bottom = top + height

    def gap(block: _TextBlock) -> int:
        if block.top >= bottom:
            return block.top - bottom
        if block.top + block.height <= top:
            return top - (block.top + block.height)
        return 0  # overlapping

    captions = [block for block in blocks if CAPTION_RE.match(block.text)]
    caption = ""
    if captions:
        caption = min(captions, key=gap).text
    nearest = sorted(blocks, key=gap)[:3]
    nearby = " ".join(block.text for block in nearest)
    if len(nearby) > 400:
        nearby = nearby[:400].rsplit(" ", 1)[0] + " [...]"
    return caption, nearby
