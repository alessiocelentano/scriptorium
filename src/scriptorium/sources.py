"""Reading source material into plain text.

A run accepts any number of sources: lecture recordings plus documents such as
slides, notes, PDFs or spreadsheets.  Media sources are transcribed by
``asr.py``; documents are converted to text here so the same extraction phase
can treat every source uniformly.  Reading never raises: an unreadable source
yields an empty text plus a warning, so one broken file cannot kill a run.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from .pdf_images import ExtractedImage
from .warnings import RunWarning, WarningCode

MEDIA_SUFFIXES = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

TEXT_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".markdown",
    ".md",
    ".org",
    ".rst",
    ".tex",
    ".txt",
}

# Suffixes we know how to turn into text; anything else is reported, never
# rejected, because the input is a free list of paths.
SUPPORTED_DOCUMENT_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx", ".pptx", ".xlsx"}


@dataclass(frozen=True)
class SourceFile:
    """One input path with its role inferred from the extension."""

    id: str
    path: Path
    kind: str  # "media" or "document"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        return f"{self.name} ({self.kind})"


@dataclass(frozen=True)
class TextSource:
    """A source as the extraction phase sees it: an id plus plain text."""

    id: str
    name: str
    kind: str
    path: Path
    text: str
    images: tuple[ExtractedImage, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.kind})"


@dataclass
class ReadResult:
    text: str = ""
    warnings: list[RunWarning] = field(default_factory=list)
    pages: tuple[PdfPage, ...] = ()


@dataclass(frozen=True)
class PdfPage:
    """One PDF page with the text found in its text layer."""

    number: int
    text: str = ""

    @property
    def lacks_text(self) -> bool:
        """Whether the page needs the scanned-page fallback.

        A page can hold a text layer and figures at the same time; that page is
        not scanned and keeps its text.  The threshold is deliberately low: a
        slide with a title and a diagram usually has a few words, and losing
        them is worse than paying one extra read.
        """
        return len(" ".join(self.text.split())) < MIN_PAGE_CHARS


@dataclass(frozen=True)
class PageText:
    """Text recovered for one page plus an optional warning."""

    text: str = ""
    warning: RunWarning | None = None


# Injected by the caller so reading stays free of LLM and subprocess knowledge.
PageReader = Callable[[Path, int], PageText]

# A page with fewer useful characters than this has no usable text layer.
MIN_PAGE_CHARS = 24


def clean_course_name(value: str) -> str:
    """Turn a MOC or source file name into a course label."""
    value = value.replace("_", " ").replace("-", " ").strip()
    value = re.sub(r"\bMOC\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value or "Lecture"


def classify(path: Path) -> str:
    return "media" if path.suffix.lower() in MEDIA_SUFFIXES else "document"


def list_sources(paths: list[Path]) -> list[SourceFile]:
    """Deduplicate and number the sources in input order."""
    seen: set[Path] = set()
    sources: list[SourceFile] = []
    for raw in paths:
        path = Path(raw).expanduser()
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceFile(id=f"s{len(sources) + 1:02d}", path=path, kind=classify(path)))
    return sources


def read_source(source: SourceFile, *, page_reader: PageReader | None = None) -> ReadResult:
    """Return the text of a document source; media sources are skipped.

    ``page_reader`` reads PDF pages that have no usable text layer.  The caller
    injects it (with the configured vision or OCR backend); without it a
    scanned page keeps an empty ``[page N]`` marker and the document is reported
    as having no extractable text.
    """
    if source.kind == "media":
        return ReadResult()
    path = source.path
    if not path.is_file():
        return _unreadable(source, f"file not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _read_pdf_source(source, page_reader)
        if suffix == ".docx":
            text = _read_docx(path)
        elif suffix == ".pptx":
            text = _read_pptx(path)
        elif suffix == ".xlsx":
            text = _read_xlsx(path)
        elif suffix in {".html", ".htm"}:
            text = _read_html(path)
        elif suffix in TEXT_SUFFIXES:
            text = _read_text(path)
        else:
            return ReadResult(
                warnings=[
                    RunWarning(
                        code=WarningCode.SOURCE_UNSUPPORTED,
                        message=(
                            f"{source.name}: no reader for '{suffix or 'no extension'}'; "
                            "the file is listed but contributes no text."
                        ),
                        source=str(path),
                    )
                ]
            )
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return _unreadable(source, str(exc))

    return _with_text(source, text, path)


def _with_text(source: SourceFile, text: str, path: Path) -> ReadResult:
    if not text.strip():
        return ReadResult(
            text="",
            warnings=[
                RunWarning(
                    code=WarningCode.SOURCE_TEXT_EMPTY,
                    message=(
                        f"{source.name}: no extractable text (a scanned PDF needs OCR or "
                        "vision, which this run did not produce)."
                    ),
                    source=str(path),
                )
            ],
        )
    return ReadResult(text=text)


def _unreadable(source: SourceFile, detail: str) -> ReadResult:
    return ReadResult(
        text="",
        warnings=[
            RunWarning(
                code=WarningCode.SOURCE_UNREADABLE,
                message=f"{source.name}: {detail}",
                source=str(source.path),
                detail=detail,
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_source(source: SourceFile, page_reader: PageReader | None) -> ReadResult:
    path = source.path
    try:
        pages = _pdf_pages(path)
    except OSError as exc:
        return _unreadable(source, str(exc))

    warnings: list[RunWarning] = []
    if page_reader is not None:
        pages = _fill_scanned_pages(path, pages, page_reader, warnings)

    if not any(page.text.strip() for page in pages):
        result = _with_text(source, "", path)
        result.warnings = warnings + result.warnings
        result.pages = tuple(pages)
        return result
    return ReadResult(text=_render_pages(pages), warnings=warnings, pages=tuple(pages))


def _fill_scanned_pages(
    path: Path,
    pages: list[PdfPage],
    page_reader: PageReader,
    warnings: list[RunWarning],
) -> list[PdfPage]:
    filled: list[PdfPage] = []
    for page in pages:
        if not page.lacks_text:
            filled.append(page)
            continue
        recovered = page_reader(path, page.number)
        if recovered.warning is not None:
            warnings.append(recovered.warning)
        text = recovered.text.strip()
        filled.append(PdfPage(page.number, text or page.text))
    return filled


def _render_pages(pages: list[PdfPage]) -> str:
    """Render pages with explicit markers for provenance; single pages stay bare."""
    if len(pages) <= 1:
        return pages[0].text if pages else ""
    blocks: list[str] = []
    for page in pages:
        body = page.text.strip()
        # Keep the marker even for a page without text: it may hold figures.
        blocks.append(f"[page {page.number}]\n{body}" if body else f"[page {page.number}]")
    return "\n\n".join(blocks)


def _pdf_pages(path: Path) -> list[PdfPage]:
    binary = shutil.which("pdftotext")
    if not binary:
        raise OSError("pdftotext is not on PATH; install poppler to read PDF sources")
    completed = subprocess.run(
        [binary, "-layout", str(path), "-"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError((completed.stderr or "pdftotext failed").strip())
    chunks = completed.stdout.split("\f")
    if len(chunks) > 1 and not chunks[-1].strip() and completed.stdout.endswith("\f"):
        chunks.pop()
    pages = [
        PdfPage(number=number, text=chunk.strip()) for number, chunk in enumerate(chunks, start=1)
    ]
    return pages or [PdfPage(number=1)]


_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("word/document.xml")
    root = ElementTree.fromstring(payload)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_DOCX_NS}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_DOCX_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_DOCX_NS}tab":
                parts.append("\t")
            elif node.tag == f"{_DOCX_NS}br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


_PPTX_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _read_pptx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=lambda name: int(re.findall(r"\d+", name)[0]),
        )
        blocks: list[str] = []
        for number, name in enumerate(slides, start=1):
            root = ElementTree.fromstring(archive.read(name))
            paragraphs: list[str] = []
            for paragraph in root.iter(f"{_PPTX_NS}p"):
                text = "".join(node.text or "" for node in paragraph.iter(f"{_PPTX_NS}t")).strip()
                if text:
                    paragraphs.append(text)
            if paragraphs:
                blocks.append(f"[slide {number}]\n" + "\n".join(paragraphs))
    return "\n\n".join(blocks)


_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(f"{_SHEET_NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{_SHEET_NS}t")))
        sheets = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
            ),
            key=lambda name: int(re.findall(r"\d+", name)[0]),
        )
        blocks: list[str] = []
        for number, name in enumerate(sheets, start=1):
            root = ElementTree.fromstring(archive.read(name))
            rows: list[str] = []
            for row in root.iter(f"{_SHEET_NS}row"):
                cells: list[str] = []
                for cell in row.iter(f"{_SHEET_NS}c"):
                    value = cell.find(f"{_SHEET_NS}v")
                    raw = value.text if value is not None and value.text else ""
                    if cell.get("t") == "s" and raw.isdigit():
                        index = int(raw)
                        raw = shared[index] if index < len(shared) else raw
                    elif cell.get("t") == "inlineStr":
                        raw = "".join(node.text or "" for node in cell.iter(f"{_SHEET_NS}t"))
                    cells.append(raw.strip())
                line = "\t".join(cell for cell in cells if cell)
                if line.strip():
                    rows.append(line)
            if rows:
                blocks.append(f"[sheet {number}]\n" + "\n".join(rows))
    return "\n\n".join(blocks)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(html.unescape(data))

    def text(self) -> str:
        joined = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def _read_html(path: Path) -> str:
    parser = _TextExtractor()
    parser.feed(_read_text(path))
    parser.close()
    return parser.text()
