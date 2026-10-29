"""Document Ingestion Adapters (DOC, DOCX, PDF) and LVM Layout Assistant (Section 7.3)."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Any, Mapping

from offline_rag.application.ports import DocumentExtractorPort, LvmLayoutAssistantPort
from offline_rag.domain.knowledge import (
    DocumentFormat,
    ExtractedBlock,
    ExtractedDocumentContent,
    ExtractedTable,
)
from offline_rag.domain.shared.errors import DomainValidationError
from offline_rag.domain.shared.value_objects import normalize_persian_text

WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Built-in Word binary style/metadata strings to ignore when extracting OLE2 .doc streams
DOC_STYLE_NOISE: frozenset[str] = frozenset(
    {
        "normal",
        "standard",
        "default style",
        "default paragraph font",
        "table normal",
        "table contents",
        "table heading",
        "no list",
        "list",
        "numbering symbols",
        "horizontal line",
        "preformatted text",
        "body text",
        "heading",
        "heading 1",
        "heading 2",
        "heading 3",
        "heading 4",
        "title",
        "subtitle",
        "caption",
        "index",
        "header",
        "footer",
        "footnote",
        "endnote",
        "quotations",
        "times new roman",
        "arial",
        "calibri",
        "tahoma",
        "b nazanin",
        "vazirmatn",
        "liberationsans",
        "liberation serif",
        "liberation sans",
        "liberation mono",
        "freesans",
        "dejavu sans",
        "microsoft word",
        "root entry",
        "worddocument",
        "summaryinformation",
        "documentsummaryinformation",
        "compobj",
        "1table",
        "0table",
    }
)

HEADING_STYLE_REGEX = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)
NUMBERED_HEADING_REGEX = re.compile(r"^([۰-۹0-9]+[\.\-\)]+)\s+.+$")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _detect_heading_level(style_name: str, text: str) -> int | None:
    """Determine if a paragraph is a heading (1..6) from its Word style or short numbered title."""
    cleaned_style = style_name.strip()
    if cleaned_style:
        m = HEADING_STYLE_REGEX.match(cleaned_style)
        if m:
            return max(1, min(int(m.group(1)), 6))
        if cleaned_style.lower() == "title":
            return 1
        if cleaned_style.lower() == "subtitle":
            return 2

    stripped = text.strip()
    if len(stripped) <= 75 and "\n" not in stripped and not stripped.endswith("."):
        if NUMBERED_HEADING_REGEX.match(stripped):
            return 2
    return None


def _update_heading_stack(stack: list[tuple[int, str]], level: int, title: str) -> tuple[str, ...]:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title.strip()))
    return tuple(item[1] for item in stack)


class DocxExtractorAdapter(DocumentExtractorPort):
    """Adapter extracting paragraphs, heading hierarchies, and tables from .docx files."""

    @property
    def supported_format(self) -> DocumentFormat:
        return DocumentFormat.DOCX

    def extract(
        self, file_path: Path, document_id: str | None = None
    ) -> ExtractedDocumentContent:
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        doc_id = document_id or file_path.stem
        return self.extract_from_bytes(file_path.read_bytes(), document_id=doc_id)

    def extract_from_bytes(
        self, raw_bytes: bytes, document_id: str
    ) -> ExtractedDocumentContent:
        import io

        content_hash = _sha256_hex(raw_bytes)
        if not zipfile.is_zipfile(io.BytesIO(raw_bytes)):
            raise DomainValidationError(f"Invalid DOCX archive for document '{document_id}'.")

        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise DomainValidationError(
                    f"DOCX archive '{document_id}' is missing word/document.xml."
                )
            xml_bytes = zf.read("word/document.xml")
            has_embedded_media = any(
                name.startswith("word/media/") for name in zf.namelist()
            )

        root = ET.fromstring(xml_bytes)
        body = root.find(f"{WORD_NAMESPACE}body")
        if body is None:
            raise DomainValidationError(f"DOCX '{document_id}' has no document body.")

        heading_stack: list[tuple[int, str]] = []
        current_headings: tuple[str, ...] = ()
        blocks: list[ExtractedBlock] = []
        tables: list[ExtractedTable] = []
        current_page = 1

        for child in body:
            if child.tag == f"{WORD_NAMESPACE}p":
                # Check for explicit page break inside paragraph
                if child.findall(f".//{WORD_NAMESPACE}br[@{WORD_NAMESPACE}type='page']"):
                    current_page += 1

                style_el = child.find(f".//{WORD_NAMESPACE}pStyle")
                style_val = (
                    style_el.attrib.get(f"{WORD_NAMESPACE}val", "")
                    if style_el is not None
                    else ""
                )
                raw_text = "".join(
                    t.text or "" for t in child.findall(f".//{WORD_NAMESPACE}t")
                ).strip()
                if not raw_text:
                    continue

                h_level = _detect_heading_level(style_val, raw_text)
                if h_level is not None:
                    current_headings = _update_heading_stack(heading_stack, h_level, raw_text)
                    blocks.append(
                        ExtractedBlock(
                            text=raw_text,
                            heading_path=current_headings,
                            page_number=current_page,
                            section_reference=current_headings[-1],
                            block_type="heading",
                            heading_level=h_level,
                        )
                    )
                else:
                    blocks.append(
                        ExtractedBlock(
                            text=raw_text,
                            heading_path=current_headings,
                            page_number=current_page,
                            section_reference=current_headings[-1] if current_headings else None,
                            block_type="paragraph",
                        )
                    )

            elif child.tag == f"{WORD_NAMESPACE}tbl":
                rows_list: list[tuple[str, ...]] = []
                for tr in child.findall(f".//{WORD_NAMESPACE}tr"):
                    row_cells: list[str] = []
                    for tc in tr.findall(f"./{WORD_NAMESPACE}tc"):
                        cell_text = " ".join(
                            "".join(t.text or "" for t in p.findall(f".//{WORD_NAMESPACE}t")).strip()
                            for p in tc.findall(f".//{WORD_NAMESPACE}p")
                        ).strip()
                        row_cells.append(cell_text)
                    if any(row_cells):
                        rows_list.append(tuple(row_cells))

                if rows_list:
                    caption = current_headings[-1] if current_headings else None
                    extracted_tbl = ExtractedTable(
                        rows=tuple(rows_list),
                        page_number=current_page,
                        heading_path=current_headings,
                        caption=caption,
                    )
                    tables.append(extracted_tbl)
                    blocks.append(
                        ExtractedBlock(
                            text=extracted_tbl.to_text(),
                            heading_path=current_headings,
                            page_number=current_page,
                            section_reference=caption,
                            block_type="table",
                        )
                    )

        return ExtractedDocumentContent(
            document_id=document_id,
            file_format=DocumentFormat.DOCX,
            blocks=tuple(blocks),
            tables=tuple(tables),
            page_count=current_page,
            requires_lvm_assistance=has_embedded_media,
            flagged_lvm_pages=(1,) if has_embedded_media else (),
            raw_content_hash=content_hash,
        )


class PdfExtractorAdapter(DocumentExtractorPort):
    """Adapter extracting page-indexed blocks from .pdf files and flagging visual/scanned pages for LVM."""

    def __init__(self, min_chars_per_page_before_lvm: int = 40) -> None:
        self._min_chars = min_chars_per_page_before_lvm

    @property
    def supported_format(self) -> DocumentFormat:
        return DocumentFormat.PDF

    def extract(
        self, file_path: Path, document_id: str | None = None
    ) -> ExtractedDocumentContent:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        doc_id = document_id or file_path.stem
        return self.extract_from_bytes(file_path.read_bytes(), document_id=doc_id)

    @staticmethod
    def _parse_pdf_pages_pure_python(raw_bytes: bytes) -> tuple[list[list[str]], set[int]]:
        """Extract text lines per page and identify pages containing images/complex graphics."""
        if not raw_bytes.startswith(b"%PDF"):
            raise DomainValidationError("Invalid PDF header: file does not start with %PDF.")

        objs: dict[int, tuple[bytes, bytes | None]] = {}
        for m in re.finditer(rb"(\d+)\s+0\s+obj\b(.*?)endobj", raw_bytes, re.DOTALL):
            onum = int(m.group(1))
            body = m.group(2)
            sm = re.search(rb"stream\r?\n(.*?)\r?\nendstream", body, re.DOTALL)
            sdata: bytes | None = None
            if sm:
                raw_stream = sm.group(1)
                try:
                    sdata = zlib.decompress(raw_stream)
                except Exception:
                    sdata = raw_stream
            objs[onum] = (body, sdata)

        # 1. Parse ToUnicode CMaps
        cmaps: dict[int, dict[int, str]] = {}
        for onum, (_body, sdata) in objs.items():
            if sdata and (b"beginbfchar" in sdata or b"beginbfrange" in sdata):
                txt = sdata.decode("latin1", errors="ignore")
                cmap: dict[int, str] = {}
                for bf_block in re.findall(r"beginbfchar(.*?)endbfchar", txt, re.DOTALL):
                    for fm in re.finditer(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", bf_block):
                        code = int(fm.group(1), 16)
                        vh = fm.group(2)
                        cmap[code] = "".join(
                            chr(int(vh[i : i + 4], 16))
                            for i in range(0, len(vh), 4)
                            if len(vh[i : i + 4]) == 4
                        )
                for bfr_block in re.findall(r"beginbfrange(.*?)endbfrange", txt, re.DOTALL):
                    for rm in re.finditer(
                        r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", bfr_block
                    ):
                        start_c, end_c, dst_c = (
                            int(rm.group(1), 16),
                            int(rm.group(2), 16),
                            int(rm.group(3), 16),
                        )
                        for c in range(start_c, end_c + 1):
                            cmap[c] = chr(dst_c + (c - start_c))
                cmaps[onum] = cmap

        # 2. Map font object -> CMap and font alias (/Fxx) -> CMap
        font_cmaps: dict[int, dict[int, str]] = {}
        for onum, (body, _) in objs.items():
            tm = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", body)
            if tm and int(tm.group(1)) in cmaps:
                font_cmaps[onum] = cmaps[int(tm.group(1))]

        fname_cmaps: dict[str, dict[int, str]] = {}
        for _onum, (body, _) in objs.items():
            for fm in re.finditer(rb"/(F\d+)\s+(\d+)\s+0\s+R", body):
                fname = fm.group(1).decode("ascii")
                fobj = int(fm.group(2))
                if fobj in font_cmaps:
                    fname_cmaps[fname] = font_cmaps[fobj]

        # 3. Extract each page's content stream
        page_entries: list[tuple[bytes, bytes | None]] = []
        for _onum, (body, _) in sorted(objs.items()):
            if re.search(rb"/Type\s*/Page\b", body):
                cm = re.search(rb"/Contents\s+(\d+)\s+0\s+R", body)
                c_stream = objs.get(int(cm.group(1)), (b"", None))[1] if cm else None
                page_entries.append((body, c_stream))

        if not page_entries:
            # Fallback if PDF uses inline stream without standard /Type /Page object table
            page_entries = [(raw_bytes, s) for _, s in objs.values() if s] or [(raw_bytes, raw_bytes)]

        pages_lines: list[list[str]] = []
        visual_pages: set[int] = set()

        for page_idx, (p_body, p_stream) in enumerate(page_entries, start=1):
            has_image = (
                b"/Image" in p_body
                or b"/XObject" in p_body
                or (p_stream is not None and b" Do" in p_stream)
            )
            if has_image:
                visual_pages.add(page_idx)

            if not p_stream:
                pages_lines.append([])
                visual_pages.add(page_idx)
                continue

            stream_str = p_stream.decode("latin1", errors="ignore")
            current_font: str | None = None
            lines: list[str] = []

            for bt in re.findall(r"BT(.*?)ET", stream_str, re.DOTALL):
                chars: list[str] = []
                in_actual: str | None = None
                in_rev = "/ReversedChars" in bt

                for tok in re.finditer(
                    r"/(F\d+)\s+[\d\.]+\s+Tf"
                    r"|/ActualText\s*<FEFF([0-9A-Fa-f]+)>"
                    r"|/ActualText\s*\(((?:\\.|[^()])*)\)"
                    r"|<([0-9A-Fa-f]+)>\s*Tj"
                    r"|\(([^()]*)\)\s*Tj",
                    bt,
                ):
                    if tok.group(1):
                        current_font = tok.group(1)
                    elif tok.group(2):
                        vh = tok.group(2)
                        in_actual = "".join(
                            chr(int(vh[i : i + 4], 16))
                            for i in range(0, len(vh), 4)
                            if len(vh[i : i + 4]) == 4
                        )
                    elif tok.group(3) is not None:
                        in_actual = (
                            tok.group(3)
                            .replace("\\)", ")")
                            .replace("\\(", "(")
                            .replace("\\\\", "\\")
                        )
                    elif tok.group(4):
                        if in_actual is not None:
                            chars.append(in_actual)
                            in_actual = None
                        else:
                            hx = tok.group(4)
                            cmap = fname_cmaps.get(current_font or "", {})
                            for i in range(0, len(hx), 2):
                                code = int(hx[i : i + 2], 16)
                                chars.append(cmap.get(code, ""))
                    elif tok.group(5) is not None:
                        lit = tok.group(5)
                        chars.append(lit)

                line_text = "".join(reversed(chars)) if in_rev else "".join(chars)
                cleaned_line = normalize_persian_text(line_text)
                if cleaned_line:
                    lines.append(cleaned_line)

            pages_lines.append(lines)

        return pages_lines, visual_pages

    def extract_from_bytes(
        self, raw_bytes: bytes, document_id: str
    ) -> ExtractedDocumentContent:
        content_hash = _sha256_hex(raw_bytes)
        pages_lines, visual_pages = self._parse_pdf_pages_pure_python(raw_bytes)

        blocks: list[ExtractedBlock] = []
        heading_stack: list[tuple[int, str]] = []
        current_headings: tuple[str, ...] = ()
        flagged_pages: set[int] = set(visual_pages)

        for page_num, lines in enumerate(pages_lines, start=1):
            page_text_len = sum(len(ln) for ln in lines)
            if page_text_len < self._min_chars:
                flagged_pages.add(page_num)

            if not lines:
                continue

            # First line on page 1 is treated as document title heading if short
            paragraph_buf: list[str] = []
            for idx, ln in enumerate(lines):
                is_first_title = page_num == 1 and idx == 0 and len(ln) <= 90
                h_level = 1 if is_first_title else _detect_heading_level("", ln)
                if h_level is not None:
                    if paragraph_buf:
                        blocks.append(
                            ExtractedBlock(
                                text=" ".join(paragraph_buf),
                                heading_path=current_headings,
                                page_number=page_num,
                                section_reference=current_headings[-1] if current_headings else f"page-{page_num}",
                                block_type="paragraph",
                            )
                        )
                        paragraph_buf = []
                    current_headings = _update_heading_stack(heading_stack, h_level, ln)
                    blocks.append(
                        ExtractedBlock(
                            text=ln,
                            heading_path=current_headings,
                            page_number=page_num,
                            section_reference=current_headings[-1],
                            block_type="heading",
                            heading_level=h_level,
                        )
                    )
                else:
                    paragraph_buf.append(ln)

            if paragraph_buf:
                blocks.append(
                    ExtractedBlock(
                        text=" ".join(paragraph_buf),
                        heading_path=current_headings,
                        page_number=page_num,
                        section_reference=current_headings[-1] if current_headings else f"page-{page_num}",
                        block_type="paragraph",
                    )
                )

        sorted_flagged = tuple(sorted(flagged_pages))
        return ExtractedDocumentContent(
            document_id=document_id,
            file_format=DocumentFormat.PDF,
            blocks=tuple(blocks),
            tables=(),
            page_count=max(len(pages_lines), 1),
            requires_lvm_assistance=len(sorted_flagged) > 0,
            flagged_lvm_pages=sorted_flagged,
            raw_content_hash=content_hash,
        )


class DocExtractorAdapter(DocumentExtractorPort):
    """Adapter extracting text and structure from legacy .doc files via multiple offline strategies."""

    @property
    def supported_format(self) -> DocumentFormat:
        return DocumentFormat.DOC

    def extract(
        self, file_path: Path, document_id: str | None = None
    ) -> ExtractedDocumentContent:
        if not file_path.exists():
            raise FileNotFoundError(f"DOC file not found: {file_path}")
        doc_id = document_id or file_path.stem
        return self.extract_from_bytes(file_path.read_bytes(), document_id=doc_id)

    @staticmethod
    def _extract_paragraphs_from_ole2_bytes(raw_bytes: bytes) -> list[str]:
        """Strategy 1: Extract clean UTF-16LE and UTF-8/CP1256 text streams from OLE2 binary .doc."""
        text16 = raw_bytes.decode("utf-16le", errors="ignore")
        # Extract contiguous readable Persian/English runs (including \r paragraph separators)
        runs = re.findall(
            r"[\u0600-\u06FFa-zA-Z0-9\s\.,:;\-\(\)\[\]/«»٪\u200c\r\n—–_@#]{8,}",
            text16,
        )
        paragraphs: list[str] = []
        for run in runs:
            for raw_para in re.split(r"[\r\n]+", run):
                cleaned = normalize_persian_text(raw_para)
                if len(cleaned) < 4:
                    continue
                low = cleaned.lower()
                if low in DOC_STYLE_NOISE:
                    continue
                if any(low.startswith(prefix) for prefix in ("normal ", "heading ", "toc ", "hyperlink")):
                    continue
                # Require at least one Persian letter or >= 3 Latin words to skip binary artifacts
                has_persian = bool(re.search(r"[\u0600-\u06FF]{2,}", cleaned))
                has_meaningful_latin = len(re.findall(r"[A-Za-z]{3,}", cleaned)) >= 2
                if has_persian or has_meaningful_latin:
                    if cleaned not in paragraphs:
                        paragraphs.append(cleaned)
        return paragraphs

    @staticmethod
    def _extract_paragraphs_from_text_stream(raw_bytes: bytes) -> list[str]:
        """Strategy 2: Decode plain UTF-8, UTF-16, or CP1256 text stream .doc files."""
        for enc in ("utf-8", "utf-16", "cp1256"):
            try:
                decoded = raw_bytes.decode(enc)
                paras = [
                    normalize_persian_text(p)
                    for p in re.split(r"[\r\n]+", decoded)
                    if normalize_persian_text(p)
                ]
                if paras:
                    return paras
            except Exception:
                continue
        return []

    def extract_from_bytes(
        self, raw_bytes: bytes, document_id: str
    ) -> ExtractedDocumentContent:
        if not raw_bytes:
            raise DomainValidationError(f"DOC file '{document_id}' is empty.")

        content_hash = _sha256_hex(raw_bytes)
        if raw_bytes.startswith(OLE2_MAGIC):
            paragraphs = self._extract_paragraphs_from_ole2_bytes(raw_bytes)
        else:
            paragraphs = self._extract_paragraphs_from_text_stream(raw_bytes)
            if not paragraphs:
                paragraphs = self._extract_paragraphs_from_ole2_bytes(raw_bytes)

        if not paragraphs:
            raise DomainValidationError(
                f"Could not extract readable text from DOC file '{document_id}'."
            )

        heading_stack: list[tuple[int, str]] = []
        current_headings: tuple[str, ...] = ()
        blocks: list[ExtractedBlock] = []

        for idx, para in enumerate(paragraphs):
            is_title = idx == 0 and len(para) <= 85
            is_short_header = (
                len(para) <= 55
                and not para.endswith(".")
                and ("(" in para or NUMBERED_HEADING_REGEX.match(para) is not None)
            )
            if is_title or is_short_header:
                level = 1 if is_title else 2
                current_headings = _update_heading_stack(heading_stack, level, para)
                blocks.append(
                    ExtractedBlock(
                        text=para,
                        heading_path=current_headings,
                        page_number=1,
                        section_reference=current_headings[-1],
                        block_type="heading",
                        heading_level=level,
                    )
                )
            else:
                blocks.append(
                    ExtractedBlock(
                        text=para,
                        heading_path=current_headings,
                        page_number=1,
                        section_reference=current_headings[-1] if current_headings else None,
                        block_type="paragraph",
                    )
                )

        return ExtractedDocumentContent(
            document_id=document_id,
            file_format=DocumentFormat.DOC,
            blocks=tuple(blocks),
            tables=(),
            page_count=1,
            requires_lvm_assistance=False,
            flagged_lvm_pages=(),
            raw_content_hash=content_hash,
        )


class MockLvmLayoutAdapter(LvmLayoutAssistantPort):
    """Offline Mock/Fallback Adapter for LVM visual and layout interpretation (Section 7.3)."""

    def __init__(self) -> None:
        self.processed_requests: list[dict[str, Any]] = []

    def describe_visual_page(
        self, document_id: str, version_id: str, page_number: int
    ) -> str:
        self.processed_requests.append(
            {
                "document_id": document_id,
                "version_id": version_id,
                "page_number": page_number,
            }
        )
        return (
            f"[تفسیر بصری LVM - سند {document_id} نسخه {version_id} صفحه {page_number}]: "
            f"چیدمان صفحه شامل جدول مقایسه‌ای شاخص‌ها و نمودار ساختاری است که به متن نسخه متصل شد."
        )

    def enrich_extracted_content(
        self, extracted: ExtractedDocumentContent, version_id: str = "v1"
    ) -> ExtractedDocumentContent:
        if not extracted.requires_lvm_assistance:
            return extracted

        target_pages = extracted.flagged_lvm_pages or (1,)
        lvm_blocks: list[ExtractedBlock] = []
        lvm_notes: list[str] = []

        for page_num in target_pages:
            desc = self.describe_visual_page(
                document_id=extracted.document_id,
                version_id=version_id,
                page_number=page_num,
            )
            lvm_notes.append(desc)
            lvm_blocks.append(
                ExtractedBlock(
                    text=desc,
                    heading_path=("تفسیر چیدمان بصری LVM",),
                    page_number=page_num,
                    section_reference=f"lvm-page-{page_num}",
                    block_type="lvm_layout",
                )
            )

        return extracted.with_lvm_enrichment(lvm_blocks=lvm_blocks, lvm_notes=lvm_notes)


class DocumentExtractorRegistry:
    """Strategy registry selecting the appropriate DocumentExtractorPort by file extension or format."""

    def __init__(
        self,
        extractors: Mapping[DocumentFormat, DocumentExtractorPort] | None = None,
    ) -> None:
        default_map: dict[DocumentFormat, DocumentExtractorPort] = {
            DocumentFormat.DOC: DocExtractorAdapter(),
            DocumentFormat.DOCX: DocxExtractorAdapter(),
            DocumentFormat.PDF: PdfExtractorAdapter(),
        }
        if extractors:
            default_map.update(extractors)
        self._extractors = default_map

    def get_extractor(self, format_or_suffix: DocumentFormat | str) -> DocumentExtractorPort:
        if isinstance(format_or_suffix, DocumentFormat):
            fmt = format_or_suffix
        else:
            cleaned = str(format_or_suffix).strip().upper().lstrip(".")
            try:
                fmt = DocumentFormat(cleaned)
            except ValueError as exc:
                raise DomainValidationError(
                    f"Unsupported document format or extension: '{format_or_suffix}'."
                ) from exc

        extractor = self._extractors.get(fmt)
        if extractor is None:
            raise DomainValidationError(f"No DocumentExtractorPort registered for {fmt.value}.")
        return extractor


__all__ = [
    "DocExtractorAdapter",
    "DocumentExtractorPort",
    "DocumentExtractorRegistry",
    "DocxExtractorAdapter",
    "LvmLayoutAssistantPort",
    "MockLvmLayoutAdapter",
    "PdfExtractorAdapter",
]
