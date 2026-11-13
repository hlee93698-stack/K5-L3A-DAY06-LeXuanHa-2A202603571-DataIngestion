"""Read and normalize source PDFs page by page, keeping page-level traceability.

Every physical page is kept (with its 1-based number); pages that are empty,
too short or excluded by the manifest are logged, never silently dropped.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

# Running header printed on every Công báo page, e.g. "CÔNG BÁO/Số 865 + 866/Ngày 30-7-2023"
GAZETTE_HEADER = re.compile(r"CÔNG BÁO\s*/\s*Số\s*[\d\s+]+?\s*(?:/\s*)?[Nn]gày\s*\d{1,2}\s*-\s*\d{1,2}\s*-\s*\d{4}")
# Digital-signature stamp added by the government portal on the first page
SIGNATURE_STAMP = re.compile(r"^(?:Người ký|Ký bởi):.*?(?:\+07:00|$)", re.MULTILINE)
PAGE_NUMBER_LINE = re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE)
UNDERSCORE_RULE = re.compile(r"_{1,}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """NFC-normalize, drop gazette headers/page numbers/signature stamp, tidy whitespace.

    Line breaks are kept (they carry paragraph structure); runs of spaces and
    blank lines are collapsed. Nothing inside the legal text itself is rewritten.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "").replace("\r", "\n")
    text = SIGNATURE_STAMP.sub("", text)
    text = GAZETTE_HEADER.sub("\n", text)
    text = PAGE_NUMBER_LINE.sub("", text)
    text = UNDERSCORE_RULE.sub("", text)          # decorative rules under the national motto
    text = re.sub(r"[ \t\f\v]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def gazette_page_number(raw: str) -> int | None:
    """Printed Công báo page number next to the running header (or on its own line)."""
    raw = unicodedata.normalize("NFC", raw)
    for pat in (GAZETTE_HEADER.pattern + r"[ \t]+(\d{1,4})[ \t]*$",
                r"^[ \t]*(\d{1,4})[ \t]+" + GAZETTE_HEADER.pattern):
        m = re.search(pat, raw, re.MULTILINE)
        if m:
            return int(m.group(1))
    m = PAGE_NUMBER_LINE.search(raw)
    return int(m.group(1)) if m else None


@dataclass
class PageLog:
    document_id: str
    page: int
    raw_chars: int
    normalized_chars: int
    status: str            # content | short | empty | excluded
    note: str = ""


def load_pdf(path: Path, document_id: str, source: str, doc_meta: dict[str, Any],
             min_content_chars: int, exclude_from: int | None = None,
             exclude_reason: str = "") -> tuple[list[DocumentPage], list[PageLog]]:
    """Extract every physical page; return content pages plus a log for all pages.

    OCR is not used: all four sources have a text layer. Pages without text are
    logged as `empty` so they can be inspected (none were expected).

    pypdf's default ("plain") mode inserts spurious spaces inside Vietnamese words
    for these Công báo PDFs ("T ổ ch ức"); "layout" mode extracts them correctly,
    so it is used here (verified on sample pages, see report).
    """
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    logs: list[PageLog] = []
    for idx, pdf_page in enumerate(reader.pages, start=1):
        try:
            raw = pdf_page.extract_text(extraction_mode="layout") or ""
        except Exception as exc:  # record and continue; never hide the failure
            logs.append(PageLog(document_id, idx, 0, 0, "empty", f"extract_error: {exc!r}"))
            continue
        norm = normalize_text(raw)
        if exclude_from is not None and idx >= exclude_from:
            logs.append(PageLog(document_id, idx, len(raw), len(norm), "excluded", exclude_reason))
            continue
        if not norm:
            logs.append(PageLog(document_id, idx, len(raw), 0, "empty", "no extractable text"))
            continue
        if len(norm) < min_content_chars:
            logs.append(PageLog(document_id, idx, len(raw), len(norm), "short",
                                f"< {min_content_chars} chars; not counted as content page"))
            # Short pages (e.g. signature-only) are still indexed so nothing is lost.
        else:
            logs.append(PageLog(document_id, idx, len(raw), len(norm), "content"))
        meta = {
            "document_type": doc_meta["document_type"],
            "legal_regime": doc_meta["legal_regime"],
            "year": doc_meta["year"],
            "document_number": doc_meta["document_number"],
            "short_name": doc_meta["short_name"],
            "gazette_page": gazette_page_number(raw),
        }
        pages.append(DocumentPage(document_id, source, idx, norm, meta))
    return pages, logs


# --------------------------------------------------------------------------- structure
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


@dataclass
class Article:
    number: int            # 0 = preamble before Điều 1
    chapter: str           # e.g. "Chương II" ("" before Chương I)
    start: int             # offset in the concatenated document text
    end: int


def concat_pages(pages: list[DocumentPage]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join pages with newlines; return text and (start, end, page) spans."""
    parts, spans, pos = [], [], 0
    for p in pages:
        spans.append((pos, pos + len(p.text), p.page))
        parts.append(p.text)
        pos += len(p.text) + 1
    return "\n".join(parts), spans


def page_at(spans: list[tuple[int, int, int]], offset: int) -> int:
    for start, end, page in spans:
        if offset <= end:
            return page
    return spans[-1][2]


def parse_articles(text: str) -> list[Article]:
    """Find `Điều N.` headings in strict sequence (N = previous + 1).

    Requiring the next expected number avoids false splits on cross-references
    such as "... quy định tại Điều 80." inside another article.
    """
    heads: list[tuple[int, int]] = []          # (number, offset)
    expected, pos = 1, 0
    pattern = re.compile(r"Điều\s+(\d+)\.\s")
    while True:
        m = None
        for cand in pattern.finditer(text, pos):
            if int(cand.group(1)) == expected:
                m = cand
                break
        if not m:
            break
        heads.append((expected, m.start()))
        expected, pos = expected + 1, m.end()

    chapters: list[tuple[str, int]] = []
    exp_ch, pos = 0, 0
    while exp_ch < len(ROMAN):
        m = re.compile(rf"^Chương\s+{ROMAN[exp_ch]}\b", re.MULTILINE).search(text, pos)
        if not m:
            break
        chapters.append((f"Chương {ROMAN[exp_ch]}", m.start()))
        exp_ch, pos = exp_ch + 1, m.end()
    # Chapter / section (Mục) headings are attached to the article that follows them.
    heading_offsets = sorted({off for _, off in chapters} |
                             {m.start() for m in re.finditer(r"^Mục\s+\d+\b", text, re.MULTILINE)})

    def chapter_at(offset: int) -> str:
        name = ""
        for ch, off in chapters:
            if off <= offset:
                name = ch
        return name

    starts: list[int] = []
    prev = 0
    for num, off in heads:
        between = [h for h in heading_offsets if prev < h < off]
        starts.append(between[0] if between else off)
        prev = off

    arts: list[Article] = []
    first = starts[0] if starts else len(text)
    if first > 0:
        arts.append(Article(0, "", 0, first))
    for i, (num, off) in enumerate(heads):
        end = starts[i + 1] if i + 1 < len(heads) else len(text)
        arts.append(Article(num, chapter_at(off), starts[i], end))
    return arts
