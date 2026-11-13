"""Two chunking strategies over the same normalized pages.

* fixed_size  – strategy A: character windows inside one page (never crosses a page).
* structure   – strategies B/C: legal structure Điều → khoản → điểm → câu, may span pages;
                every chunk is prefixed with a context header [văn bản · Chương · Điều].

Both attach the same metadata fields so that only the chunking differs between A and B.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

from .ingestion import Article, concat_pages, page_at, parse_articles
from .models import Chunk, DocumentPage


@dataclass
class DocContext:
    document_id: str
    source: str
    pages: list[DocumentPage]
    text: str
    spans: list[tuple[int, int, int]]     # (start, end, page) in `text`
    articles: list[Article]

    @classmethod
    def build(cls, pages: list[DocumentPage]) -> "DocContext":
        text, spans = concat_pages(pages)
        return cls(pages[0].document_id, pages[0].source, pages, text, spans, parse_articles(text))

    def article_at(self, offset: int) -> Article:
        for art in self.articles:
            if art.start <= offset < art.end:
                return art
        return self.articles[-1]

    def page_meta(self, page: int) -> dict:
        for p in self.pages:
            if p.page == page:
                return p.metadata
        raise KeyError(page)


def _chunk_id(tag: str, doc: str, page: int, start: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{tag}:{doc}:p{page}:o{start}:{digest}"


def _metadata(ctx: DocContext, start: int, end: int, chunking: str) -> tuple[int, int, dict]:
    page = page_at(ctx.spans, start)
    page_end = page_at(ctx.spans, max(start, end - 1))
    art = ctx.article_at(start)
    base = dict(ctx.page_meta(page))
    base.pop("gazette_page", None)
    base.update({
        "chunking": chunking,
        "chapter": art.chapter or "",
        "article": art.number,                       # 0 = phần mở đầu (trước Điều 1)
        "pages": ",".join(str(p) for p in range(page, page_end + 1)),
        "char_start": start,
        "char_end": end,
    })
    return page, page_end, base


# --------------------------------------------------------------------------- strategy A
def fixed_size(ctx: DocContext, size: int, overlap: int) -> list[Chunk]:
    """Overlapping windows of `size` characters, restarted on every page."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    tag = f"fixed{size}o{overlap}"
    chunks: list[Chunk] = []
    for (p_start, p_end, _page) in ctx.spans:
        start = p_start
        while start < p_end:
            end = min(start + size, p_end)
            body = ctx.text[start:end]
            if body.strip():
                page, page_end, meta = _metadata(ctx, start, end, tag)
                chunks.append(Chunk(_chunk_id(tag, ctx.document_id, page, start, body),
                                    ctx.document_id, ctx.source, page, body, meta, page_end))
            if end >= p_end:
                break
            start = end - overlap
    return chunks


# --------------------------------------------------------------------------- strategy B/C
Finder = Callable[[str, int, int], list[int]]


def _line_start_or_after_punct(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" ")
    return not before or before[-1] in "\n.:;"


def _khoan_finder(text: str, s: int, e: int) -> list[int]:
    """Offsets of sequential clause markers `1. `, `2. ` ... inside an article body."""
    head = re.compile(r"Điều\s+\d+\.\s").match(text, s)
    pos = head.end() if head else s
    out, expected = [], 1
    pat = re.compile(r"(\d{1,2})\.\s")
    while True:
        found = None
        for m in pat.finditer(text, pos, e):
            if int(m.group(1)) == expected and _line_start_or_after_punct(text, m.start()):
                found = m
                break
        if not found:
            return out
        out.append(found.start())
        expected, pos = expected + 1, found.end()


def _diem_finder(text: str, s: int, e: int) -> list[int]:
    """Point markers `a) `, `b) `, `đ) ` at a line start or after punctuation."""
    pat = re.compile(r"(?<!\w)[a-zđ]\)\s")
    return [m.start() for m in pat.finditer(text, s, e)
            if m.start() > s and _line_start_or_after_punct(text, m.start())]


def _sentence_finder(text: str, s: int, e: int) -> list[int]:
    return [s + m.end() for m in re.finditer(r"(?:[.;:]\s+|\n)", text[s:e]) if s + m.end() < e]


def _split(text: str, s: int, e: int, budget: int, finders: list[Finder]) -> list[tuple[int, int]]:
    if e - s <= budget:
        return [(s, e)]
    if not finders:                                   # hard cut at the last space before budget
        out, cur = [], s
        while e - cur > budget:
            cut = text.rfind(" ", cur + budget // 2, cur + budget)
            cut = cut if cut > cur else cur + budget
            out.append((cur, cut))
            cur = cut
        out.append((cur, e))
        return out
    bounds = [b for b in finders[0](text, s, e) if s < b < e]
    if not bounds:
        return _split(text, s, e, budget, finders[1:])
    edges = [s] + bounds + [e]
    pieces: list[tuple[int, int]] = []
    for a, b in zip(edges, edges[1:]):
        pieces.extend(_split(text, a, b, budget, finders[1:]))
    return _pack(pieces, budget)


def _pack(spans: list[tuple[int, int]], budget: int) -> list[tuple[int, int]]:
    """Greedily merge adjacent spans while the merged span fits the budget."""
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if out and b - out[-1][0] <= budget:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def by_structure(ctx: DocContext, max_size: int) -> list[Chunk]:
    """Article-aware chunks: whole Điều if it fits, else packed khoản / điểm / câu."""
    tag = f"struct{max_size}"
    short = ctx.pages[0].metadata["short_name"]
    chunks: list[Chunk] = []
    for art in ctx.articles:
        label = f"Điều {art.number}" if art.number else "Phần mở đầu"
        header = "[" + " · ".join(x for x in (short, art.chapter, label) if x) + "]"
        budget = max_size - len(header) - 1
        for s, e in _split(ctx.text, art.start, art.end, budget,
                           [_khoan_finder, _diem_finder, _sentence_finder]):
            while s < e and ctx.text[s].isspace():
                s += 1
            while e > s and ctx.text[e - 1].isspace():
                e -= 1
            if s >= e:
                continue
            body = ctx.text[s:e]
            text = f"{header}\n{body}"
            page, page_end, meta = _metadata(ctx, s, e, tag)
            meta["article"] = art.number
            meta["chapter"] = art.chapter or ""
            chunks.append(Chunk(_chunk_id(tag, ctx.document_id, page, s, text),
                                ctx.document_id, ctx.source, page, text, meta, page_end))
    return chunks
