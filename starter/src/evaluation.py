"""Automatic retrieval scoring (Hit@1, Evidence Hit@3) with full per-query traces.

A retrieved chunk is a *hit* when, for at least one documented gold entry,
  * chunk.document_id == gold.source_id,
  * gold.page lies in [chunk.page, chunk.page_end], and
  * every evidence fragment of that gold entry occurs in the chunk text
    (after NFC + lower-case + whitespace normalization).
Coherence and grounding are NOT inferred here; they are labelled manually.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def fragments(gold: dict[str, Any]) -> list[str]:
    q = gold["evidence_quote"]
    return [q] if isinstance(q, str) else list(q)


def chunk_hits_gold(hit: dict[str, Any], gold: dict[str, Any]) -> bool:
    m = hit["metadata"]
    if m["document_id"] != gold["source_id"]:
        return False
    if not (int(m["page"]) <= int(gold["page"]) <= int(m["page_end"])):
        return False
    text = norm(hit["text"])
    return all(norm(f) in text for f in fragments(gold))


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    hit_flags = [any(chunk_hits_gold(h, g) for g in question["gold"]) for h in hits]
    # Evidence split: no single chunk has all fragments, but the top-k together do.
    joined = norm(" ".join(h["text"] for h in hits))
    split_evidence = (not any(hit_flags)) and any(
        all(norm(f) in joined for f in fragments(g)) for g in question["gold"])
    source_page_match = [any(h["metadata"]["document_id"] == g["source_id"] and
                             int(h["metadata"]["page"]) <= int(g["page"]) <= int(h["metadata"]["page_end"])
                             for g in question["gold"]) for h in hits]
    flt = question.get("filter")
    filter_ok = None
    if flt:
        filter_ok = all(all(str(h["metadata"].get(k)) == str(v) for k, v in flt.items()) for h in hits)
    return {
        "hit_at_rank": hit_flags,
        "hit1": bool(hit_flags[:1] and hit_flags[0]),
        "hit3": any(hit_flags[:3]),
        "first_hit_rank": (hit_flags.index(True) + 1) if any(hit_flags) else None,
        "source_page_match_at_rank": source_page_match,
        "evidence_split_across_chunks": split_evidence,
        "filter_compliant_top3": filter_ok,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by (strategy, split); report numerators and denominators."""
    out: dict[str, Any] = {}
    for t in traces:
        key = f"{t['strategy']}|{t['split']}"
        s = out.setdefault(key, {"strategy": t["strategy"], "split": t["split"], "N": 0, "hit1": 0,
                                 "hit3": 0, "filter_N": 0, "filter_ok": 0, "split_evidence": 0,
                                 "filter_q_hit3": 0, "nofilter_q_hit3": 0, "nofilter_N": 0})
        r = t["result"]
        s["N"] += 1
        s["hit1"] += int(r["hit1"])
        s["hit3"] += int(r["hit3"])
        s["split_evidence"] += int(r["evidence_split_across_chunks"])
        if t.get("filter"):
            s["filter_N"] += 1
            s["filter_ok"] += int(bool(r["filter_compliant_top3"]))
            s["filter_q_hit3"] += int(r["hit3"])
        else:
            s["nofilter_N"] += 1
            s["nofilter_q_hit3"] += int(r["hit3"])
    for s in out.values():
        s["hit1_rate"] = round(s["hit1"] / s["N"], 3) if s["N"] else None
        s["hit3_rate"] = round(s["hit3"] / s["N"], 3) if s["N"] else None
    return out
