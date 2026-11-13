"""Manual-audit helpers.

    python -m src.audit sample     # build review sheets (coherence / metadata / grounding)
    python -m src.audit summarize  # aggregate the labels written in results/audit/labels.json

The sampling rules are fixed in code before labels are read:
  * coherence: content pages in manifest order, take index 5, 16, 27, ... (every 11th → 10
    positions); the position is the middle character of that page; for each strategy the
    chunk covering that position is labelled 0/1/2.
  * metadata: every top-3 record of every test question for each strategy.
  * grounding: every test question for each strategy (top-3 together), 0/1/2.
Labels themselves are never computed here; only mechanical checks (is the chunk text really
on the stated page of the stated PDF?) are automated and shown next to the label.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from . import config as C
from .chunking import DocContext
from .evaluation import norm
from .run import load_corpus, read_jsonl, write_json

AUDIT = C.RESULTS_DIR / "audit"


def _body(text: str) -> str:
    """Strip the structure header line ([...]) that B/C prepend."""
    if text.startswith("[") and "]\n" in text[:200]:
        return text.split("]\n", 1)[1]
    return text


def sample() -> None:
    sources, docs, _ = load_corpus()
    regime = {d["source_id"]: d["legal_regime"] for d in sources["documents"]}
    ctxs = {d: DocContext.build(p) for d, p in docs.items()}
    chunks = {s: read_jsonl(C.RESULTS_DIR / "chunks" / f"{s}.jsonl") for s in ("A", "B")}
    chunks["C"] = chunks["B"]
    AUDIT.mkdir(parents=True, exist_ok=True)

    # ---------------- coherence positions
    manifest = json.loads(C.MANIFEST_FILE.read_text(encoding="utf-8"))
    ordered = [(d["source_id"], p) for d in manifest["documents"] for p in d["content_pages_counted"]]
    picks = [ordered[i] for i in range(5, len(ordered), 11)][:10]
    rows = []
    for k, (doc, page) in enumerate(picks, 1):
        ctx = ctxs[doc]
        start, end = next((s, e) for s, e, p in ctx.spans if p == page)
        off = (start + end) // 2
        for strat in ("A", "B", "C"):
            cand = [c for c in chunks[strat] if c["document_id"] == doc and
                    c["metadata"]["char_start"] <= off < c["metadata"]["char_end"]]
            c = cand[0]
            rows.append({"position": f"P{k:02d}", "document_id": doc, "page": page, "offset": off,
                         "context": ctx.text[max(start, off - 60):off + 60].replace("\n", " "),
                         "strategy": strat, "chunk_id": c["chunk_id"], "chunk_chars": len(c["text"]),
                         "chunk_text": c["text"]})
    with (AUDIT / "coherence_sample.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- metadata (test × top-3) with mechanical source/page check
    page_text = {(d, p.page): norm(p.text) for d, ps in docs.items() for p in ps}
    mrows = []
    for strat in ("A", "B", "C"):
        for t in read_jsonl(C.RESULTS_DIR / "traces" / f"{strat}_test.jsonl"):
            for h in t["hits"]:
                m = h["metadata"]
                span = " ".join(page_text.get((m["document_id"], p), "")
                                for p in range(int(m["page"]), int(m["page_end"]) + 1))
                body = norm(_body(h["text"]))
                # the chunk body must be found on the stated page range of the stated document
                on_page = body in span
                mrows.append({
                    "strategy": strat, "question": t["id"], "rank": h["rank"], "chunk_id": h["chunk_id"],
                    "document_id": m["document_id"], "source": m["source"], "page": m["page"],
                    "page_end": m["page_end"], "legal_regime": m["legal_regime"],
                    "document_type": m["document_type"], "article": m["article"],
                    "auto_source_file_exists": (C.REPO / m["source"]).exists(),
                    "auto_text_on_stated_pages": on_page,
                    "auto_regime_matches_manifest": m["legal_regime"] == regime[m["document_id"]],
                    "question_filter": json.dumps(t["filter"], ensure_ascii=False) if t["filter"] else "",
                    "filter_respected": (all(str(m.get(k)) == str(v) for k, v in t["filter"].items())
                                         if t["filter"] else ""),
                })
    with (AUDIT / "metadata_check.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrows[0]))
        w.writeheader()
        w.writerows(mrows)

    # ---------------- grounding review sheet
    lines = ["# Grounding review sheet (test split, top-3 per strategy)", ""]
    for strat in ("A", "B", "C"):
        for t in read_jsonl(C.RESULTS_DIR / "traces" / f"{strat}_test.jsonl"):
            lines += [f"## {strat} · {t['id']} — {t['question']}",
                      f"*Đáp án kỳ vọng:* {t['expected_answer']}  ",
                      f"*Filter áp dụng:* {t['filter_applied']} · *Evidence Hit@3 (auto):* {t['result']['hit3']}", ""]
            for h in t["hits"]:
                lines += [f"**#{h['rank']}** `{h['document_id']}` p{h['page']}-{h['page_end']} "
                          f"score={h['score']}", "", "> " + h["text"].replace("\n", " "), ""]
    (AUDIT / "grounding_review.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", [p.name for p in AUDIT.iterdir()])


def summarize() -> None:
    labels = json.loads((AUDIT / "labels.json").read_text(encoding="utf-8"))
    out = {"labeller": labels.get("labeller"), "strategies": {}}
    for strat in ("A", "B", "C"):
        coh = [x["label"] for x in labels["coherence"] if x["strategy"] == strat]
        gr = [x["label"] for x in labels["grounding"] if x["strategy"] == strat]
        md = [x for x in labels["metadata"] if x["strategy"] == strat]
        fq = {}
        for x in md:
            if x.get("filter_required"):
                fq.setdefault(x["question"], []).append(bool(x["filter_respected"]))
        out["strategies"][strat] = {
            "coherence": {"n": len(coh), "mean": round(sum(coh) / len(coh), 2) if coh else None,
                          "counts": {k: coh.count(k) for k in (0, 1, 2)}},
            "grounding": {"n": len(gr), "mean": round(sum(gr) / len(gr), 2) if gr else None,
                          "counts": {k: gr.count(k) for k in (0, 1, 2)}},
            "metadata_complete_and_correct": f"{sum(bool(x['ok']) for x in md)}/{len(md)}",
            "filter_compliance": (f"{sum(all(v) for v in fq.values())}/{len(fq)}" if fq else "N/A"),
        }
    write_json(AUDIT / "summary.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def review(section: str = "all") -> None:
    """Print every labelled item next to the text it was judged on (for the labeller to re-check)."""
    labels = json.loads((AUDIT / "labels.json").read_text(encoding="utf-8"))
    sep = "-" * 100
    if section in ("all", "coherence"):
        rows = {(r["position"], r["strategy"]): r for r in
                csv.DictReader((AUDIT / "coherence_sample.csv").open(encoding="utf-8"))}
        print("=" * 100 + "\nCOHERENCE (0 = khó hiểu/thiếu chủ thể, 1 = hiểu một phần, 2 = tự hiểu, trọn ý)"
              "\nC dùng đúng chunk của B nên chỉ hiện A và B.")
        for x in labels["coherence"]:
            if x["strategy"] == "C":
                continue
            r = rows[(x["position"], x["strategy"])]
            print(f"{sep}\n[{x['position']} · {x['strategy']}] {x['document_id']} trang {x['page']}  "
                  f"→ NHÃN {x['label']}  ({x['reason']})\n{r['chunk_text']}")
    if section in ("all", "grounding"):
        traces = {(s, t["id"]): t for s in ("A", "B", "C")
                  for t in read_jsonl(C.RESULTS_DIR / "traces" / f"{s}_test.jsonl")}
        print("=" * 100 + "\nGROUNDING (0 = không có bằng chứng, 1 = có nhưng thiếu/phải ghép, 2 = đủ trả lời và truy nguồn)")
        for x in labels["grounding"]:
            t = traces[(x["strategy"], x["question"])]
            print(f"{sep}\n[{x['strategy']} · {x['question']}] {t['question']}\nĐáp án kỳ vọng: {t['expected_answer']}"
                  f"\n→ NHÃN {x['label']}  ({x['reason']})")
            for h in t["hits"]:
                print(f"  #{h['rank']} {h['document_id']} trang {h['page']}-{h['page_end']}: "
                      + h["text"].replace("\n", " "))
    if section in ("all", "metadata"):
        bad = [x for x in labels["metadata"] if not x["ok"]]
        notes = [x for x in labels["metadata"] if x["note"]]
        print("=" * 100 + f"\nMETADATA: {len(labels['metadata']) - len(bad)}/{len(labels['metadata'])} bản ghi đúng "
              "(kiểm tra tự động: text chunk nằm đúng trang của đúng file, chế độ pháp lý khớp manifest). "
              "Chi tiết: results/audit/metadata_check.csv")
        for x in notes:
            print(f"  {x['strategy']} {x['question']} #{x['rank']}: {x['note']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "review":
        review(sys.argv[2] if len(sys.argv) > 2 else "all")
    else:
        {"sample": sample, "summarize": summarize}[cmd]()
