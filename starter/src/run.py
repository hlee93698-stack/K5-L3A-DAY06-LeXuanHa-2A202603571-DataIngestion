"""Day 06 pipeline entry point.

    python -m src.run prepare          # read PDFs, manifest, page log, validate benchmark
    python -m src.run ingest           # chunk A/B, embed, store in Chroma (C mirrors B), reload check
    python -m src.run eval --split dev # top-3 traces + Hit@1/Hit@3 on dev
    python -m src.run select           # apply the pre-registered selection rule on dev only
    python -m src.run eval --split test
    python -m src.run all              # everything above, in this order

Run from the `starter/` directory. Any backend/model failure raises; there is no mock fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from . import config as C
from .chunking import DocContext, by_structure, fixed_size
from .evaluation import evaluate_query, norm, summarize
from .ingestion import load_pdf, sha256_file
from .models import Chunk, DocumentPage


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- corpus
def load_corpus() -> tuple[dict, dict[str, list[DocumentPage]], list]:
    sources = json.loads(C.SOURCES_FILE.read_text(encoding="utf-8"))
    docs: dict[str, list[DocumentPage]] = {}
    logs = []
    for d in sources["documents"]:
        path = C.REPO / d["file"]
        if not path.exists():
            raise FileNotFoundError(f"missing source file {path} (see data/manifest.json for URL)")
        pages, plog = load_pdf(path, d["source_id"], d["file"], d, C.MIN_CONTENT_CHARS,
                               d.get("exclude_pages_from"), d.get("exclude_reason", ""))
        docs[d["source_id"]] = pages
        logs.extend(plog)
    return sources, docs, logs


def cmd_prepare(_args) -> None:
    sources, docs, logs = load_corpus()
    with (C.RESULTS_DIR / "page_log.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["document_id", "page", "raw_chars", "normalized_chars", "status", "note"])
        for l in logs:
            w.writerow([l.document_id, l.page, l.raw_chars, l.normalized_chars, l.status, l.note])

    manifest_docs, total_content = [], 0
    for d in sources["documents"]:
        dl = [l for l in logs if l.document_id == d["source_id"]]
        content = [l for l in dl if l.status == "content"]
        total_content += len(content)
        manifest_docs.append({
            **{k: d[k] for k in ("source_id", "title", "document_number", "document_type", "issuer",
                                 "passed_date", "effective_date", "legal_regime", "year", "snapshot",
                                 "notes")},
            "source_path_or_url": d["file"], "download_url": d["source_url"],
            "landing_url": d["landing_url"],
            "license_or_permission": sources["license_or_permission"],
            "accessed_at": sources["accessed_at"],
            "sha256": sha256_file(C.REPO / d["file"]),
            "physical_pages": len(dl),
            "content_pages_counted": [l.page for l in content],
            "content_page_count": len(content),
            "non_counted_pages": [{"page": l.page, "status": l.status, "note": l.note}
                                  for l in dl if l.status != "content"],
            "normalized_chars_total": sum(l.normalized_chars for l in dl if l.status != "excluded"),
            "chars_per_page": {str(l.page): l.normalized_chars for l in dl},
        })
    manifest = {
        "generated_at": now(),
        "page_counting_rule": (f"1 trang = 1 trang vật lý của PDF (đánh số từ 1). Trang được tính là trang nội dung "
                               f"khi văn bản đã chuẩn hóa ≥ {C.MIN_CONTENT_CHARS} ký tự và không thuộc phụ lục "
                               f"mẫu biểu bị loại trừ. Không có trang bìa, mục lục hay trang trống trong 4 nguồn."),
        "total_content_pages": total_content,
        "documents": manifest_docs,
    }
    write_json(C.MANIFEST_FILE, manifest)
    if total_content < 30:
        raise SystemExit(f"Only {total_content} content pages (< 30)")
    print(f"content pages: {total_content}")
    validate_benchmark(docs)


def validate_benchmark(docs: dict[str, list[DocumentPage]]) -> None:
    qs = read_jsonl(C.BENCHMARK_FILE)
    problems, checks = [], []
    regimes = {p.metadata["legal_regime"] for pages in docs.values() for p in pages}
    for q in qs:
        for g in q["gold"]:
            pages = {p.page: p for p in docs.get(g["source_id"], [])}
            page = pages.get(int(g["page"]))
            frags = [g["evidence_quote"]] if isinstance(g["evidence_quote"], str) else g["evidence_quote"]
            ok = page is not None and all(norm(f) in norm(page.text) for f in frags)
            found_on = [] if ok else [p.page for p in docs.get(g["source_id"], [])
                                      if all(norm(f) in norm(p.text) for f in frags)]
            checks.append({"id": q["id"], "source_id": g["source_id"], "page": g["page"], "ok": ok,
                           "found_on_pages": found_on})
            if not ok:
                problems.append(f"{q['id']}: evidence not on {g['source_id']} p{g['page']} (found on {found_on})")
        if q.get("filter"):
            for k, v in q["filter"].items():
                if k not in C.ALLOWED_FILTER_FIELDS:
                    problems.append(f"{q['id']}: filter field {k} not allowed")
                if k == "legal_regime" and v not in regimes:
                    problems.append(f"{q['id']}: filter value {v} not in metadata")
    n_dev = sum(q["split"] == "dev" for q in qs)
    n_test = sum(q["split"] == "test" for q in qs)
    n_f = sum(bool(q.get("filter")) for q in qs)
    n_f_test = sum(bool(q.get("filter")) and q["split"] == "test" for q in qs)
    summary = {"checked_at": now(), "questions": len(qs), "dev": n_dev, "test": n_test,
               "filter_questions": n_f, "filter_questions_test": n_f_test,
               "all_evidence_verified": not problems, "problems": problems, "checks": checks}
    write_json(C.RESULTS_DIR / "benchmark_validation.json", summary)
    if len(qs) < 20 or n_dev < 15 or n_test < 5 or n_f < 3 or n_f_test < 1:
        problems.append("benchmark size/split/filter requirements not met")
    if problems:
        raise SystemExit("Benchmark validation failed:\n" + "\n".join(problems))
    print(f"benchmark OK: {len(qs)} questions ({n_dev} dev / {n_test} test), {n_f} with filter")


# --------------------------------------------------------------------------- ingest
def build_chunks(docs: dict[str, list[DocumentPage]]) -> dict[str, dict[str, list[Chunk]]]:
    out: dict[str, dict[str, list[Chunk]]] = {"A": {}, "B": {}}
    for doc_id, pages in docs.items():
        ctx = DocContext.build(pages)
        a, b = C.STRATEGIES["A"], C.STRATEGIES["B"]
        out["A"][doc_id] = fixed_size(ctx, a["chunk_size"], a["overlap"])
        out["B"][doc_id] = by_structure(ctx, b["max_size"])
    return out


def length_stats(values: list[int]) -> dict:
    q = statistics.quantiles(values, n=4)
    return {"min": min(values), "p25": round(q[0], 1), "median": statistics.median(values),
            "p75": round(q[2], 1), "max": max(values), "mean": round(statistics.mean(values), 1)}


def model_revision(model_name: str) -> str | None:
    from huggingface_hub.constants import HF_HUB_CACHE
    snap = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}" / "snapshots"
    revs = sorted(p.name for p in snap.iterdir()) if snap.exists() else []
    return revs[-1] if revs else None


def get_embedder():
    from .embeddings import E5Embedder
    return E5Embedder(C.EMBEDDING_MODEL, C.QUERY_PREFIX, C.PASSAGE_PREFIX, C.EMBED_BATCH_SIZE,
                      C.EMBEDDING_REVISION)


def cmd_ingest(_args) -> None:
    from .store import ChromaStore

    _, docs, _ = load_corpus()
    t0 = time.perf_counter()
    chunks = build_chunks(docs)
    chunk_time = time.perf_counter() - t0
    t_load = time.perf_counter()
    emb = get_embedder()
    load_time = time.perf_counter() - t_load
    store = ChromaStore(C.CHROMA_DIR, emb.dimensions, C.COLLECTION_PREFIX)

    stats, reload = {}, {"first_pass": {}, "second_pass": {}}
    for strat in ("A", "B"):
        flat = [c for doc_chunks in chunks[strat].values() for c in doc_chunks]
        write_jsonl(C.RESULTS_DIR / "chunks" / f"{strat}.jsonl",
                    [{"chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source,
                      "page": c.page, "page_end": c.page_end, "metadata": c.metadata, "text": c.text}
                     for c in flat])
        t = time.perf_counter()
        for doc_id, doc_chunks in chunks[strat].items():
            reload["first_pass"][f"{strat}:{doc_id}"] = store.sync_document(
                strat, doc_id, doc_chunks, emb.embed_documents, emb.model_name)
        elapsed = time.perf_counter() - t
        lens = [len(c.text) for c in flat]
        toks = emb.count_tokens([c.text for c in flat])
        stats[strat] = {
            "config": C.STRATEGIES[strat], "chunks": len(flat), "collection_count": store.count(strat),
            "chunks_per_document": {d: len(v) for d, v in chunks[strat].items()},
            "chars": length_stats(lens), "total_chars": sum(lens),
            "tokens_with_prefix": length_stats(toks), "total_tokens": sum(toks),
            "chunks_over_max_seq_len": sum(t > emb.max_seq_length for t in toks),
            "chunks_spanning_pages": sum(c.page_end != c.page for c in flat),
            "embed_and_store_seconds": round(elapsed, 2),
        }
    t = time.perf_counter()
    reload["first_pass"]["C<-B"] = store.copy_collection("B", "C", emb.model_name)
    stats["C"] = {**{k: v for k, v in stats["B"].items() if k not in ("config", "embed_and_store_seconds")},
                  "config": C.STRATEGIES["C"], "collection_count": store.count("C"),
                  "embed_and_store_seconds": round(time.perf_counter() - t, 2),
                  "note": "C dùng lại đúng chunk và vector của B (copy collection, không embed lại)."}

    # Reload check: run the same sync again; nothing should be added or deleted.
    for strat in ("A", "B"):
        for doc_id, doc_chunks in chunks[strat].items():
            reload["second_pass"][f"{strat}:{doc_id}"] = store.sync_document(
                strat, doc_id, doc_chunks, emb.embed_documents, emb.model_name)
    reload["second_pass"]["C<-B"] = store.copy_collection("B", "C", emb.model_name)
    added = sum(v["added"] for v in reload["second_pass"].values())
    deleted = sum(v["deleted"] for v in reload["second_pass"].values())
    reload["second_pass_added_total"], reload["second_pass_deleted_total"] = added, deleted
    reload["counts_after"] = {s: store.count(s) for s in ("A", "B", "C")}
    reload["no_duplicates"] = (added == 0 and deleted == 0 and
                               all(reload["counts_after"][s] == stats[s]["chunks"] for s in ("A", "B", "C")))
    write_json(C.RESULTS_DIR / "reload_check.json", reload)

    run_info = {
        "ingested_at": now(), "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {p: importlib_metadata.version(p) for p in
                     ("chromadb", "sentence-transformers", "transformers", "torch", "pypdf")},
        "embedding": {"provider": "local (Sentence Transformers, CPU)", "model": emb.model_name,
                      "revision": model_revision(emb.model_name), "dimensions": emb.dimensions,
                      "max_seq_length": emb.max_seq_length, "query_prefix": C.QUERY_PREFIX,
                      "passage_prefix": C.PASSAGE_PREFIX, "normalize_embeddings": True,
                      "model_load_seconds": round(load_time, 2)},
        "vector_store": {"name": "Chroma PersistentClient", "path": str(C.CHROMA_DIR.relative_to(C.REPO)),
                         "collections": [f"{C.COLLECTION_PREFIX}_{s}" for s in ("A", "B", "C")],
                         "distance": "cosine (hnsw:space=cosine)", "search": "approximate (HNSW)"},
        "chunking_seconds_total": round(chunk_time, 3), "cost_usd": 0.0,
    }
    write_json(C.RESULTS_DIR / "run_info.json", run_info)
    write_json(C.RESULTS_DIR / "ingest_stats.json", stats)
    print(json.dumps({s: {"chunks": v["chunks"], "sec": v["embed_and_store_seconds"]} for s, v in stats.items()}))
    print("reload no_duplicates:", reload["no_duplicates"])
    if not reload["no_duplicates"]:
        raise SystemExit("Reload check failed")


# --------------------------------------------------------------------------- eval
def cmd_eval(args) -> None:
    from .store import ChromaStore

    split = args.split
    if split == "test" and not (C.RESULTS_DIR / "selection.json").exists():
        raise SystemExit("Run `select` on dev before evaluating test (protocol).")
    qs = [q for q in read_jsonl(C.BENCHMARK_FILE) if q["split"] == split]
    emb = get_embedder()
    store = ChromaStore(C.CHROMA_DIR, emb.dimensions, C.COLLECTION_PREFIX)
    for s in ("A", "B", "C"):
        if store.count(s) == 0:
            raise SystemExit(f"collection {s} is empty: run `ingest` first")
    traces = []
    for q in qs:
        qvec = emb.embed_query(q["question"])
        for strat in ("A", "B", "C"):
            flt = q.get("filter") if C.STRATEGIES[strat]["use_filter"] else None
            t = time.perf_counter()
            hits = store.search(strat, qvec, C.TOP_K, flt)
            latency = time.perf_counter() - t
            res = evaluate_query({**q, "filter": q.get("filter")}, hits)
            traces.append({
                "strategy": strat, "split": split, "id": q["id"], "type": q["type"],
                "question": q["question"], "expected_answer": q["expected_answer"],
                "filter": q.get("filter"), "filter_applied": flt, "gold": q["gold"],
                "hits": [{"rank": h["rank"], "chunk_id": h["chunk_id"], "score": round(h["score"], 4),
                          "document_id": h["metadata"]["document_id"], "page": h["metadata"]["page"],
                          "page_end": h["metadata"]["page_end"], "metadata": h["metadata"], "text": h["text"]}
                         for h in hits],
                "result": res, "search_ms": round(latency * 1000, 1), "run_at": now(),
            })
    for strat in ("A", "B", "C"):
        write_jsonl(C.RESULTS_DIR / "traces" / f"{strat}_{split}.jsonl",
                    [t for t in traces if t["strategy"] == strat])
    summ = summarize(traces)
    metrics_path = C.RESULTS_DIR / "metrics.json"
    allm = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    allm.update(summ)
    write_json(metrics_path, allm)
    for k, s in summ.items():
        print(f"{k}: Hit@1 {s['hit1']}/{s['N']}  Hit@3 {s['hit3']}/{s['N']}  "
              f"filter-compliant {s['filter_ok']}/{s['filter_N']}")


def cmd_select(_args) -> None:
    """Pre-registered rule (benchmark/README.md): dev Hit@3 → dev Hit@1 → fewer chunks."""
    m = json.loads((C.RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    stats = json.loads((C.RESULTS_DIR / "ingest_stats.json").read_text(encoding="utf-8"))
    dev = {k.split("|")[0]: v for k, v in m.items() if k.endswith("|dev")}
    if set(dev) != {"A", "B", "C"}:
        raise SystemExit("dev metrics missing")
    ranking = sorted(dev, key=lambda s: (-dev[s]["hit3"], -dev[s]["hit1"], stats[s]["chunks"], s))
    sel = {"selected_at": now(), "rule": "max dev Hit@3, then max dev Hit@1, then fewer chunks, then name",
           "ranking": ranking, "selected": ranking[0],
           "dev": {s: {"hit1": f"{dev[s]['hit1']}/{dev[s]['N']}", "hit3": f"{dev[s]['hit3']}/{dev[s]['N']}",
                       "chunks": stats[s]["chunks"]} for s in ranking},
           "test_seen_before_selection": False}
    write_json(C.RESULTS_DIR / "selection.json", sel)
    print("selected:", sel["selected"], "ranking:", ranking)


def cmd_all(args) -> None:
    cmd_prepare(args)
    cmd_ingest(args)
    args.split = "dev"
    cmd_eval(args)
    cmd_select(args)
    args.split = "test"
    cmd_eval(args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("ingest")
    e = sub.add_parser("eval")
    e.add_argument("--split", choices=["dev", "test"], required=True)
    sub.add_parser("select")
    sub.add_parser("all")
    args = ap.parse_args(argv)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    {"prepare": cmd_prepare, "ingest": cmd_ingest, "eval": cmd_eval, "select": cmd_select,
     "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
