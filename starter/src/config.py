"""Central configuration: paths, model and the three strategies (A/B/C).

Corpus, benchmark, embedding model and vector store are identical for every
strategy; only the parameters below change.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
SOURCES_FILE = DATA_DIR / "sources.json"
MANIFEST_FILE = DATA_DIR / "manifest.json"
BENCHMARK_FILE = REPO / "benchmark" / "questions.jsonl"
RESULTS_DIR = REPO / "results"
CHROMA_DIR = REPO / "starter" / "chroma_db"   # persistent store (gitignored, rebuilt by `ingest`)

# A page counts as a content page when its normalized text has at least this many characters.
MIN_CONTENT_CHARS = 300

# Embedding model: local, multilingual, query/passage prefixes per the model card.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_REVISION = None          # resolved commit hash is recorded in results/run_info.json
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
EMBED_BATCH_SIZE = 16

TOP_K = 3
COLLECTION_PREFIX = "lab06"
DISTANCE = "cosine"

STRATEGIES = {
    "A": {
        "chunking": "fixed_size",
        "chunk_size": 500,
        "overlap": 50,
        "use_filter": False,
        "description": "Cửa sổ cố định 500 ký tự, overlap 50, trong phạm vi từng trang; metadata đầy đủ; không lọc.",
    },
    "B": {
        "chunking": "structure",
        "max_size": 1000,
        "use_filter": False,
        "description": "Theo cấu trúc Điều → khoản (gộp khoản tới 1000 ký tự, tách khoản quá dài theo điểm/câu), "
                       "mỗi chunk có tiêu đề ngữ cảnh [văn bản · Chương · Điều]; metadata đầy đủ; không lọc.",
    },
    "C": {
        "chunking": "structure",
        "max_size": 1000,
        "use_filter": True,
        "description": "Giống hệt B (cùng chunk, cùng vector); thêm lọc metadata `legal_regime` chỉ khi câu hỏi có filter.",
    },
}
# Fields a question filter may use (values must exist in the metadata, see data/manifest.json).
ALLOWED_FILTER_FIELDS = {"legal_regime", "document_type", "document_id"}
