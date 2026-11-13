# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bài làm — Lê Xuân Hạ (2A202603571)

- **Corpus:** 4 văn bản về bảo vệ quyền lợi người tiêu dùng, lấy từ Công báo: Luật 59/2010, NĐ 99/2011, Luật 19/2023, NĐ 55/2024. Tổng 109 trang nội dung, lưu ở `data/raw/`. Nguồn, quyền sử dụng và cách tính trang ghi trong [data/manifest.json](data/manifest.json).
- **Embedding và vector store:** `intfloat/multilingual-e5-base` chạy local (768 chiều, không tốn chi phí). Vector store là Chroma `PersistentClient` lưu ở `starter/chroma_db`, dùng cosine.
- **Ba chiến lược:**
  - **A**: chunk cố định 500/50.
  - **B**: chunk theo cấu trúc Điều/khoản.
  - **C**: giống B, thêm filter `legal_regime`.
- **Benchmark:** 24 câu (18 dev / 6 test), xem [benchmark/](benchmark/README.md).
- **Kết quả:** trace, chỉ số và nhãn audit nằm trong [results/](results/). Báo cáo: [report/individual/2A202603571_LeXuanHa.md](report/individual/2A202603571_LeXuanHa.md).

### Cài đặt và chạy lại (Windows PowerShell; macOS/Linux tương tự)

```bash
cd starter
python -m venv .venv
.venv\Scripts\Activate.ps1                     # macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m src.run prepare        # đọc PDF, ghi manifest + page_log, kiểm chứng evidence của benchmark
python -m src.run ingest         # chunk A/B, embed, lưu Chroma (C copy từ B), kiểm tra nạp lại không trùng
python -m src.run eval --split dev
python -m src.run select         # chọn chiến lược theo tiêu chí đã chốt, chỉ dùng dev
python -m src.run eval --split test
python -m src.audit sample       # tạo bảng audit coherence / metadata / grounding
python -m src.audit review       # in từng nhãn kèm đoạn text đã chấm để rà soát (coherence | grounding | metadata)
python -m src.audit summarize    # tổng hợp nhãn trong results/audit/labels.json
```

- Lần đầu chạy sẽ tải model khoảng 1,1 GB từ Hugging Face. Model này không cần đăng nhập hay API key, nên không cần file `.env`.
- Cũng có thể chạy toàn bộ bằng `python -m src.run all`.
- Code bài làm nằm trong [starter/src/](starter/src/). Bài **không dùng** Docker/pgvector của starter; file `scripts/check_setup.py` chỉ còn là tiện ích tùy chọn.

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
