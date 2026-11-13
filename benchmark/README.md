# Benchmark — Bảo vệ quyền lợi người tiêu dùng (2010 vs 2023)

File: [`questions.jsonl`](questions.jsonl), gồm 24 câu: **18 dev / 6 test**. Bộ câu hỏi và phần chia dev/test được viết và chốt **trước** lần truy xuất đầu tiên.

## Trường dữ liệu

| Trường | Ý nghĩa |
|---|---|
| `id`, `split` | `dNN` = dev, `tNN` = test |
| `type` | `truc_tiep`, `dien_dat_lai`, `dieu_kien`, `ngoai_le`, `metadata_filter` (nhãn dùng để phân tích lỗi) |
| `question`, `expected_answer` | câu hỏi và đáp án viết từ văn bản nguồn |
| `filter` | `null` hoặc `{"legal_regime": "2010" \| "2023"}`. Giá trị lấy **từ ngữ cảnh mà câu hỏi đã nêu** (ví dụ "giao dịch xác lập năm 2015", "quy định hiện hành", "Nghị định 99/2011"), không lấy từ gold |
| `gold[]` | danh sách nguồn hợp lệ `{source_id, page, evidence_quote}`. Nếu có nhiều nguồn hợp lệ (ví dụ cùng quy định ở cả luật cũ và luật mới) thì ghi đủ, và **bất kỳ nguồn nào** khớp đều được tính. `evidence_quote` có thể là danh sách mảnh; khi đó chunk phải chứa **tất cả** các mảnh |

Ánh xạ sang tên trường trong BENCHMARK.md: `gold_source_id` = `gold[i].source_id`, `gold_page` = `gold[i].page`, trong đó `page` là trang vật lý của file PDF, đánh số từ 1.

## Thiết kế

- **Phủ các loại lỗi.** Có 6 câu trực tiếp, 4 câu diễn đạt lại, 6 câu điều kiện, 2 câu ngoại lệ và 6 câu metadata filter (4 dev: d05, d06, d07, d16; 2 test: t01, t02).
- **Vì sao filter có ích.** Luật 2010 và Luật 2023 có nhiều điều gần như trùng chữ (bảo hành, thu hồi hàng khuyết tật, thương lượng, hòa giải viên). Trường `legal_regime` gom văn bản theo chế độ pháp lý: chế độ `2010` gồm Luật 59/2010 và NĐ 99/2011, chế độ `2023` gồm Luật 19/2023 và NĐ 55/2024. Như vậy filter **không trùng 1:1 với tài liệu gold**, vì mỗi giá trị bao phủ 2 văn bản.
- **Câu có nhiều nguồn gold.** d03, d08, d18, t06 có bằng chứng tương đương ở cả hai chế độ. d04 có hai vị trí hợp lệ trong Luật 2023 (Điều 70 và Điều 78).
- **Nguy cơ rò rỉ giữa dev và test.** Không thể tách hoàn toàn theo tài liệu, vì 4 văn bản liên quan chặt chẽ với nhau. Câu test hỏi về các **điều khác** với câu dev (Điều 64, Điều 71 Luật 2023; Điều 22 Luật 2010; Điều 23 NĐ 55; Điều 22 NĐ 99). Riêng t06 (số lần bảo hành) nằm cùng Điều 30 / Điều 21 với d08 (thời hạn bảo hành khi đổi mới), nên có rủi ro rò rỉ về chủ đề. Vì không có tham số nào được tinh chỉnh theo dev, rủi ro này chỉ ảnh hưởng đến tính đại diện của test, không ảnh hưởng tính độc lập.

## Kiểm chứng evidence

`python -m src.run prepare` kiểm tra từng `evidence_quote` có nằm trong văn bản **đã chuẩn hóa** của đúng `source_id` và `page` hay không. Kết quả ghi ở [`results/benchmark_validation.json`](../results/benchmark_validation.json).

Lần kiểm tra đầu phát hiện 2 lỗi số trang: d18 (NĐ 55, gold ghi trang 5, đúng là trang 4) và t06 (Luật 2010, gold ghi trang 11, đúng là trang 10). Hai lỗi này đã được sửa **trước khi chạy truy xuất**. Ngoài ra, chế độ trích xuất mặc định của pypdf chèn khoảng trắng vào giữa từ, nên pipeline chuyển sang chế độ `layout` (xem báo cáo).

## Tiêu chí chọn chiến lược (chốt trước khi xem test)

1. Hit@3 cao nhất trên **dev**.
2. Nếu hòa, chọn Hit@1 cao nhất trên dev.
3. Nếu vẫn hòa, chọn cấu hình có ít chunk hơn (ít tài nguyên hơn).
4. Nếu vẫn hòa, chọn theo tên (A < B < C).

Tiêu chí được cài sẵn trong `python -m src.run select`, chỉ đọc chỉ số dev và ghi kết quả vào `results/selection.json`. Lệnh `eval --split test` từ chối chạy nếu chưa có file selection. Không có tham số nào được tinh chỉnh sau khi xem kết quả, cả dev lẫn test: các tham số chunk (500/50, 1000) được chọn từ trước theo gợi ý trong GUIDE.md.
