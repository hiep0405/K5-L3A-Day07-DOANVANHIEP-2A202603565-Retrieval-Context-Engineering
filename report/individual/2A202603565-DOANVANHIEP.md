# Báo cáo cá nhân — Day 7: Agentic RAG & Context Engineering

- **Họ và tên:** Đoàn Văn Hiệp
- **MSSV:** 2A202603565
- **Lớp:** L3A

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - **Chủ đề & Nguồn:** Bộ luật Lao động Việt Nam năm 2019 (Luật số 45/2019/QH14, ban hành ngày 20/11/2019, có hiệu lực từ ngày 01/01/2021). Văn bản quy phạm pháp luật thuộc phạm vi công cộng (Public Domain) theo Điều 19 Luật Sở hữu trí tuệ 2005 (sửa đổi, bổ sung 2022).
  - **Số trang & quy tắc tính trang:** File PDF số hóa clean (`data/bo_luat_lao_dong_2019_clean.pdf`) gồm **145 trang** đầy đủ 17 Chương và 220 Điều (tổng cộng 190.098 ký tự, trung bình 1.311 ký tự/trang, không có trang trắng hay trang scan rỗng).
  - **Số chunk:** Kế thừa trực tiếp từ Strategy B của Day 6, toàn bộ văn bản được chia thành **436 chunks** theo cấu trúc phân đoạn/điều khoản pháp lý (max size 600 ký tự), lưu trữ dưới dạng JSON array chuẩn tại `data/chunks.json`. Mỗi chunk có đầy đủ `id`, `source_id`, `source`, `page`, `text`, `metadata` (chương, chiến lược).
- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** **ChromaDB v1.5.9** (Local Persistent Client tại thư mục `./chroma_db` từ Day 6), sử dụng khoảng cách Cosine (`hnsw:space: cosine`).
  - **Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, chuẩn hóa L2, chạy on-premise CPU trên PyTorch 2.15.0).
- **Agent model, cách chạy và phân chia dev/test:**
  - **Agent Controller Model:** **Gemini 3.8 Flash** (gọi thông qua `agy.exe` non-interactive CLI với flag `--model gemini-3.8-flash-low`). Không tốn chi phí token thương mại ($0.0), phản hồi ổn định với schema JSON nghiêm ngặt điều khiển vòng lặp tool calling.
  - **Phân chia Dev / Test:**
    - **Tập Dev (17 câu):** 15 câu có đáp án trong tài liệu (`q01`–`q15`) + 2 câu thiếu bằng chứng ngoài phạm vi (`q_unans_dev_1`, `q_unans_dev_2`).
    - **Tập Test (7 câu):** 5 câu có đáp án trong tài liệu (`q16`–`q20`) + 2 câu thiếu bằng chứng ngoài phạm vi (`q_unans_test_1`, `q_unans_test_2`).
    - **Tổng số câu benchmark:** **24 câu** (thỏa mãn trọn vẹn yêu cầu tối thiểu tại `BENCHMARK.md`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có evidence:** `gold` là chuỗi ký tự nguyên văn (*verbatim substring*) trích xuất trực tiếp từ chunk trên đúng trang quy định của Bộ luật Lao động, đảm bảo khớp hàm `hit()` trong `baseline.py`. Tiêu chí chấm: 2 = đúng, đủ và trích dẫn chuẩn; 1 = đúng một phần; 0 = sai hoặc từ chối nhầm.
  - **Câu thiếu evidence:** Đặt câu hỏi thuộc các lĩnh vực pháp lý khác không thuộc Bộ luật Lao động 2019 (Bảo hiểm Y tế, Giao thông đường bộ, Thuế Thu nhập Cá nhân, Tố tụng Hình sự). Tiêu chí chấm: Agent bắt buộc phải tra cứu và từ chối trả lời (*abstain*); nếu suy đoán hoặc bịa câu trả lời thì bị chấm thất bại.

---

## 2. Retrieval và thử nghiệm

### Bảng kết quả so sánh các phương pháp và cấu hình retrieval

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | TF-IDF Lexical, top-k=3, không rerank, budget 800 ký tự | Baseline từ khóa chuẩn ngữ nghĩa tiếng Việt NFC | dev | 9/15 (60.0%) | 12/15 (80.0%) | 9/15 (60.0%) | 1.22ms / $0.0 |
| **Phương pháp 2** | Dense Retrieval (MiniLM-L12-v2 + ChromaDB), top-k=3, budget 800 | Tìm kiếm ngữ nghĩa vector embedding đa ngôn ngữ | dev | 7/15 (46.7%) | 10/15 (66.7%) | 7/15 (46.7%) | 20.60ms / $0.0 |
| **Phương pháp 3** | Hybrid RRF (Dense + TF-IDF, k=60), top-k=3, budget 800 | Hợp nhất thứ hạng đối ứng giữa Dense và Lexical để bù trừ nhược điểm từ khóa/ngữ nghĩa | dev | 10/15 (66.7%) | 11/15 (73.3%) | 10/15 (66.7%) | 20.64ms / $0.0 |
| **Cải thiện 1** | Improved Hybrid + Keyword Rerank (Boost số Điều luật & số liệu), top-k=3, budget 800 | Khuếch đại điểm số cho các chunk chứa đúng số Điều (Điều \d+) hoặc số liệu trong query | dev | 11/15 (73.3%) | 12/15 (80.0%) | 11/15 (73.3%) | 20.30ms / $0.0 |
| **Cải thiện 2 (Đã chọn)** | Improved Hybrid Rerank + Mở rộng Top-k=4, Budget=1200 ký tự | Tránh rơi rớt chunk do giới hạn ngữ cảnh khi điều luật dài; tăng khả năng bao phủ evidence | dev | 11/15 (73.3%) | 12/15 (80.0%) | 11/15 (73.3%) | 19.48ms / $0.0 |
| **Baseline (TF-IDF)** | TF-IDF Lexical, top-k=3, budget 800 ký tự | Đánh giá độc lập trên tập test kiểm chứng | test | 4/5 (80.0%) | 5/5 (100.0%) | 4/5 (80.0%) | 1.09ms / $0.0 |
| **Cấu hình đã chọn** | Improved Hybrid Rerank, top-k=4, budget 1200 ký tự | Cấu hình tối ưu chốt từ dev chạy trên test | test | 2/5 (40.0%) | 4/5 (80.0%) | 2/5 (40.0%) | 21.43ms / $0.0 |

### Phân tích lỗi trên tập Dev và lý do chọn cấu hình

1. **Lỗi 1 — q01 (Dev): Nhầm lẫn ngữ nghĩa giữa Định nghĩa chung và Quy định chuyên biệt**
   - *Câu hỏi:* "Độ tuổi lao động tối thiểu của người lao động theo Bộ luật Lao động 2019 là bao nhiêu tuổi?"
   - *Nguyên nhân:* Bằng chứng chuẩn nằm ở Điều 3 khoản 1 (Trang 3). Tuy nhiên, cả Dense và TF-IDF đều ưu tiên đẩy các chunk thuộc Chương XI (Lao động chưa thành niên, Điều 143, Trang 89) lên đầu do tần suất xuất hiện dày đặc của cụm "tuổi lao động", "chưa đủ 15 tuổi". Kết quả là chunk Điều 3 bị đẩy ra ngoài top-3.
   - *Biện pháp cải thiện:* Áp dụng cơ chế Agentic Multi-turn search để agent có thể tra cứu bổ sung theo từ khóa định nghĩa hoặc Điều 3; đồng thời Reranker nhận diện các từ khóa trực tiếp.
2. **Lỗi 2 — q03 (Dev): Tràn Context Budget (Context Budget Overflow)**
   - *Câu hỏi:* "Bộ luật Lao động 2019 quy định hợp đồng lao động được giao kết theo những loại nào?"
   - *Nguyên nhân:* Chunk chứa đáp án (Điều 20, Trang 11) được xếp hạng ở vị trí thứ 3 trong top-k. Tuy nhiên, do chunk 1 và chunk 2 có độ dài lớn (tổng cộng 1.142 ký tự), hàm `pack_context` với ngân sách mặc định 800 ký tự đã loại bỏ chunk thứ 3, khiến `Context Hit` bị thất bại dù `Hit@3` đạt.
   - *Biện pháp cải thiện:* Tăng `context budget` từ 800 ký tự lên 1.200 ký tự (Cải thiện 2). Điều này giúp giữ trọn vẹn chunk thứ 3 trong ngữ cảnh gửi sang LLM mà không làm loãng thông tin.

---

## 3. Agent và đánh giá

### Thiết kế Agent và cơ chế điều phối context
- **Kiến trúc:** Agentic RAG Controller dựa trên mô hình LLM thật (**Gemini 3.8 Flash**).
- **Vòng lặp tương tác (ReAct loop):**
  - **Lượt 1:** Nhận câu hỏi, Agent phân tích và phát lệnh `action: search` với `query` tối ưu hóa.
  - **Retriever:** Truy vấn vector store kết hợp Hybrid RRF và Reranker, trả về top chunk liên quan kèm metadata và ID.
  - **Context Packing:** Hệ thống gom các chunk hợp lệ vào context budget (1.200 ký tự), gán nhãn citation `[chunk_id]`.
  - **Lượt 2+:** Controller đọc ngữ cảnh:
    - Nếu đủ bằng chứng: Trả lời `action: answer` kèm trích dẫn `[chunk_id]` bắt buộc.
    - Nếu câu hỏi ngoài phạm vi: Trả lời từ chối `Không đủ bằng chứng để trả lời.`
    - Nếu chưa đủ: Tiếp tục gọi thêm lượt search (tối đa `max_searches = 3`).
- **Xử lý lỗi & Abstention:** Kiểm tra bắt buộc mọi `chunk_id` trong `citations` phải tồn tại trong context và phải xuất hiện dưới dạng `[chunk_id]` trong nội dung trả lời; nếu vi phạm, agent tự động chuyển về trạng thái `abstain`.

### Bảng tổng hợp đánh giá Agent (Agent Metrics)

| Cấu hình / split | Tool use | Lượt gọi TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Baseline Agent (dev)** | 17/17 (100.0%) | 1.6 lượt | 0: 5 \| 1: 0 \| 2: 10 (TB: 1.33/2.0) | 10/10 (100.0%) | 10/10 (100.0%) | 2/2 (100.0%) | 5/15 (33.3%) |
| **Chosen Agent (dev)** | 17/17 (100.0%) | 1.8 lượt | 0: 5 \| 1: 0 \| 2: 10 (TB: 1.33/2.0) | 10/10 (100.0%) | 10/10 (100.0%) | 2/2 (100.0%) | 5/15 (33.3%) |
| **Baseline Agent (test)** | 7/7 (100.0%) | 1.6 lượt | 0: 2 \| 1: 1 \| 2: 2 (TB: 1.00/2.0) | 3/4 (75.0%) | 3/3 (100.0%) | 2/2 (100.0%) | 2/5 (40.0%) |
| **Chosen Agent (test)** | 7/7 (100.0%) | 2.1 lượt | 0: 2 \| 1: 0 \| 2: 3 (TB: 1.20/2.0) | 3/3 (100.0%) | 3/3 (100.0%) | 2/2 (100.0%) | 2/5 (40.0%) |

> **Nhận xét chính:**
> - Trên tập Test, **Chosen Agent** vượt trội so với Baseline: Điểm chất lượng trung bình tăng từ **1.00 lên 1.20**, số câu đạt điểm tuyệt đối (grade 2) tăng từ **2 lên 3 câu**.
> - **Độ chính xác trích dẫn (Citation validity)** của Chosen Agent đạt **100% (3/3)** trên Test (so với 75% ở Baseline).
> - **Năng lực từ chối (Abstention):** Cả hai cấu hình đều đạt **100% (2/2)** câu từ chối chính xác trên tập Test đối với các câu hỏi ngoài phạm vi, không ghi nhận bất kỳ hiện tượng ảo giác (hallucination) nào.

---

### Bảng minh họa chi tiết các trường hợp tiêu biểu

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q19** (test) | Có đáp án (Tiền lương làm thêm giờ lễ tết) | 1 lượt search: `"tiền lương làm thêm giờ ngày nghỉ lễ tết ít nhất"`. Kết quả kéo được chunk `bo-luat-lao-dong-2019_p61_b_1` (score: 0.03279, Điều 98). | "Theo quy định, người lao động làm thêm giờ vào ngày nghỉ lễ, tết, ngày nghỉ có hưởng lương thì được hưởng ít nhất bằng 300% tiền lương tính theo đơn giá tiền lương hoặc tiền lương thực trả theo công việc đang làm, chưa kể tiền lương ngày lễ, tết, ngày nghỉ có hưởng lương đối với người lao động hưởng lương ngày [bo-luat-lao-dong-2019_p61_b_1]." | **Grade: 2/2**; Citation hợp lệ: **True**. Trả lời đầy đủ, chính xác con số 300% và viện dẫn đúng Điều 98. |
| **q_unans_test_1** (test) | Thiếu bằng chứng (Thuế suất biểu thuế lũy tiến TNCN) | 1 lượt search: `"thuế thu nhập cá nhân"`. Kết quả chỉ có các chunk khấu trừ lương Điều 129 và chính sách thuế Điều 84, không có biểu thuế lũy tiến từng phần. | "Không đủ bằng chứng để trả lời." | **Abstention: True**. Agent phát hiện câu hỏi thuộc Luật Thuế TNCN, không có trong Bộ luật Lao động và từ chối chuẩn xác. |
| **q16** (test) | Lỗi / Từ chối nhầm (Các hình thức kỷ luật lao động) | Lượt 1 search: `"hình thức xử lý kỷ luật lao động"`. Lượt 2 search: `"Điều 124 hình thức xử lý kỷ luật lao động"`. Kết quả trả về Điều 125 (sa thải) và Điều 128 (tạm đình chỉ), không kịp gói khoản 1 Điều 124 vào ngữ cảnh. | "Không đủ bằng chứng để trả lời." | **Grade: 0/2**; False abstention: **True**. Do các điều luật lân cận có mật độ từ khóa kỷ luật cao làm loãng thứ hạng đoạn mở đầu Điều 124. |

---

## 4. Tái lập và đường dẫn kết quả

- **Đường dẫn thư mục kết quả:**
  - Kết quả Retrieval Benchmark: `starter/results/retrieval_benchmark_summary.json`
  - Trace Baseline Dev: `starter/results/agent_baseline_dev/` (`traces.json`, `reviews.json`, `summary.json`)
  - Trace Chosen Dev: `starter/results/agent_chosen_dev/` (`traces.json`, `reviews.json`, `summary.json`)
  - Trace Baseline Test: `starter/results/agent_baseline_test/` (`traces.json`, `reviews.json`, `summary.json`)
  - Trace Chosen Test: `starter/results/agent_chosen_test/` (`traces.json`, `reviews.json`, `summary.json`)
- **Lệnh chạy tái lập toàn bộ:**
  ```powershell
  # 1. Chạy unit tests kiểm tra hợp đồng code
  python -m unittest discover -s tests -v

  # 2. Chạy benchmark so sánh 5 cấu hình retrieval
  python starter/eval_retrieval.py

  # 3. Chạy đánh giá Agent và tổng hợp nhãn kiểm chứng
  python starter/run_agent_eval.py
  ```
- **Chi phí & Thời gian:**
  - Tổng chi phí: **0.0 USD** (Hoàn toàn miễn phí, sử dụng tài nguyên local CPU và model Gemini Flash nghiên cứu).
  - Thời gian retrieval: trung bình **19.48ms** cho cấu hình Chosen Hybrid Rerank.
- **Giới hạn & Hướng phát triển:**
  - Đối với các câu hỏi liệt kê toàn diện (như 4 hình thức kỷ luật tại Điều 124), nếu câu hỏi không nêu rõ tên điều luật, retrieval dạng chunk đơn lẻ có thể bị nhiễu bởi các điều khoản áp dụng chi tiết phía sau. Hướng phát triển tiếp theo là áp dụng phân đoạn đa cấp độ (Parent-Child chunking) hoặc mở rộng ngữ cảnh phân cấp theo Chương.
