# Mã nguồn Day 7 — Agentic RAG

Thư mục chứa các module triển khai hệ thống Agentic RAG cho Bộ luật Lao động 2019:

- `agentic.py`: Vòng lặp ReAct core của Agent điều phối tool search, đóng gói context và trích dẫn citations.
- `agent_controller.py`: LLM Controller kết nối Gemini 3.8 Flash qua CLI `agy.exe`, phản hồi schema JSON có cấu trúc.
- `retrieval.py`: Triển khai 3 phương pháp retrieval và các cấu hình cải tiến (Dense ChromaDB, TF-IDF Lexical, Hybrid RRF, Improved Hybrid + Reranker).
- `baseline.py`: TF-IDF Lexical baseline và các hàm tính `hit()`, `pack_context()`.
- `convert_day6.py`: Tiện ích chuyển đổi schema từ Day 6.
- `prepare_data.py`: Pipeline trích xuất 436 chunks từ Day 6 ChromaDB và xây dựng bộ benchmark 24 câu.
- `eval_retrieval.py`: Benchmark so sánh 5 cấu hình retrieval trên dev và test.
- `run_agent_eval.py`: Benchmark toàn diện vòng lặp Agentic RAG, lưu vết traces và kiểm chứng nhãn audit.
- `agent_metrics.py`: Công cụ tổng hợp metrics đánh giá Agent (tool use, grounded answer, citation validity, abstention).

## Hướng dẫn chạy nhanh

```powershell
# Chạy toàn bộ 15 unit tests
python -m unittest discover -s tests -v

# Chạy đánh giá retrieval
python src/eval_retrieval.py

# Chạy đánh giá agent
python src/run_agent_eval.py
```
