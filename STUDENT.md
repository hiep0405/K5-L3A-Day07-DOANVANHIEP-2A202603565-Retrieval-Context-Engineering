# Thông tin học viên — Day 7: Agentic RAG

- **Họ và tên:** Đoàn Văn Hiệp
- **MSSV:** 2A202603565
- **Lớp:** L3A
- **Chủ đề & Corpus:** Bộ luật Lao động Việt Nam 2019 (Luật số 45/2019/QH14, hiệu lực từ 01/01/2021), 145 trang số hóa clean, 190.098 ký tự, 436 chunks (Public Domain).
- **Vector Store & Embedding:** ChromaDB v1.5.9 (Local Persistent Client tại `./chroma_db` kết nối từ Day 6), embedding model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, Cosine distance).
- **Agent Model & Framework:** Gemini 3.8 Flash (`gemini-3.8-flash-low` điều phối qua `agy.exe` non-interactive CLI), vòng lặp ReAct đa lượt với tool search, pack context, trích dẫn bắt buộc `[chunk_id]` và từ chối khi thiếu dữ liệu (*abstention*).
- **Bộ dữ liệu Benchmark:** 24 câu hỏi độc lập (Dev: 17 câu gồm 15 có đáp án + 2 thiếu bằng chứng; Test: 7 câu gồm 5 có đáp án + 2 thiếu bằng chứng).
- **File báo cáo chi tiết:** `report/individual/2A202603565-DOANVANHIEP.md` (đồng bộ tại `report/individual/2A202603565_DOANVANHIEP.md`).
