"""Day 7 Agent Controller & Runner.
Uses Gemini via agy CLI as the LLM Controller, and integrates with Day 7 Retrievers.
"""
from __future__ import annotations
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from agentic import run_agent
from baseline import pack_context
from retrieval import ImprovedHybridRetriever, HybridRetriever, DenseRetriever, TfidfRetriever


class GeminiAgyController:
    """Real LLM Controller using Gemini 3.8 Flash via agy CLI."""
    def __init__(self, agy_path: str = r"C:\Users\HIEPHIEP\.gemini\bin\agy.exe", model: str = "gemini-3.8-flash-low"):
        self.agy_path = agy_path
        self.model = model

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            prompt = f"""Bạn là AI phân tích pháp lý phục vụ hệ thống Agentic RAG cho Bộ luật Lao động Việt Nam 2019.
Người dùng hỏi: "{question}"

Bạn chưa có tài liệu ngữ cảnh nào. Hãy tạo 1 truy vấn tìm kiếm ngắn gọn, chuẩn xác nhất để tìm trong văn bản Bộ luật Lao động.
Chỉ trả về DUY NHẤT một chuỗi JSON hợp lệ không có markdown phụ trợ:
{{"action": "search", "query": "<từ khóa tìm kiếm tốt nhất>"}}"""
        else:
            latest = history[-1]
            context = latest.get('context', '')
            context_ids = latest.get('context_ids', [])
            queries_so_far = latest.get('queries', [])

            prompt = f"""Bạn là AI phân tích pháp lý phục vụ hệ thống Agentic RAG cho Bộ luật Lao động Việt Nam 2019.
Người dùng hỏi: "{question}"

CÁC TRUY VẤN ĐÃ TÌM: {queries_so_far}

TÀI LIỆU NGỮ CẢNH ĐÃ TÌM ĐƯỢC:
{context}

DANH SÁCH MÃ CHUNK TRONG NGỮ CẢNH: {context_ids}

HƯỚNG DẪN XỬ LÝ QUYẾT ĐỊNH:
1. Nếu ngữ cảnh ĐÃ CÓ ĐỦ BẰNG CHỨNG để trả lời câu hỏi:
   - Viết câu trả lời ngắn gọn, chuẩn xác.
   - BẮT BUỘC chèn ký hiệu trích dẫn [mã_chunk] ngay sau mỗi nhận định có nguồn (ví dụ: "...độ tuổi tối thiểu là 15 tuổi [bo-luat-lao-dong-2019_p3_b_0].").
   - Tất cả mã trong "citations" BẮT BUỘC phải nằm trong DANH SÁCH MÃ CHUNK TRONG NGỮ CẢNH ở trên và BẮT BUỘC phải xuất hiện trong "answer".
   - Trả về JSON:
     {{"action": "answer", "answer": "<câu trả lời kèm các mã [mã_chunk]>", "citations": ["<mã_chunk>"]}}

2. Nếu câu hỏi KHÔNG THUỘC PHẠM VI Bộ luật Lao động 2019 (ví dụ hỏi về bảo hiểm y tế, luật giao thông, luật thuế, luật hình sự...) hoặc ngữ cảnh cho thấy hoàn toàn không có bằng chứng liên quan:
   - Trả về JSON từ chối:
     {{"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}}

3. Nếu thông tin có thể có trong Bộ luật Lao động nhưng từ khóa trước chưa trúng, và bạn muốn tìm kiếm thêm:
   - Trả về JSON tìm kiếm tiếp:
     {{"action": "search", "query": "<từ khóa mới cụ thể hơn>"}}

Chỉ xuất DUY NHẤT 1 chuỗi JSON hợp lệ:"""

        res = subprocess.run(
            [self.agy_path, '--model', self.model, '-p', prompt],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        out = res.stdout.strip()
        m = re.search(r'\{.*\}', out, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if data.get('action') in ('search', 'answer'):
                    return data
            except Exception:
                pass

        # Fallback heuristic
        if not history:
            return {'action': 'search', 'query': question}
        return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}
