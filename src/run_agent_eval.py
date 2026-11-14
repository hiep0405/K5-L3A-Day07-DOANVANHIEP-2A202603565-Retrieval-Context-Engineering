"""Comprehensive evaluation of Agentic RAG pipeline on Dev and Test splits.
Generates traces.json, reviews.json, and summary.json conforming strictly to Day 7 specs.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any

from agentic import run_agent
from agent_controller import GeminiAgyController
from agent_metrics import aggregate
from retrieval import DenseRetriever, TfidfRetriever, HybridRetriever, ImprovedHybridRetriever


def audit_question(case: dict[str, Any], trace: dict[str, Any], chunk_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cid = case['id']
    status = trace['status']
    citations = trace.get('citations', [])
    answer = trace.get('answer', '')

    review: dict[str, Any] = {'query_id': cid}

    if not case['answerable']:
        # Unanswerable case
        is_abstain = (status == 'abstain')
        review['abstention_correct'] = is_abstain
        review['citation_support'] = []
        review['reason'] = "Thành công từ chối trả lời câu hỏi ngoài phạm vi Bộ luật Lao động." if is_abstain else "Thất bại: Đã trả lời suy đoán câu hỏi ngoài phạm vi."
        return review

    # Answerable case
    if status == 'abstain':
        review['answer_grade'] = 0
        review['citation_support'] = []
        review['reason'] = "Từ chối nhầm (false abstention) trên câu có đáp án trong tài liệu."
        return review

    # Evaluate citations
    citation_support = []
    gold_info = case.get('expected_answer', '').lower()
    for c_id in citations:
        c_chunk = chunk_map.get(c_id)
        if not c_chunk:
            citation_support.append(False)
            continue
        c_text = c_chunk['text'].lower()
        # Check if cited chunk text genuinely relates to the topic and provides grounding
        words = [w for w in re.findall(r'\w+', gold_info) if len(w) > 3]
        overlap = sum(1 for w in words if w in c_text)
        is_supported = (overlap >= 1 or any(num in c_text for num in re.findall(r'\d+', gold_info)))
        citation_support.append(bool(is_supported))

    review['citation_support'] = citation_support

    # Grade answer: 2 = correct & sufficient, 1 = partially correct, 0 = incorrect
    has_valid_cits = all(citation_support) and len(citation_support) > 0
    ans_lower = answer.lower()
    
    # Check key numbers or terms
    expected_tokens = [w for w in re.findall(r'\w+', gold_info) if len(w) > 3]
    match_count = sum(1 for w in expected_tokens if w in ans_lower)
    match_ratio = match_count / len(expected_tokens) if expected_tokens else 1.0

    if has_valid_cits and match_ratio >= 0.5:
        grade = 2
        reason = "Trả lời chính xác, trích dẫn chuẩn xác và hỗ trợ đầy đủ nhận định."
    elif match_ratio >= 0.3:
        grade = 1
        reason = "Trả lời đúng một phần hoặc trích dẫn chưa bao quát toàn bộ điều kiện."
    else:
        grade = 0
        reason = "Trả lời chưa chính xác hoặc thiếu bằng chứng xác thực."

    review['answer_grade'] = grade
    review['reason'] = reason
    return review


def run_evaluation_suite(controller, retriever, cases: list[dict[str, Any]], chunk_map: dict[str, dict[str, Any]],
                         output_dir: Path, split_name: str, config_name: str,
                         top_k: int = 4, budget: int = 1200, max_searches: int = 3):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [c for c in cases if c.get('split') == split_name]

    print(f"\n[{config_name}] Running Agent Evaluation on split '{split_name}' ({len(selected_cases)} cases)...")
    traces = []
    reviews = []

    for idx, c in enumerate(selected_cases, start=1):
        t0 = time.perf_counter()
        print(f"  ({idx}/{len(selected_cases)}) [{c['id']}] {c['question'][:60]}...", flush=True)
        try:
            res = run_agent(
                question=c['question'],
                controller=controller,
                retriever=retriever,
                filters=c.get('filter'),
                max_searches=max_searches,
                top_k=top_k,
                budget=budget
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            trace_entry = {
                'query_id': c['id'],
                'split': split_name,
                'status': res['status'],
                'answer': res['answer'],
                'citations': res['citations'],
                'events': res['events'],
                'elapsed_ms': elapsed_ms,
                'cost_usd': 0.0
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            print(f"    ERROR running query {c['id']}: {exc}")
            trace_entry = {
                'query_id': c['id'],
                'split': split_name,
                'status': 'error',
                'error': str(exc),
                'events': [],
                'citations': [],
                'elapsed_ms': elapsed_ms,
                'cost_usd': 0.0
            }

        traces.append(trace_entry)
        review = audit_question(c, trace_entry, chunk_map)
        reviews.append(review)
        print(f"    -> Status: {trace_entry['status']} | Citations: {trace_entry['citations']} | Latency: {elapsed_ms}ms")

    # Save traces and reviews
    with open(output_dir / 'traces.json', 'w', encoding='utf-8') as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'reviews.json', 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    # Compute aggregate metrics
    metrics = aggregate(selected_cases, traces, reviews)
    summary_report = {
        'config_name': config_name,
        'split': split_name,
        'top_k': top_k,
        'budget': budget,
        'max_searches': max_searches,
        **metrics
    }

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    print(f"  Summary Report for {config_name} ({split_name}):")
    print(json.dumps(summary_report, ensure_ascii=False, indent=2))
    return summary_report, traces, reviews


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / 'data'
    results_base = root / 'starter' / 'results'

    with open(data_dir / 'chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)
    chunk_map = {c['id']: c for c in chunks}

    with open(data_dir / 'cases.json', encoding='utf-8') as f:
        cases = json.load(f)

    # Connect retrievers
    chroma_path = str(root.parent.parent / 'DAY06_VIN_AI' / 'K5-L3-Day06-2A202603565-DataIngestionEngineering' / 'chroma_db')
    dense_retriever = DenseRetriever(chroma_path=chroma_path)
    tfidf_retriever = TfidfRetriever(chunks=chunks)
    hybrid_retriever = HybridRetriever(dense_retriever, tfidf_retriever, rrf_k=60)
    improved_retriever = ImprovedHybridRetriever(hybrid_retriever)

    controller = GeminiAgyController()

    # 1. Run Baseline Agent on DEV
    run_evaluation_suite(
        controller=controller,
        retriever=tfidf_retriever,
        cases=cases,
        chunk_map=chunk_map,
        output_dir=results_base / 'agent_baseline_dev',
        split_name='dev',
        config_name='Baseline Agent (TF-IDF, top_k=3, budget=800, max_searches=2)',
        top_k=3,
        budget=800,
        max_searches=2
    )

    # 2. Run Chosen Agent (Improved Hybrid Rerank) on DEV
    run_evaluation_suite(
        controller=controller,
        retriever=improved_retriever,
        cases=cases,
        chunk_map=chunk_map,
        output_dir=results_base / 'agent_chosen_dev',
        split_name='dev',
        config_name='Chosen Agent (Hybrid Rerank, top_k=4, budget=1200, max_searches=3)',
        top_k=4,
        budget=1200,
        max_searches=3
    )

    # 3. Run Baseline Agent on TEST
    run_evaluation_suite(
        controller=controller,
        retriever=tfidf_retriever,
        cases=cases,
        chunk_map=chunk_map,
        output_dir=results_base / 'agent_baseline_test',
        split_name='test',
        config_name='Baseline Agent (TF-IDF, top_k=3, budget=800, max_searches=2) [TEST]',
        top_k=3,
        budget=800,
        max_searches=2
    )

    # 4. Run Chosen Agent on TEST
    run_evaluation_suite(
        controller=controller,
        retriever=improved_retriever,
        cases=cases,
        chunk_map=chunk_map,
        output_dir=results_base / 'agent_chosen_test',
        split_name='test',
        config_name='Chosen Agent (Hybrid Rerank, top_k=4, budget=1200, max_searches=3) [TEST]',
        top_k=4,
        budget=1200,
        max_searches=3
    )


if __name__ == '__main__':
    main()
