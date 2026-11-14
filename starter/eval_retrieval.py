"""Benchmark runner for Day 7 retrieval configurations."""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path
from typing import Any

from baseline import hit, pack_context
from retrieval import DenseRetriever, TfidfRetriever, HybridRetriever, ImprovedHybridRetriever


def evaluate_retriever(retriever, questions: list[dict[str, Any]], top_k: int = 3, budget: int = 800) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    latencies = []
    for q in questions:
        start = time.perf_counter()
        filters = q.get('filter')
        results = retriever.search(q['question'], filters=filters, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        h1 = bool(marked and marked[0]['hit'])
        h3 = any(c['hit'] for c in marked[:3])
        c_hit = any(c['hit'] and c['id'] in ids for c in marked)

        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'filter': filters,
            'top_k': top_k,
            'budget_chars': budget,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': round(elapsed_ms, 2),
            'hit1': h1,
            'hit3': h3,
            'context_hit': c_hit
        })

    n = len(questions)
    hit1_count = sum(t['hit1'] for t in traces)
    hit3_count = sum(t['hit3'] for t in traces)
    context_hit_count = sum(t['context_hit'] for t in traces)

    summary = {
        'n': n,
        'hit1': f"{hit1_count}/{n} ({hit1_count / n * 100:.1f}%)",
        'hit3': f"{hit3_count}/{n} ({hit3_count / n * 100:.1f}%)",
        'context_evidence_rate': f"{context_hit_count}/{n} ({context_hit_count / n * 100:.1f}%)",
        'hit1_raw': hit1_count,
        'hit3_raw': hit3_count,
        'context_hit_raw': context_hit_count,
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 1),
        'median_latency_ms': round(statistics.median(latencies), 2),
        'cost_usd': 0.0
    }
    return summary, traces


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / 'data'
    results_dir = root / 'starter' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / 'chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)

    with open(data_dir / 'questions.json', encoding='utf-8') as f:
        all_questions = json.load(f)

    dev_questions = [q for q in all_questions if q.get('split') == 'dev']
    test_questions = [q for q in all_questions if q.get('split') == 'test']

    print(f"Loaded {len(chunks)} chunks, {len(dev_questions)} dev questions, {len(test_questions)} test questions.")

    # Instantiate retrievers
    chroma_path = str(root.parent.parent / 'DAY06_VIN_AI' / 'K5-L3-Day06-2A202603565-DataIngestionEngineering' / 'chroma_db')
    print(f"Connecting ChromaDB from: {chroma_path}")
    
    dense_retriever = DenseRetriever(chroma_path=chroma_path)
    tfidf_retriever = TfidfRetriever(chunks=chunks)
    hybrid_retriever = HybridRetriever(dense_retriever, tfidf_retriever, rrf_k=60)
    improved_retriever = ImprovedHybridRetriever(hybrid_retriever)

    configs = [
        {
            'name': 'Baseline (TF-IDF)',
            'retriever': tfidf_retriever,
            'split': 'dev',
            'top_k': 3,
            'budget': 800,
            'desc': 'TF-IDF Lexical, top-k=3, budget=800'
        },
        {
            'name': 'Phương pháp 2 (Dense)',
            'retriever': dense_retriever,
            'split': 'dev',
            'top_k': 3,
            'budget': 800,
            'desc': 'Dense MiniLM-L12-v2 + ChromaDB Cosine, top-k=3, budget=800'
        },
        {
            'name': 'Phương pháp 3 (Hybrid RRF)',
            'retriever': hybrid_retriever,
            'split': 'dev',
            'top_k': 3,
            'budget': 800,
            'desc': 'Hybrid RRF (Dense + TF-IDF, k=60), top-k=3, budget=800'
        },
        {
            'name': 'Cải thiện 1 (Hybrid + Rerank)',
            'retriever': improved_retriever,
            'split': 'dev',
            'top_k': 3,
            'budget': 800,
            'desc': 'Hybrid RRF + Article/Number Keyword Boost Rerank, top-k=3, budget=800'
        },
        {
            'name': 'Cải thiện 2 (Hybrid Rerank + Budget Expand)',
            'retriever': improved_retriever,
            'split': 'dev',
            'top_k': 4,
            'budget': 1200,
            'desc': 'Hybrid RRF + Rerank, mở rộng top-k=4, budget=1200 chars'
        },
        # Held-out Test evaluations
        {
            'name': 'Baseline (TF-IDF) [TEST]',
            'retriever': tfidf_retriever,
            'split': 'test',
            'top_k': 3,
            'budget': 800,
            'desc': 'TF-IDF Lexical, top-k=3, budget=800'
        },
        {
            'name': 'Cấu hình đã chọn (Cải thiện 2) [TEST]',
            'retriever': improved_retriever,
            'split': 'test',
            'top_k': 4,
            'budget': 1200,
            'desc': 'Hybrid RRF + Rerank, top-k=4, budget=1200 chars'
        }
    ]

    all_summaries = []
    for cfg in configs:
        qs = dev_questions if cfg['split'] == 'dev' else test_questions
        summary, traces = evaluate_retriever(cfg['retriever'], qs, top_k=cfg['top_k'], budget=cfg['budget'])
        summary['config_name'] = cfg['name']
        summary['desc'] = cfg['desc']
        summary['split'] = cfg['split']
        all_summaries.append(summary)

        # Save traces
        slug = cfg['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        with open(results_dir / f"traces_{slug}.json", 'w', encoding='utf-8') as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

    with open(results_dir / 'retrieval_benchmark_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print(f"{'Cấu hình':<35} | {'Split':<5} | {'Hit@1':<12} | {'Hit@3':<12} | {'Context Rate':<14} | Latency")
    print("="*80)
    for s in all_summaries:
        print(f"{s['config_name']:<35} | {s['split']:<5} | {s['hit1']:<12} | {s['hit3']:<12} | {s['context_evidence_rate']:<14} | {s['median_latency_ms']}ms")
    print("="*80)


if __name__ == '__main__':
    main()
