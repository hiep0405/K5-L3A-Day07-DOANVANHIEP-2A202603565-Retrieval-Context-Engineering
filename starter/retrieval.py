"""Day 7 Retrieval Methods: Dense, Lexical (TF-IDF), Hybrid (RRF), and Enhanced Hybrid.
Supports metadata filtering and outputs chunks matching the Day 7 contract.
"""
from __future__ import annotations
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class TfidfRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.chunk_by_id = {c['id']: c for c in chunks}
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                match = True
                meta = chunk.get('metadata', {})
                for k, v in filters.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            score = sum(val * vector.get(term, 0.0) for term, val in query_vec.items())
            if score > 0:
                results.append({**chunk, 'score': round(score, 4), 'strategy': 'tfidf'})
        results.sort(key=lambda c: (-c['score'], c['id']))
        return results[:top_k]


class DenseRetriever:
    def __init__(self, chroma_path: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.col = self.client.get_collection('collection_strategy_b')

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self.model.encode(query, normalize_embeddings=True).tolist()
        where_clause = {k: v for k, v in filters.items()} if filters else None
        
        raw = self.col.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_clause
        )
        hits = []
        if not raw or not raw['ids'] or not raw['ids'][0]:
            return hits
            
        ids = raw['ids'][0]
        docs = raw['documents'][0]
        metas = raw['metadatas'][0]
        distances = raw['distances'][0] if 'distances' in raw and raw['distances'] else [0.0] * len(ids)
        
        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            score = round(1.0 - float(dist), 4)
            hits.append({
                'id': cid,
                'source_id': meta.get('document_id', 'bo-luat-lao-dong-2019'),
                'source': meta.get('source', 'bo_luat_lao_dong_2019_clean.pdf'),
                'page': int(meta.get('page', 1)),
                'text': doc,
                'score': score,
                'strategy': 'dense',
                'metadata': {
                    'chapter': meta.get('chapter', ''),
                    'strategy': meta.get('strategy', 'B')
                }
            })
        return hits


class HybridRetriever:
    """Combines Dense and Lexical retrieval using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, tfidf_retriever: TfidfRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        pool_k = max(top_k * 3, 10)
        dense_hits = self.dense.search(query, filters=filters, top_k=pool_k)
        tfidf_hits = self.tfidf.search(query, filters=filters, top_k=pool_k)

        scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit['id']
            chunk_map[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, hit in enumerate(tfidf_hits, start=1):
            cid = hit['id']
            chunk_map[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))
        results = []
        for cid in sorted_ids[:top_k]:
            c = chunk_map[cid].copy()
            c['score'] = round(scores[cid], 5)
            c['strategy'] = 'hybrid_rrf'
            results.append(c)
        return results


class ImprovedHybridRetriever:
    """Enhanced Hybrid with query keyword amplification (Law articles, numbers, key terms) & reranking."""
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid = hybrid_retriever

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        # 1. Gather candidate pool from Hybrid RRF
        candidates = self.hybrid.search(query, filters=filters, top_k=max(top_k * 3, 10))
        
        # 2. Extract salient law signals: Article numbers (Điều \d+), numeric thresholds (\d+ tháng/tuổi/ngày)
        article_match = re.search(r'điều\s+(\d+)', query.lower())
        article_num = article_match.group(1) if article_match else None
        
        num_patterns = re.findall(r'\b\d+\b', query)

        reranked = []
        for c in candidates:
            boost = 0.0
            c_text_lower = c['text'].lower()
            
            # Direct article match boost
            if article_num and f"điều {article_num}." in c_text_lower:
                boost += 0.05
            elif article_num and f"điều {article_num}" in c_text_lower:
                boost += 0.03
                
            # Exact numbers in query match in chunk
            matched_nums = sum(1 for n in num_patterns if n in c_text_lower)
            if num_patterns:
                boost += 0.01 * (matched_nums / len(num_patterns))

            final_score = c['score'] + boost
            c_copy = c.copy()
            c_copy['score'] = round(final_score, 5)
            c_copy['strategy'] = 'improved_hybrid_rerank'
            reranked.append(c_copy)

        reranked.sort(key=lambda x: (-x['score'], x['id']))
        return reranked[:top_k]
