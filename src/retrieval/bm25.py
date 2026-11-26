"""BM25 keyword search aur RRF (Reciprocal Rank Fusion) for Hybrid Retrieval."""

import math
import re
from collections import Counter


def tokenize(text):
    """Text ko lowercase words/code-identifiers mein todta hai (underscore intact rehta hai)."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """Chroma ke chunks par lightweight BM25 keyword search index."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        """
        Args:
            chunks: VectorStore.get_all_chunks() se aaye hue SearchResult objects.
            k1: term frequency saturation (default 1.5 — standard BM25 value).
            b: document length normalization (default 0.75 — lambe chunks ko penalize karta hai).
        """
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b

        # Har chunk ke liye path + text dono tokenize karte hain, taaki filename bhi match ho!
        self._doc_tokens = [
            tokenize(f"{c.file_path} {c.text}") for c in self._chunks
        ]
        self._doc_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_lens = [len(tokens) for tokens in self._doc_tokens]
        self._avg_len = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0

        # Har word kitne alag chunks mein aata hai (DF = Document Frequency)
        self._df = Counter()
        for freqs in self._doc_freqs:
            for term in freqs:
                self._df[term] += 1

    def _idf(self, term):
        """Inverse Document Frequency: rare word ko high weight, common word ko low weight."""
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query, k=5):
        """Query ke liye top-k chunks BM25 score ke hisaab se return karta hai."""
        if not self._chunks or k <= 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored = []
        for idx, chunk in enumerate(self._chunks):
            freqs = self._doc_freqs[idx]
            doc_len = self._doc_lens[idx]
            score = 0.0

            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                # Standard BM25 formula
                denom = tf + self._k1 * (1.0 - self._b + self._b * (doc_len / self._avg_len))
                score += idf * (tf * (self._k1 + 1.0)) / denom

            if score > 0.0:
                scored.append((score, chunk))

        # Sabse high BM25 score pehle
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]


def rrf_fuse(ranked_lists, k_rrf=60):
    """Reciprocal Rank Fusion: multiple ranked lists ko ek combined list mein merge karta hai.

    Args:
        ranked_lists: e.g. [vector_results, bm25_results] (har list best-first sorted).
        k_rrf: smoothing constant (default 60 — industry standard).

    Returns:
        Merged list of SearchResult objects (best RRF score first).
        Vector search wala original cosine `.score` preserve rehta hai!
    """
    rrf_scores = {}
    by_id = {}

    for results in ranked_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
            # Agar vector list pehle aayi hai, toh uska cosine score preserve rakho
            if cid not in by_id or chunk.score > by_id[cid].score:
                by_id[cid] = chunk

    ordered_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [by_id[cid] for cid in ordered_ids]