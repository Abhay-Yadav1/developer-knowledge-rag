"""Retriever: turn a question into a short list of relevant evidence.

    question -> embed -> search store (top-k) -> drop weak matches -> evidence

Two settings control it:

  * k          how many candidates to fetch ("top-k").
  * min_score  the SIMILARITY THRESHOLD. Search always returns the k closest
               chunks, even when none of them is actually related to the
               question. Chunks scoring below min_score are rejected, so the
               system can honestly say "I could not find this in the
               repository" instead of passing weak evidence to the LLM later.

IMPORTANT: the right min_score depends on the embedding model and on your
data. The default below is only a starting guess. Use the command-line tool
at the bottom of this file to look at real scores and tune it.
"""

import argparse
import re
import sys
from dataclasses import dataclass
import math
from src.ingestion.embedder import FastEmbedEmbedder, HashingEmbedder
from src.ingestion.vector_store import VectorStore

DEFAULT_K = 5
DEFAULT_MIN_SCORE = 0.65  # a starting guess for bge-small; calibrate on your data!
_CANDIDATE_MULTIPLIER = 4
DEFAULT_MMR_LAMBDA = 0.7  # 70% relevance + 30% diversity

def _terms(text):
    """Tokenize natural-language terms while keeping code identifiers intact."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))

def _cosine_similarity(vec_a, vec_b):
    """Do vectors ke beech cosine similarity (-1 se 1) nikalta hai.

    Formula: (A · B) / (||A|| * ||B||)
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select(candidates, k, lambda_mult=DEFAULT_MMR_LAMBDA):
    """Maximal Marginal Relevance: relevance aur diversity balance karke k chunks chunta hai.

    Har step par us candidate ko chunta hai jiska score sabse zyada ho:
        mmr = lambda * query_similarity - (1 - lambda) * max_similarity_to_already_selected
    """
    if not candidates or k <= 0:
        return []

    # Step 1: Sabse pehla slot hamesha top-ranked (most relevant) chunk ko milta hai.
    selected = [candidates[0]]
    remaining = list(candidates[1:])

    # Step 2: Baaki (k - 1) slots ek-ek karke bharo.
    while len(selected) < k and remaining:
        best_idx = 0
        best_mmr = float("-inf")

        for idx, cand in enumerate(remaining):
            # Ye candidate already selected chunks se kitna milta-julta (redundant) hai?
            max_sim_to_selected = max(
                _cosine_similarity(cand.embedding, chosen.embedding)
                for chosen in selected
            )
            mmr_score = (
                lambda_mult * cand.score
                - (1.0 - lambda_mult) * max_sim_to_selected
            )
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        # Jo winner mila usko selected mein daalo aur remaining se hatao.
        selected.append(remaining.pop(best_idx))

    return selected

def _rerank(query, candidates):
    """Use query term coverage as a small tie-break signal among vector hits.

    Cosine scores remain untouched because they drive the calibrated relevance
    threshold. The lexical signal only changes ordering when it is strong
    enough to outweigh a small cosine-score difference.
    """
    query_terms = _terms(query)
    if not query_terms:
        return candidates

    def rank(candidate):
        content_terms = _terms(candidate.text)
        path_terms = _terms(candidate.file_path)
        coverage = len(query_terms & content_terms) / len(query_terms)
        path_match = bool(query_terms & path_terms)
        # Coverage helps exact identifiers and API names without changing the
        # similarity score or making lexical-only matches pass the threshold.
        combined = candidate.score + 0.08 * coverage + (0.02 if path_match else 0.0)
        return combined, candidate.score

    return sorted(candidates, key=rank, reverse=True)


@dataclass(frozen=True)
class RetrievalResult:
    """What one retrieval returned."""

    query: str
    chunks: list    # SearchResult objects that passed the threshold, best first
    rejected: list  # SearchResult objects that scored below it, best first

    @property
    def found(self):
        """True if at least one chunk passed the threshold."""
        return len(self.chunks) > 0

    @property
    def best_score(self):
        """Highest score seen, even if it was rejected (None if store empty)."""
        candidates = self.chunks + self.rejected
        return max((c.score for c in candidates), default=None)


class Retriever:
    def __init__(
        self,
        embedder,
        store,
        k=DEFAULT_K,
        min_score=DEFAULT_MIN_SCORE,
        use_mmr=False,
        mmr_lambda=DEFAULT_MMR_LAMBDA,
    ):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1 (cosine similarity)")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        self._embedder = embedder
        self._store = store
        self._k = k
        self._min_score = min_score
        self._use_mmr = use_mmr
        self._mmr_lambda = mmr_lambda

    def retrieve(self, query):
        """Find the chunks most relevant to `query`."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # Wider pool fetch karo; agar MMR on hai toh Chroma se vectors bhi mangwao.
        pool_size = self._k * _CANDIDATE_MULTIPLIER
        candidates = self._store.search(
            self._embedder.embed_query(query),
            k=pool_size,
            include_embeddings=self._use_mmr,
        )
        candidates = _rerank(query, candidates)

        # MMR on ho toh diverse k chunks chuno, warna normal top-k slice karo.
        if self._use_mmr:
            candidates = _mmr_select(candidates, k=self._k, lambda_mult=self._mmr_lambda)
        else:
            candidates = candidates[:self._k]

        # Threshold filter bilkul pehle jaisa (pure cosine score par).
        kept = [c for c in candidates if c.score >= self._min_score]
        rejected = [c for c in candidates if c.score < self._min_score]
        return RetrievalResult(query=query, chunks=kept, rejected=rejected)

def format_context(result):
    """Write the kept chunks as numbered evidence blocks.

    Example output:

        [1] src/auth.py (lines 10-20)
        def reset_password(...): ...

        [2] README.md (lines 5-9)
        ...

    The numbers are what the LLM will refer to later, and the header holds the
    file path and line range we need for citations. Returns "" if nothing was
    found (check result.found first).
    """
    blocks = []
    for number, chunk in enumerate(result.chunks, start=1):
        header = f"[{number}] {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask a question against an indexed database and see scores (for tuning)."
    )
    parser.add_argument("question", help="the question to search for")
    parser.add_argument("--db", default="chroma_db", help="database folder (default: chroma_db)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"candidates to fetch (default: {DEFAULT_K})")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"similarity threshold (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--demo-embedder", action="store_true",
                        help="use the simple word-hashing embedder (must match how the db was indexed)")
    args = parser.parse_args(argv)

    try:
        embedder = HashingEmbedder() if args.demo_embedder else FastEmbedEmbedder()
        store = VectorStore(args.db, embedding_model=embedder.name)
        if store.count() == 0:
            print(f"The database '{args.db}' is empty. Run src.ingestion.indexer first.")
            return 1

        retriever = Retriever(embedder, store, k=args.k, min_score=args.min_score)
        result = retriever.retrieve(args.question)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    print(f"Question: {args.question}")
    print(f"Threshold: {args.min_score}   (k = {args.k})\n")

    rows = [(c, "kept") for c in result.chunks] + [(c, "below threshold") for c in result.rejected]
    for rank, (chunk, status) in enumerate(rows, start=1):
        first_line = chunk.text.strip().splitlines()[0][:90] if chunk.text.strip() else ""
        print(f"#{rank}  score {chunk.score:.3f}  [{status}]  "
              f"{chunk.file_path} L{chunk.start_line}-L{chunk.end_line}")
        print(f"      {first_line}")

    print()
    if result.found:
        print(f"Found {len(result.chunks)} relevant chunk(s).")
    else:
        print(f"Nothing passed the threshold (best score {result.best_score:.3f}). "
              "The assistant should say it could not find this in the repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
