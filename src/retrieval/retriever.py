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
import sys
from dataclasses import dataclass

from src.ingestion.embedder import FastEmbedEmbedder, HashingEmbedder
from src.ingestion.vector_store import VectorStore

DEFAULT_K = 5
DEFAULT_MIN_SCORE = 0.5  # a starting guess for bge-small; calibrate on your data!


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
    def __init__(self, embedder, store, k=DEFAULT_K, min_score=DEFAULT_MIN_SCORE):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1 (cosine similarity)")
        self._embedder = embedder
        self._store = store
        self._k = k
        self._min_score = min_score

    def retrieve(self, query):
        """Find the chunks most relevant to `query`."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # Questions use embed_query (not embed_documents): see embedder.py.
        candidates = self._store.search(self._embedder.embed_query(query), k=self._k)

        # Candidates arrive best-first, so both lists stay sorted.
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