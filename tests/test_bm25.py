"""Tests for BM25 keyword search and RRF fusion."""

from src.ingestion.vector_store import SearchResult
from src.retrieval.bm25 import BM25Index, rrf_fuse


def _res(chunk_id, file_path, text, score=0.0):
    return SearchResult(chunk_id, file_path, 1, 5, "py", text, score)


def test_bm25_finds_exact_filename_and_identifier():
    chunks = [
        _res("a", "README.md", "overview of the whole project and workflow"),
        _res("b", "graphs/state1.py", "class AgentState: pass"),
        _res("c", "main.py", "run the application"),
    ]
    index = BM25Index(chunks)

    hits = index.search("where is state1 defined", k=2)

    assert hits[0].file_path == "graphs/state1.py"


def test_bm25_returns_empty_for_unmatched_query():
    chunks = [_res("a", "auth.py", "reset password")]
    index = BM25Index(chunks)

    assert index.search("quantum physics", k=5) == []


def test_rrf_boosts_chunk_ranked_high_in_both_lists():
    a = _res("a", "README.md", "overview", score=0.80)
    b = _res("b", "graphs/state1.py", "AgentState", score=0.72)
    c = _res("c", "main.py", "entry point", score=0.75)

    # Vector list: a (#1), c (#2), b (#3)
    # BM25 list:   b (#1), a (#2)
    # Dono mein top-2/3 aane se 'a' aur 'b' upar aane chahiye, aur cosine score preserve rehna chahiye
    fused = rrf_fuse([[a, c, b], [b, a]])

    assert [x.chunk_id for x in fused[:2]] == ["a", "b"]
    # Cosine score preserve hua (0.0 se overwrite nahi hua)
    assert fused[1].score == 0.72