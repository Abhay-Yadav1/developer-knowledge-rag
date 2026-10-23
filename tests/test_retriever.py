"""Tests for src/retrieval/retriever.py (uses the model-free HashingEmbedder)."""

import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import HashingEmbedder
from src.ingestion.indexer import index_repository
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever, RetrievalResult, format_context, main

MODEL = "hashing-embedder"


def make_chunk(file_path, text, start=1, end=3):
    return Chunk(f"{file_path}:{start}-{end}", file_path, start, end, "py", text)


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    chunks = [
        make_chunk("auth.py", "reset the user password and invalidate all sessions", 10, 20),
        make_chunk("billing.py", "render the invoice as a pdf file for the customer", 1, 8),
        make_chunk("README.md", "install the dependencies and run the server", 1, 4),
    ]
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store


def test_returns_relevant_chunk_first_with_metadata(embedder, store):
    result = Retriever(embedder, store, min_score=0.3).retrieve("how do I reset a user password")

    assert result.found
    best = result.chunks[0]
    assert (best.file_path, best.start_line, best.end_line) == ("auth.py", 10, 20)
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_off_topic_question_finds_nothing(embedder, store):
    result = Retriever(embedder, store, min_score=0.3).retrieve("quantum entanglement physics experiment")

    assert not result.found
    assert result.chunks == []
    assert len(result.rejected) > 0           # candidates existed but were too weak
    assert result.best_score < 0.3


def test_weak_matches_are_moved_to_rejected(embedder, store):
    # Every chunk contains the word "the", so even unrelated chunks get a
    # modest score (about 0.3 to 0.4). A threshold of 0.5 separates them.
    result = Retriever(embedder, store, k=3, min_score=0.5).retrieve("reset the password")

    assert [c.file_path for c in result.chunks] == ["auth.py"]
    assert {c.file_path for c in result.rejected} == {"billing.py", "README.md"}
    assert all(c.score < 0.5 for c in result.rejected)


def test_threshold_is_inclusive(embedder, store):
    exact_text = "render the invoice as a pdf file for the customer"
    top_score = Retriever(embedder, store, k=1, min_score=-1).retrieve(exact_text).chunks[0].score

    kept = Retriever(embedder, store, k=1, min_score=top_score).retrieve(exact_text)

    assert kept.found  # score == threshold counts as a pass


def test_lowest_threshold_keeps_all_k_candidates(embedder, store):
    result = Retriever(embedder, store, k=3, min_score=-1.0).retrieve("anything at all")

    assert len(result.chunks) == 3
    assert result.rejected == []


def test_k_limits_number_of_candidates(embedder, store):
    result = Retriever(embedder, store, k=2, min_score=-1.0).retrieve("server")

    assert len(result.chunks) + len(result.rejected) == 2


def test_empty_store_returns_nothing_without_crashing(tmp_path, embedder):
    empty = VectorStore(tmp_path / "empty_db", embedding_model=MODEL)

    result = Retriever(embedder, empty).retrieve("anything")

    assert not result.found
    assert result.best_score is None


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_question_raises_clear_error(embedder, store, query):
    with pytest.raises(ValueError):
        Retriever(embedder, store).retrieve(query)


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -1}, {"min_score": 1.5}, {"min_score": -2}])
def test_invalid_settings_are_rejected(embedder, store, kwargs):
    with pytest.raises(ValueError):
        Retriever(embedder, store, **kwargs)


# ---- format_context ----

def test_format_context_numbers_chunks_and_shows_citation_info(embedder, store):
    result = Retriever(embedder, store, k=3, min_score=-1.0).retrieve("reset the user password")

    context = format_context(result)

    assert context.startswith("[1] auth.py (lines 10-20)\nreset the user password")
    assert "[2] " in context and "[3] " in context
    assert context.index("[1]") < context.index("[2]") < context.index("[3]")


def test_format_context_is_empty_when_nothing_found():
    result = RetrievalResult(query="q", chunks=[], rejected=[])

    assert format_context(result) == ""
    assert not result.found


# ---- command-line tool ----

def test_command_line_tool_shows_scores_and_verdict(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("# Reset the user password and invalidate all sessions\n")
    (repo / "billing.py").write_text("# Render the invoice as a pdf file for the customer\n")
    db = tmp_path / "db"
    index_repository(repo, HashingEmbedder(), VectorStore(db, embedding_model=MODEL))

    code = main(["reset the user password", "--db", str(db), "--min-score", "0.3", "--demo-embedder"])

    output = capsys.readouterr().out
    assert code == 0
    assert "[kept]" in output and "auth.py" in output
    assert "[below threshold]" in output
    assert "Found 1 relevant chunk" in output


def test_command_line_tool_says_when_nothing_is_relevant(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("# Reset the user password and invalidate all sessions\n")
    db = tmp_path / "db"
    index_repository(repo, HashingEmbedder(), VectorStore(db, embedding_model=MODEL))

    main(["quantum entanglement physics", "--db", str(db), "--min-score", "0.3", "--demo-embedder"])

    assert "could not find this" in capsys.readouterr().out


def test_command_line_tool_reports_empty_database(tmp_path, capsys):
    code = main(["anything", "--db", str(tmp_path / "nothing_here"), "--demo-embedder"])

    assert code == 1
    assert "empty" in capsys.readouterr().out