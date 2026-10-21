"""Tests for src/ingestion/vector_store.py (uses a real Chroma database in a
temporary folder, and the model-free HashingEmbedder)."""

import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import HashingEmbedder
from src.ingestion.vector_store import VectorStore

MODEL = "hashing-embedder"


def make_chunk(file_path, text, start=1, end=3, file_type="py"):
    return Chunk(
        chunk_id=f"{file_path}:{start}-{end}",
        file_path=file_path,
        start_line=start,
        end_line=end,
        file_type=file_type,
        text=text,
    )


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def chunks():
    return [
        make_chunk("auth.py", "reset the user password and invalidate all sessions", 10, 20),
        make_chunk("billing.py", "render the invoice as a pdf file for the customer", 1, 8),
        make_chunk("README.md", "install the dependencies and run the server", 1, 4, "md"),
    ]


def fill(store, embedder, chunks):
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))


def test_new_store_is_empty(tmp_path):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)

    assert store.count() == 0
    assert store.search(HashingEmbedder().embed_query("anything")) == []


def test_search_returns_best_match_first_with_full_metadata(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    fill(store, embedder, chunks)

    results = store.search(embedder.embed_query("how to reset a user password"), k=3)

    best = results[0]
    assert best.file_path == "auth.py"
    assert (best.start_line, best.end_line) == (10, 20)
    assert best.chunk_id == "auth.py:10-20"
    assert best.file_type == "py"
    assert best.text == "reset the user password and invalidate all sessions"
    assert isinstance(best.start_line, int)


def test_results_are_ordered_by_similarity(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    fill(store, embedder, chunks)

    results = store.search(embedder.embed_query("invoice pdf customer"), k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].file_path == "billing.py"


def test_searching_with_exact_text_gives_score_near_one(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    fill(store, embedder, chunks)

    top = store.search(embedder.embed_query(chunks[1].text), k=1)[0]

    assert top.score == pytest.approx(1.0, abs=1e-3)


def test_k_limits_results_and_larger_k_than_stored_is_fine(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    fill(store, embedder, chunks)
    query = embedder.embed_query("server")

    assert len(store.search(query, k=1)) == 1
    assert len(store.search(query, k=50)) == 3


def test_invalid_k_raises_error(tmp_path, embedder):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)

    with pytest.raises(ValueError):
        store.search(embedder.embed_query("x"), k=0)


def test_adding_same_chunks_again_does_not_create_duplicates(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)

    fill(store, embedder, chunks)
    fill(store, embedder, chunks)

    assert store.count() == 3


def test_same_chunk_id_overwrites_old_text(tmp_path, embedder):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    fill(store, embedder, [make_chunk("a.py", "old version of the text")])
    fill(store, embedder, [make_chunk("a.py", "brand new text")])

    result = store.search(embedder.embed_query("brand new text"), k=1)[0]

    assert store.count() == 1
    assert result.text == "brand new text"


def test_mismatched_chunks_and_embeddings_raise_error(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)

    with pytest.raises(ValueError):
        store.add_chunks(chunks, embedder.embed_documents(["only one"]))


def test_data_survives_reopening_the_database(tmp_path, embedder, chunks):
    db = tmp_path / "db"
    fill(VectorStore(db, embedding_model=MODEL), embedder, chunks)

    reopened = VectorStore(db, embedding_model=MODEL)

    assert reopened.count() == 3
    assert reopened.search(embedder.embed_query("install server"), k=1)[0].file_path == "README.md"


def test_clear_removes_everything_and_store_stays_usable(tmp_path, embedder, chunks):
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    fill(store, embedder, chunks)

    store.clear()
    assert store.count() == 0

    fill(store, embedder, chunks[:1])
    assert store.count() == 1


def test_opening_a_store_with_a_different_model_is_refused(tmp_path, embedder, chunks):
    db = tmp_path / "db"
    fill(VectorStore(db, embedding_model="model-a"), embedder, chunks)

    with pytest.raises(ValueError, match="model-a"):
        VectorStore(db, embedding_model="model-b")


def test_different_collection_names_are_independent(tmp_path, embedder, chunks):
    db = tmp_path / "db"
    first = VectorStore(db, embedding_model=MODEL, collection_name="repo-one")
    second = VectorStore(db, embedding_model=MODEL, collection_name="repo-two")

    fill(first, embedder, chunks)

    assert first.count() == 3
    assert second.count() == 0