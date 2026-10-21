"""Tests for src/ingestion/indexer.py: the whole pipeline on fake repositories."""

import pytest

from src.ingestion.chunker import chunk_repository
from src.ingestion.embedder import HashingEmbedder
from src.ingestion.indexer import index_repository, main
from src.ingestion.vector_store import VectorStore

MODEL = "hashing-embedder"


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text(
        "# Reset the user password and invalidate all sessions\n"
        "def reset_flow():\n    pass\n"
    )
    (root / "src" / "billing.py").write_text(
        "# Render the invoice as a pdf file for the customer\n"
        "def invoice_flow():\n    pass\n"
    )
    (root / "README.md").write_text("# Project\n\nInstall the dependencies and run the server.\n")
    (root / ".env").write_text("PASSWORD=super-secret")
    return root


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "db", embedding_model=MODEL)


def test_indexes_every_chunk_of_the_repository(repo, embedder, store):
    stored = index_repository(repo, embedder, store)

    assert stored == len(chunk_repository(repo))
    assert store.count() == stored
    assert stored > 0


def test_question_finds_the_relevant_file(repo, embedder, store):
    index_repository(repo, embedder, store)

    top = store.search(embedder.embed_query("how do I reset a user password"), k=1)[0]

    assert top.file_path == "src/auth.py"


def test_secret_files_never_reach_the_database(repo, embedder, store):
    index_repository(repo, embedder, store)

    everything = store.search(embedder.embed_query("password super secret"), k=100)

    assert all("super-secret" not in r.text for r in everything)
    assert all(r.file_path != ".env" for r in everything)


def test_rebuild_removes_chunks_of_deleted_files(repo, embedder, store):
    index_repository(repo, embedder, store)
    (repo / "src" / "billing.py").unlink()

    index_repository(repo, embedder, store)

    everything = store.search(embedder.embed_query("invoice"), k=100)
    assert all(r.file_path != "src/billing.py" for r in everything)
    assert store.count() == len(chunk_repository(repo))


def test_running_twice_without_rebuild_does_not_duplicate(repo, embedder, store):
    index_repository(repo, embedder, store, rebuild=False)
    first_count = store.count()

    index_repository(repo, embedder, store, rebuild=False)

    assert store.count() == first_count


def test_bad_repository_path_does_not_wipe_existing_index(repo, tmp_path, embedder, store):
    index_repository(repo, embedder, store)
    before = store.count()

    with pytest.raises(FileNotFoundError):
        index_repository(tmp_path / "does_not_exist", embedder, store)

    assert store.count() == before


def test_empty_repository_indexes_nothing(tmp_path, embedder, store):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert index_repository(empty, embedder, store) == 0
    assert store.count() == 0


def test_command_line_tool_indexes_and_searches(repo, tmp_path, capsys):
    db = tmp_path / "cli_db"

    main([str(repo), "--db", str(db), "--query", "reset a user password",
          "--k", "2", "--demo-embedder"])

    output = capsys.readouterr().out
    assert "Indexed" in output
    assert "src/auth.py" in output
    assert "super-secret" not in output