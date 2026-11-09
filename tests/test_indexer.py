"""Tests for src/ingestion/indexer.py: the whole pipeline on fake repositories."""

import pytest

from src.ingestion.chunker import Chunk, chunk_repository
from src.ingestion.embedder import HashingEmbedder
from src.ingestion.indexer import embedding_text, index_repository, main
from src.ingestion.vector_store import VectorStore

MODEL = "hashing-embedder"


class RecordingEmbedder:
    """Wraps HashingEmbedder and records every text passed to embed_documents."""

    def __init__(self):
        self._inner = HashingEmbedder()
        self.name = self._inner.name
        self.embedded_texts = []

    def embed_documents(self, texts):
        self.embedded_texts.extend(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


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


# ---- path header (embedding_text) ----

def test_embedding_text_formats_file_header():
    chunk = Chunk("graphs/state1.py:1-5", "graphs/state1.py", 1, 5, "py", "class AgentState:\n    pass")

    assert embedding_text(chunk) == "File: graphs/state1.py\nclass AgentState:\n    pass"
    assert embedding_text(chunk, include_path=False) == "class AgentState:\n    pass"


def test_embeds_path_header_but_stores_clean_chunk_text(repo, store):
    recorder = RecordingEmbedder()

    index_repository(repo, recorder, store)

    assert any(t.startswith("File: src/auth.py\n") for t in recorder.embedded_texts)
    results = store.search(recorder.embed_query("reset password"), k=10)
    assert all(not r.text.startswith("File: ") for r in results)


def test_path_header_can_be_switched_off(repo, store):
    recorder = RecordingEmbedder()

    index_repository(repo, recorder, store, include_path=False)

    assert all(not t.startswith("File: ") for t in recorder.embedded_texts)

def test_file_name_alone_makes_chunk_findable(tmp_path, embedder, store):
    root = tmp_path / "name_only_repo"
    root.mkdir()
    # Body never mentions "requirements" or "dependencies"
    (root / "requirements.txt").write_text("fastapi==0.110.0\nuvicorn==0.29.0\n")
    (root / "notes.txt").write_text("meeting agenda for monday morning\n")

    index_repository(root, embedder, store, include_path=True)

    top = store.search(embedder.embed_query("requirements"), k=1)[0]
    assert top.file_path == "requirements.txt"

def test_cli_no_path_header_flag_disables_header(tmp_path, monkeypatch):
    root = tmp_path / "cli_repo"
    root.mkdir()
    (root / "requirements.txt").write_text("fastapi==0.110.0\n")
    db = tmp_path / "cli_no_header_db"
    recorder = RecordingEmbedder()
    monkeypatch.setattr("src.ingestion.indexer.HashingEmbedder", lambda: recorder)

    main([str(root), "--db", str(db), "--demo-embedder", "--no-path-header"])

    assert len(recorder.embedded_texts) == 1
    assert not recorder.embedded_texts[0].startswith("File: ")