"""Tests for src/evaluation/runner.py"""

import json

import pytest

from src.evaluation.runner import (
    format_report,
    load_questions,
    main,
    question_status,
    run_retrieval,
)
from src.ingestion.embedder import HashingEmbedder
from src.ingestion.indexer import index_repository
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever

MODEL = "hashing-embedder"


def write_questions(path, questions):
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return path


# ---- loading and validating the question file ----

def test_loads_valid_questions(tmp_path):
    f = write_questions(tmp_path / "q.json", [
        {"id": "a", "question": "How?", "expected_files": ["x.py"]},
        {"id": "u", "question": "Why?", "expected_files": []},
    ])

    assert [q["id"] for q in load_questions(f)] == ["a", "u"]


def test_accepts_files_saved_with_a_windows_bom(tmp_path):
    f = tmp_path / "q.json"
    f.write_bytes(b"\xef\xbb\xbf" + json.dumps(
        {"questions": [{"id": "a", "question": "How?", "expected_files": ["x.py"]}]}).encode())

    assert len(load_questions(f)) == 1


@pytest.mark.parametrize("questions, message", [
    ([{"question": "How?", "expected_files": ["x.py"]}], "missing \"id\""),
    ([{"id": "a", "expected_files": ["x.py"]}], "missing \"question\""),
    ([{"id": "a", "question": "How?"}], "missing \"expected_files\""),
    ([{"id": "a", "question": "  ", "expected_files": ["x.py"]}], "empty"),
    ([{"id": "a", "question": "How?", "expected_files": "x.py"}], "must be a list"),
    ([{"id": "a", "question": "1?", "expected_files": ["x.py"]},
      {"id": "a", "question": "2?", "expected_files": ["y.py"]}], "Duplicate"),
    ([{"id": "u", "question": "Why?", "expected_files": []}], "answerable"),
    ([], "non-empty"),
])
def test_invalid_question_files_give_clear_errors(tmp_path, questions, message):
    f = write_questions(tmp_path / "q.json", questions)

    with pytest.raises(ValueError, match=message):
        load_questions(f)


def test_missing_and_malformed_files_give_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_questions(tmp_path / "nope.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_questions(bad)


def test_the_starter_question_file_in_the_repo_is_valid():
    from pathlib import Path
    starter = Path("evaluation/questions_intelligent_db_query_agent.json")
    if not starter.exists():
        pytest.skip("starter file not present")

    questions = load_questions(starter)

    assert any(not q["expected_files"] for q in questions)      # has unanswerable ones
    assert any(q["expected_files"] for q in questions)


# ---- running against a real (fake-repo) index ----

@pytest.fixture
def indexed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("# Reset the user password and invalidate all sessions\n")
    (repo / "billing.py").write_text("# Render the invoice as a pdf file for the customer\n")
    (repo / "README.md").write_text("Install the dependencies and run the server.\n")
    embedder = HashingEmbedder()
    store = VectorStore(tmp_path / "db", embedding_model=MODEL)
    index_repository(repo, embedder, store)
    return tmp_path / "db", embedder, store


QUESTIONS = [
    {"id": "q1", "question": "how to reset a user password", "expected_files": ["auth.py"]},
    {"id": "q2", "question": "render invoice pdf for customer", "expected_files": ["billing.py"]},
    {"id": "u1", "question": "quantum entanglement physics experiment", "expected_files": []},
]


def test_run_retrieval_returns_all_candidates_ranked(indexed):
    _, embedder, store = indexed

    results = run_retrieval(QUESTIONS, Retriever(embedder, store, k=3, min_score=-1.0))

    assert [r.id for r in results] == ["q1", "q2", "u1"]
    assert results[0].candidates[0].file_path == "auth.py"
    assert all(len(r.candidates) == 3 for r in results)
    assert results[2].expected_files == ()


def test_question_status_labels(indexed):
    _, embedder, store = indexed
    q1, q2, u1 = run_retrieval(QUESTIONS, Retriever(embedder, store, k=3, min_score=-1.0))

    assert question_status(q1, 0.3) == "HIT @1"
    assert question_status(q1, 0.99) == "MISS (nothing passed)"
    assert question_status(u1, 0.5) == "refused OK"
    assert question_status(u1, -1.0) == "WRONGLY ANSWERED"


def test_report_contains_summary_sweep_and_suggestion(indexed):
    _, embedder, store = indexed
    results = run_retrieval(QUESTIONS, Retriever(embedder, store, k=3, min_score=-1.0))

    report = format_report(results, k=3, min_score=0.5)

    assert "3 questions (2 answerable, 1 unanswerable)" in report
    assert "HIT @1" in report and "refused OK" in report
    assert "Threshold sweep:" in report
    assert "Suggested threshold:" in report


def test_command_line_tool_prints_report_and_saves_json(indexed, tmp_path, capsys):
    db, _, _ = indexed
    questions = write_questions(tmp_path / "q.json", QUESTIONS)
    output = tmp_path / "results.json"

    code = main([str(questions), "--db", str(db), "--k", "3", "--min-score", "0.5",
                 "--demo-embedder", "--output", str(output)])

    assert code == 0
    assert "Suggested threshold" in capsys.readouterr().out
    saved = json.loads(output.read_text())
    assert saved["summary"]["answerable"] == 2
    assert len(saved["sweep"]) == 11
    assert [q["id"] for q in saved["questions"]] == ["q1", "q2", "u1"]


def test_command_line_tool_reports_bad_input_and_empty_database(tmp_path, capsys):
    code = main([str(tmp_path / "missing.json"), "--db", str(tmp_path / "db"), "--demo-embedder"])
    assert code == 1 and "not found" in capsys.readouterr().out

    questions = write_questions(tmp_path / "q.json", QUESTIONS)
    code = main([str(questions), "--db", str(tmp_path / "empty_db"), "--demo-embedder"])
    assert code == 1 and "empty" in capsys.readouterr().out