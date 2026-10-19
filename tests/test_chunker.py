"""Tests for src/ingestion/chunker.py"""

import pytest

from src.ingestion.chunker import chunk_document, chunk_repository
from src.ingestion.loader import Document


def make_doc(text, path="src/app.py"):
    return Document(file_path=path, text=text)


def numbered_lines(count, width=20):
    """`count` lines like 'line 001 xxxxxxxx', each about `width` chars long."""
    return [f"line {i:03d} ".ljust(width, "x") for i in range(1, count + 1)]


# ---- basic behaviour ------------------------------------------------------

def test_short_file_becomes_one_chunk_with_correct_metadata():
    chunks = chunk_document(make_doc("a = 1\nb = 2\nc = 3"))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.file_path == "src/app.py"
    assert (chunk.start_line, chunk.end_line) == (1, 3)
    assert chunk.chunk_id == "src/app.py:1-3"
    assert chunk.file_type == "py"
    assert chunk.text == "a = 1\nb = 2\nc = 3"


@pytest.mark.parametrize("text", ["", "\n\n", "   \n  \n"])
def test_empty_or_blank_files_produce_no_chunks(text):
    assert chunk_document(make_doc(text)) == []


def test_windows_line_endings_are_handled():
    chunks = chunk_document(make_doc("a = 1\r\nb = 2\r\n"))

    assert len(chunks) == 1
    assert chunks[0].text == "a = 1\nb = 2"  # no stray \r characters


# ---- the most important property: line numbers must be truthful ----------

def test_every_chunk_text_matches_its_line_range():
    lines = numbered_lines(100)
    chunks = chunk_document(make_doc("\n".join(lines)), max_chars=300)

    assert len(chunks) > 1
    for chunk in chunks:
        expected = "\n".join(lines[chunk.start_line - 1:chunk.end_line])
        assert chunk.text == expected


def test_every_non_blank_line_is_covered_by_some_chunk():
    lines = numbered_lines(60)
    lines[10] = ""   # sprinkle in some blank lines
    lines[30] = ""
    chunks = chunk_document(make_doc("\n".join(lines)), max_chars=250)

    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start_line, chunk.end_line + 1))

    for number, line in enumerate(lines, start=1):
        if line.strip():
            assert number in covered


def test_chunks_respect_max_chars():
    lines = numbered_lines(80)
    chunks = chunk_document(make_doc("\n".join(lines)), max_chars=300)

    assert all(len(chunk.text) <= 300 for chunk in chunks)


# ---- overlap and blank-line breaks ---------------------------------------

def test_neighbouring_chunks_overlap_by_requested_lines():
    lines = numbered_lines(50)
    chunks = chunk_document(make_doc("\n".join(lines)), max_chars=300, overlap_lines=2)

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_line == previous.end_line - 2 + 1


def test_zero_overlap_means_no_repeated_lines():
    lines = numbered_lines(50)
    chunks = chunk_document(make_doc("\n".join(lines)), max_chars=300, overlap_lines=0)

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_line == previous.end_line + 1


def test_prefers_cutting_at_a_blank_line():
    function_a = ["def a():"] + [f"    step_{i} = {i}" for i in range(8)]
    function_b = ["def b():"] + [f"    step_{i} = {i}" for i in range(8)]
    text = "\n".join(function_a + [""] + function_b)

    # max_chars is large enough for function_a + a bit of function_b, so a
    # naive splitter would cut function_b in half.
    chunks = chunk_document(make_doc(text), max_chars=170, overlap_lines=0)

    assert chunks[0].text.endswith("step_7 = 7")
    assert chunks[1].text.startswith("def b():")


# ---- edge cases -----------------------------------------------------------

def test_single_line_longer_than_limit_does_not_loop_forever():
    text = "x" * 5000 + "\nshort line"
    chunks = chunk_document(make_doc(text), max_chars=100)

    assert chunks[0].text == "x" * 5000   # kept whole; we never cut mid-line
    assert chunks[-1].text.endswith("short line")


def test_overlap_larger_than_chunk_still_terminates():
    lines = numbered_lines(30)
    chunks = chunk_document(make_doc("\n".join(lines)), max_chars=60, overlap_lines=10)

    assert chunks[-1].end_line == 30


def test_results_are_repeatable():
    doc = make_doc("\n".join(numbered_lines(70)))

    assert chunk_document(doc, max_chars=300) == chunk_document(doc, max_chars=300)


@pytest.mark.parametrize("kwargs", [{"max_chars": 0}, {"max_chars": -5}, {"overlap_lines": -1}])
def test_invalid_settings_raise_clear_errors(kwargs):
    with pytest.raises(ValueError):
        chunk_document(make_doc("a = 1"), **kwargs)


# ---- whole-repository integration ----------------------------------------

def test_chunk_repository_end_to_end(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 1\n")
    (tmp_path / "README.md").write_text("# Title\n\nSome docs.")
    (tmp_path / ".env").write_text("PASSWORD=super-secret")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.py").write_text("x = 1")

    chunks = chunk_repository(tmp_path)

    assert [c.chunk_id for c in chunks] == ["README.md:1-3", "src/app.py:1-2"]
    assert all("super-secret" not in c.text for c in chunks)


def test_chunk_ids_are_unique(tmp_path):
    (tmp_path / "a.py").write_text("\n".join(numbered_lines(100)))
    (tmp_path / "b.py").write_text("\n".join(numbered_lines(100)))

    chunks = chunk_repository(tmp_path, max_chars=200)
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))