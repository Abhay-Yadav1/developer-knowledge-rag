"""Tests for src/ingestion/loader.py (using fake files in temporary folders)."""

from src.ingestion.loader import Document, load_repository, read_text_file


def test_reads_plain_utf8_text(tmp_path):
    f = tmp_path / "a.txt"
    # write_bytes writes EXACTLY these bytes. (write_text would turn "\n" into
    # "\r\n" on Windows, which made this test pass on Linux but fail on Windows.)
    f.write_bytes(b"hello\nworld")

    assert read_text_file(f) == "hello\nworld"


def test_keeps_windows_line_endings_as_stored(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello\r\nworld")

    # The loader returns the file exactly as stored; the chunker is the one
    # that handles \r\n (it uses splitlines()).
    assert read_text_file(f) == "hello\r\nworld"


def test_removes_windows_bom_marker(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"\xef\xbb\xbfhello")  # UTF-8 BOM + "hello"

    assert read_text_file(f) == "hello"


def test_skips_binary_looking_files(tmp_path):
    f = tmp_path / "fake.txt"
    f.write_bytes(b"abc\x00def")

    assert read_text_file(f) is None


def test_skips_files_that_are_too_large(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)

    assert read_text_file(f, max_bytes=50) is None
    assert read_text_file(f, max_bytes=500) == "x" * 100


def test_skips_files_that_are_not_utf8(tmp_path):
    f = tmp_path / "old.txt"
    f.write_bytes("café".encode("cp1252"))  # byte 0xE9 is invalid UTF-8 here

    assert read_text_file(f) is None


def test_missing_file_returns_none(tmp_path):
    assert read_text_file(tmp_path / "nope.txt") is None


def test_load_repository_returns_documents_with_relative_paths(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')")
    (tmp_path / "README.md").write_text("# Title")

    docs = load_repository(tmp_path)

    assert docs == [
        Document("README.md", "# Title"),
        Document("src/app.py", "print('hi')"),
    ]


def test_load_repository_never_reads_secret_files(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / ".env").write_text("PASSWORD=super-secret")
    (tmp_path / "secrets.yaml").write_text("key: super-secret")

    opened = []
    real_read_bytes = type(tmp_path).read_bytes

    def spy_read_bytes(self):
        opened.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(type(tmp_path), "read_bytes", spy_read_bytes)

    docs = load_repository(tmp_path)

    assert [d.file_path for d in docs] == ["app.py"]
    assert opened == ["app.py"]  # the secret files were never opened


def test_load_repository_skips_unreadable_files_without_crashing(tmp_path):
    (tmp_path / "good.py").write_text("x = 1")
    (tmp_path / "binary.txt").write_bytes(b"\x00\x01\x02")

    docs = load_repository(tmp_path)

    assert [d.file_path for d in docs] == ["good.py"]