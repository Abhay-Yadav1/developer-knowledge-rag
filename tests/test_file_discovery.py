"""Tests for src/ingestion/file_discovery.py

Each test builds a small FAKE repository inside a temporary folder (pytest's
`tmp_path`), so the tests never touch your real projects and are deleted
automatically afterwards.
"""

import pytest

from src.ingestion.file_discovery import discover_files


def make_files(root, names):
    """Helper: create empty files (and any missing folders) under `root`."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")


def test_finds_supported_files_including_nested_ones(tmp_path):
    make_files(tmp_path, [
        "Main.java", "app.py", "README.md", "notes.txt",
        "config.yml", "config.yaml",
        "src/deep/nested/module.py",
    ])

    assert discover_files(tmp_path) == [
        "Main.java",
        "README.md",
        "app.py",
        "config.yaml",
        "config.yml",
        "notes.txt",
        "src/deep/nested/module.py",
    ]


def test_ignores_unsupported_file_types(tmp_path):
    make_files(tmp_path, ["app.py", "logo.png", "data.json", "script.js", "Makefile"])

    assert discover_files(tmp_path) == ["app.py"]


@pytest.mark.parametrize("folder", [
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".next",
])
def test_skips_excluded_folders(tmp_path, folder):
    make_files(tmp_path, ["app.py", f"{folder}/hidden.py", f"src/{folder}/nested.py"])

    assert discover_files(tmp_path) == ["app.py"]


def test_skips_secret_files(tmp_path):
    make_files(tmp_path, [
        "app.py",
        ".env", ".env.local", ".env.production",
        "server.pem", "private.key", "id_rsa",
        "credentials.yml", "secrets.yaml",
    ])

    assert discover_files(tmp_path) == ["app.py"]


def test_never_opens_secret_files(tmp_path, monkeypatch):
    """Discovery must look at names only, never at file contents."""
    make_files(tmp_path, ["app.py", "secrets.yaml"])
    (tmp_path / "secrets.yaml").write_text("PASSWORD=super-secret")

    opened = []
    real_open = open

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", spy_open)

    discover_files(tmp_path)

    assert opened == []


def test_works_with_different_repository_folders(tmp_path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    make_files(repo_a, ["a.py"])
    make_files(repo_b, ["b.md", "sub/c.java"])

    assert discover_files(repo_a) == ["a.py"]
    assert discover_files(repo_b) == ["b.md", "sub/c.java"]


def test_accepts_string_paths(tmp_path):
    make_files(tmp_path, ["app.py"])

    assert discover_files(str(tmp_path)) == ["app.py"]


def test_result_is_sorted_and_repeatable(tmp_path):
    make_files(tmp_path, ["z.py", "a.py", "m/b.py", "B.md"])

    first = discover_files(tmp_path)
    second = discover_files(tmp_path)

    assert first == second
    assert first == sorted(first)


def test_missing_folder_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "does_not_exist")


def test_path_that_is_a_file_raises_clear_error(tmp_path):
    make_files(tmp_path, ["app.py"])

    with pytest.raises(NotADirectoryError):
        discover_files(tmp_path / "app.py")


def test_does_not_follow_symlinked_folders(tmp_path):
    outside = tmp_path / "outside"
    repo = tmp_path / "repo"
    make_files(outside, ["leaked.py"])
    make_files(repo, ["app.py"])

    try:
        (repo / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not available here (on Windows they need special rights)")

    assert discover_files(repo) == ["app.py"]