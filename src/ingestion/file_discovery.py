"""File discovery: find the files in a repository that our RAG app may read.

This is the first step of the ingestion pipeline:

    discover files  ->  (later) load  ->  chunk  ->  embed  ->  store

This module only looks at file NAMES and FOLDERS. It never opens or reads
any file, so it cannot leak the contents of a secret file.

Usage:
    from src.ingestion.file_discovery import discover_files

    files = discover_files("path/to/some/repository")
    # ['README.md', 'src/app.py', 'src/utils/helpers.py']
"""

from pathlib import Path
import fnmatch
import os

# --- Settings -----------------------------------------------------------
# Kept as plain constants at the top so they are easy to find and change.

# Only files with these extensions are returned.
SUPPORTED_EXTENSIONS = {".java", ".py", ".md", ".txt", ".yml", ".yaml"}

# We never walk into folders with these names. Skipping a folder skips
# everything inside it, which also makes discovery much faster
# (node_modules can contain tens of thousands of files).
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".next",
}

# Files whose NAME looks like they hold secrets. Patterns use * as a wildcard.
# This is a safety net based on names only: it cannot detect a password that
# someone pasted into a normal file like config.yaml.
SECRET_FILE_PATTERNS = [
    ".env",          # the real environment file
    ".env.*",        # .env.local, .env.production, ...
    "*.pem",         # certificates / private keys
    "*.key",         # private keys
    "id_rsa*",       # SSH private keys
    "credentials*",  # credentials.yml, credentials.json, ...
    "secrets.*",     # secrets.yaml, secrets.txt, ...
]


# --- Helper functions ---------------------------------------------------

def is_secret_file(filename):
    """Return True if the file name looks like it may contain secrets."""
    name = filename.lower()  # so ".ENV" and ".env" are treated the same
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILE_PATTERNS)


def is_supported_file(filename):
    """Return True if the file has an extension we want to index."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# --- Main function ------------------------------------------------------

def discover_files(repo_root):
    """Find all supported, non-secret files under `repo_root`.

    Args:
        repo_root: path to a repository folder (a string or a Path).
                   It can be any folder, so this works for any project.

    Returns:
        A sorted list of file paths, RELATIVE to repo_root and written with
        forward slashes (e.g. "src/app.py"), the same on Windows and Linux.

    Raises:
        FileNotFoundError: if repo_root does not exist.
        NotADirectoryError: if repo_root is a file, not a folder.
    """
    root = Path(repo_root)

    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a folder: {root}")

    found = []

    # os.walk visits every folder under root. For each folder it gives us:
    #   current  - the folder being visited
    #   dirnames - names of its sub-folders
    #   filenames - names of the files directly inside it
    for current, dirnames, filenames in os.walk(root):

        # Remove excluded folders from `dirnames`. Changing this list IN PLACE
        # (note the [:]) tells os.walk not to go into them at all.
        # We also skip symlinked folders: a link could point outside the
        # repository or back to a parent folder and loop forever.
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not (Path(current) / d).is_symlink()
        ]

        for filename in filenames:
            path = Path(current) / filename

            if path.is_symlink():          # same reason as for folders
                continue
            if is_secret_file(filename):   # check secrets first: safety wins
                continue
            if not is_supported_file(filename):
                continue

            # as_posix() gives forward slashes ("src/app.py") on every OS,
            # so results look the same on Windows and Linux.
            found.append(path.relative_to(root).as_posix())

    # os.walk's order depends on the file system. Sorting makes the result
    # identical on every run, which makes tests and debugging much easier.
    return sorted(found)


if __name__ == "__main__":
    # Quick manual check from the terminal:
    #   python -m src.ingestion.file_discovery C:\path\to\any\repo
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.file_discovery <repository_folder>")
        sys.exit(1)

    for file_path in discover_files(sys.argv[1]):
        print(file_path)