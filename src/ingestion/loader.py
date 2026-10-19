"""Loader: read the files found by file_discovery into memory.

Pipeline so far:

    discover_files  ->  load_repository  ->  (next) chunk  ->  embed  ->  store
    (names only)        (reads contents)

Because this is the first module that actually OPENS files, it is careful:
  * it only reads files that discover_files already approved
    (so .env files, keys, and similar are never opened),
  * it skips files that are huge or look binary,
  * it never crashes the whole run because one file is unreadable.
"""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.file_discovery import discover_files

# Files larger than this are skipped. Giant generated or data files are rarely
# useful for answering questions and would produce thousands of chunks.
DEFAULT_MAX_FILE_BYTES = 1_000_000  # about 1 MB


@dataclass(frozen=True)
class Document:
    """One file's contents plus where it came from."""

    file_path: str  # relative to the repository root, e.g. "src/app.py"
    text: str


def read_text_file(path, max_bytes=DEFAULT_MAX_FILE_BYTES):
    """Read one file and return its text, or None if it should be skipped.

    Returns None (instead of raising) for files that are too large, look like
    binary data, or cannot be read. The caller can just skip them.
    """
    path = Path(path)

    try:
        if path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None  # permission problem, file vanished, etc.

    # Text files never contain null bytes. If we find one, this is probably a
    # binary file that was renamed to .txt or similar. Skip it.
    if b"\x00" in data:
        return None

    try:
        # "utf-8-sig" decodes normal UTF-8 AND removes the invisible BOM marker
        # that some Windows editors (like Notepad) add to the start of a file.
        # Without it, the first line would start with a stray "\ufeff" character.
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Not valid UTF-8 (for example an old Windows-1252 file). We skip it
        # rather than guess the encoding and risk garbled text in the index.
        return None


def load_repository(repo_root, max_bytes=DEFAULT_MAX_FILE_BYTES):
    """Discover and read all eligible files under `repo_root`.

    Returns a list of Document objects, in the same sorted order as
    discover_files. Files that cannot be read are skipped.
    """
    root = Path(repo_root)
    documents = []

    for relative_path in discover_files(root):
        text = read_text_file(root / relative_path, max_bytes=max_bytes)
        if text is not None:
            documents.append(Document(file_path=relative_path, text=text))

    return documents