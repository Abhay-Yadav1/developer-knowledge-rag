"""Chunker: split each document into small pieces that remember their origin.

Why chunk? (see docs/module-2-chunking.md) An embedding of a whole file is
blurry, and an LLM prompt can only hold so much text. Small, focused chunks
are easier to search and to cite.

THIS IS VERSION 1 (a simple baseline). It works on lines of text:

  1. Add lines to the current chunk until it would exceed `max_chars`.
  2. If we have to cut, prefer to cut at a BLANK LINE (between functions,
     paragraphs, or sections) instead of in the middle of something.
  3. Start the next chunk a couple of lines BEFORE the cut (overlap), so a
     sentence or statement right at the boundary appears in both chunks.

Every chunk records its file path and START/END LINE numbers. These are what
our citations will use later ("src/app.py, L10-L24").

Planned upgrade (Module 2 plan): structure-aware splitting, i.e. one chunk per
Java/Python function and per Markdown heading. This baseline lets us build and
test the rest of the pipeline first, and gives us something to compare against.
"""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.loader import Document, load_repository

DEFAULT_MAX_CHARS = 1000     # roughly 25 lines of code
DEFAULT_OVERLAP_LINES = 2


@dataclass(frozen=True)
class Chunk:
    """A piece of a file, with the metadata we need for citations."""

    chunk_id: str     # unique id, e.g. "src/app.py:10-24"
    file_path: str    # relative to the repository root
    start_line: int   # first line of the chunk, counting from 1
    end_line: int     # last line of the chunk (inclusive)
    file_type: str    # the extension without a dot: "py", "md", ...
    text: str


def _find_break(lines, start, end):
    """Choose a nicer place to end a chunk that currently spans [start, end).

    Looks backwards for a blank line, but only in the second half of the chunk
    (otherwise we could end up with tiny chunks). Returns the new `end`, or
    the original `end` if no blank line was found.
    """
    halfway = start + (end - start) // 2
    for i in range(end - 1, halfway - 1, -1):
        if lines[i].strip() == "":
            return i + 1  # cut just after the blank line
    return end


def _trim_blank_lines(lines, start, end):
    """Shrink [start, end) so it neither starts nor ends with a blank line."""
    while start < end and lines[start].strip() == "":
        start += 1
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return start, end


def chunk_document(document, max_chars=DEFAULT_MAX_CHARS,
                   overlap_lines=DEFAULT_OVERLAP_LINES):
    """Split one Document into a list of Chunk objects."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative")

    # splitlines() understands both Windows (\r\n) and Linux (\n) line endings.
    lines = document.text.splitlines()
    file_type = Path(document.file_path).suffix.lstrip(".").lower()

    chunks = []
    start = 0  # index into `lines`; line numbers are index + 1

    while start < len(lines):
        # Step 1: grow the chunk line by line until it is full.
        end = start
        size = 0
        # `end == start` guarantees at least one line per chunk, even if a
        # single line is longer than max_chars (otherwise we'd loop forever).
        while end < len(lines) and (end == start or size + len(lines[end]) + 1 <= max_chars):
            size += len(lines[end]) + 1  # +1 for the newline character
            end += 1

        # Step 2: if the file continues, try to cut at a blank line.
        if end < len(lines):
            end = _find_break(lines, start, end)

        # Step 3: save the chunk (ignoring blank lines at its edges).
        chunk_start, chunk_end = _trim_blank_lines(lines, start, end)
        if chunk_start < chunk_end:  # skip chunks that are only blank lines
            first_line = chunk_start + 1  # convert index -> 1-based line number
            last_line = chunk_end
            chunks.append(Chunk(
                chunk_id=f"{document.file_path}:{first_line}-{last_line}",
                file_path=document.file_path,
                start_line=first_line,
                end_line=last_line,
                file_type=file_type,
                text="\n".join(lines[chunk_start:chunk_end]),
            ))

        # Step 4: move on. Step back by `overlap_lines`, but always make
        # progress (next start must be later than the current start).
        if end >= len(lines):
            break
        next_start = end - overlap_lines
        start = next_start if next_start > start else end

    return chunks


def chunk_repository(repo_root, max_chars=DEFAULT_MAX_CHARS,
                     overlap_lines=DEFAULT_OVERLAP_LINES):
    """Discover, load, and chunk every eligible file under `repo_root`."""
    chunks = []
    for document in load_repository(repo_root):
        chunks.extend(chunk_document(document, max_chars, overlap_lines))
    return chunks


if __name__ == "__main__":
    # Manual check from the terminal:
    #   python -m src.ingestion.chunker C:\path\to\any\repo
    import sys
    from collections import Counter

    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.chunker <repository_folder>")
        sys.exit(1)

    all_chunks = chunk_repository(sys.argv[1])
    per_file = Counter(chunk.file_path for chunk in all_chunks)

    print(f"{len(per_file)} files -> {len(all_chunks)} chunks\n")
    print("Chunks per file (top 5):")
    for path, count in per_file.most_common(5):
        print(f"  {count:3d}  {path}")

    print("\nFirst 3 chunks:")
    for chunk in all_chunks[:3]:
        print(f"\n--- {chunk.chunk_id} ({chunk.end_line - chunk.start_line + 1} lines)")
        print(chunk.text[:300])