# Developer Knowledge & Codebase Assistant

A local retrieval-augmented generation (RAG) project for searching software documentation and source code. The current implementation covers repository ingestion, chunking, embeddings, persistent vector storage, and similarity search. It does not yet generate natural-language answers with an LLM.

## Current capabilities

- Discover supported source and documentation files while excluding common generated folders and secret-like filenames.
- Load readable UTF-8 text files, skipping oversized, binary, or unreadable files.
- Split files into line-based chunks with blank-line-aware boundaries, overlap, and source line metadata.
- Embed chunks with a local FastEmbed model, or use a deterministic word-hashing embedder for quick demos and tests.
- Store vectors, chunk text, and metadata in a persistent local Chroma database.
- Search the database by vector similarity and display file paths and line ranges for results.
- Rebuild an index to remove stale chunks from edited or deleted files.

## Ingestion flow

```text
repository -> file discovery -> loading -> chunking -> embeddings -> Chroma -> similarity search
```

The current supported extensions are `.java`, `.py`, `.md`, `.txt`, `.yml`, and `.yaml`. Discovery skips common build, cache, virtual-environment, and dependency folders; symlinks; and filenames matching secret patterns such as `.env*`, `*.pem`, `*.key`, `id_rsa*`, `credentials*`, and `secrets.*`. This filename-based filtering is a safety measure, not a general secret scanner.

Files larger than 1 MB, files containing null bytes, and files that cannot be decoded as UTF-8 are skipped by default. Chunking uses a 1,000-character limit and two-line overlap by default, prefers blank-line cut points, and records the path, file type, and inclusive start/end line numbers.

## Setup

Use Python 3. Install the project dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default `FastEmbedEmbedder` uses `BAAI/bge-small-en-v1.5` (384-dimensional embeddings). FastEmbed downloads the model the first time it is used; later runs use its local cache. The `--demo-embedder` option avoids a model download, but its word-hashing vectors are lexical rather than semantic.

## Index and search

Index a repository and then search it in one command:

```bash
python -m src.ingestion.indexer path/to/repository --db chroma_db --query "How does authentication work?" --k 3
```

On Windows, for example:

```powershell
python -m src.ingestion.indexer "C:\path\to\repository" --db chroma_db --query "How does authentication work?" --k 3
```

For a quick demo without downloading an embedding model:

```bash
python -m src.ingestion.indexer path/to/repository --db chroma_db --query "reset a user password" --demo-embedder
```

The default run rebuilds the database contents from the repository, which removes chunks for changed and deleted files. Search an existing database without re-indexing by passing `--no-index`:

```bash
python -m src.ingestion.indexer path/to/repository --db chroma_db --no-index --query "password reset" --k 5
```

The `repo` positional argument is still required by the CLI when using `--no-index`, although the search reads from the existing database. Results include a similarity score and source file line range. Chroma data is written to the selected local database folder (default `chroma_db`).

To use the pipeline from Python:

```python
from src.ingestion.embedder import FastEmbedEmbedder
from src.ingestion.indexer import index_repository
from src.ingestion.vector_store import VectorStore

embedder = FastEmbedEmbedder()
store = VectorStore("chroma_db", embedding_model=embedder.name)
count = index_repository("path/to/repository", embedder, store)

query_vector = embedder.embed_query("How does authentication work?")
for result in store.search(query_vector, k=3):
    print(result.file_path, result.start_line, result.end_line, result.score)
    print(result.text)
```

## Project layout

```text
src/ingestion/
  file_discovery.py  Find eligible repository files
  loader.py          Read eligible text files safely
  chunker.py         Create line-based chunks with citation metadata
  embedder.py        FastEmbed and hashing embedding implementations
  vector_store.py    Persistent Chroma storage and similarity search
  indexer.py         CLI and end-to-end indexing pipeline
tests/               Unit and integration tests for ingestion and search
docs/                 Architecture, chunking, and RAG design notes
```

## Tests

Run the test suite with:

```bash
pytest
```

Tests use temporary repositories and the hashing embedder, so they do not need the FastEmbed model download.

## Design notes and roadmap

- [Architecture](docs/module-1-architecture.md) describes the intended RAG system and design considerations.
- [Chunking](docs/module-2-chunking.md) compares strategies and discusses structure-aware chunking as a future improvement.
- [RAG implementation reference](docs/Rag_implementation.md) outlines retrieval, generation, evaluation, and production topics.

The implementation is currently a tested ingestion and vector-search foundation. Planned work includes structure-aware chunking, incremental indexing, hybrid keyword/vector retrieval, retrieval evaluation, an LLM answer-generation layer with verified citations, and deployment. Those capabilities are design goals, not part of the current code.
