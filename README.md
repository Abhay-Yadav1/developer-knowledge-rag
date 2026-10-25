# Developer Knowledge & Codebase Assistant

A local retrieval-augmented generation (RAG) application for searching software documentation and source code. It indexes a repository into a persistent Chroma database, retrieves relevant chunks with source locations, and can pass that evidence to Groq for a cited natural-language answer.

## What it does

- Discovers `.java`, `.py`, `.md`, `.txt`, `.yml`, and `.yaml` files while skipping common generated folders, symlinks, and secret-like filenames.
- Loads readable UTF-8 text files, skipping files larger than 1 MB, files containing null bytes, and files that cannot be decoded as UTF-8.
- Splits files into line-based chunks (1,000-character limit and two-line overlap by default), preferring blank-line boundaries and retaining inclusive source line numbers.
- Embeds chunks locally with FastEmbed (`BAAI/bge-small-en-v1.5` by default) and stores vectors, text, and metadata in Chroma.
- Retrieves the closest chunks and filters weak matches using a configurable similarity threshold.
- Optionally sends retrieved context to Groq to generate an answer with citations to numbered evidence blocks.

Filename filtering is a safety measure, not a general secret scanner: sensitive content in an ordinary supported file may still be indexed. Review the repository and discovery rules before indexing it.

## Setup

Use Python 3 and install the project dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default FastEmbed model is downloaded the first time it is used (about 100–150 MB) and cached locally. Later runs can use the cached model offline. For the answer-generation command, install the optional Groq client and configure an API key:

```bash
python -m pip install groq
```

Set `GROQ_API_KEY` in your environment or create a project-root `.env` file containing:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or other secrets. The indexer excludes `.env` files, but you should still keep credentials out of source control.

## Index a repository

Index a repository and optionally print the top matching chunks:

```bash
python -m src.ingestion.indexer path/to/repository --db chroma_db --query "How does authentication work?" --k 3
```

On Windows PowerShell:

```powershell
python -m src.ingestion.indexer "C:\path\to\repository" --db chroma_db --query "How does authentication work?" --k 3
```

By default, indexing rebuilds the selected database contents from the repository, removing chunks for changed or deleted files. To search an existing database without re-indexing, use `--no-index`:

```bash
python -m src.ingestion.indexer path/to/repository --db chroma_db --no-index --query "password reset" --k 5
```

The `repo` argument remains required with `--no-index`, though it is not read. The indexer and search commands must use the same embedding model. The database is stored in the selected local folder (`chroma_db` by default).

For a quick pipeline demo without downloading a model, use the deterministic word-hashing embedder:

```bash
python -m src.ingestion.indexer path/to/repository --db demo_db --demo-embedder --query "reset a user password"
```

Hashing vectors are lexical, not semantic. When searching an existing index, use `--demo-embedder` only if that index was also built with it.

## Retrieve evidence

The retrieval CLI shows the candidate scores, source locations, and which chunks pass the similarity threshold. This is useful for checking whether the default threshold (0.5) suits your repository:

```bash
python -m src.retrieval.retriever "How does authentication work?" --db chroma_db --k 5 --min-score 0.5
```

For a hashing-embedder index, add `--demo-embedder`. The threshold is a starting point and depends on the model and data; inspect real scores and adjust it as needed.

## Ask a question

After indexing with the default FastEmbed model, ask a question using the configured Groq API key:

```bash
python ask.py "How does authentication work?"
```

`ask.py` reads the default `chroma_db` database and uses the default model and retrieval settings. It expects the database to exist and contain chunks. The generated response is instructed to answer from retrieved evidence and cite evidence block numbers; verify citations against the displayed source paths and line ranges when accuracy matters.

## Use the pipeline from Python

```python
from src.ingestion.embedder import FastEmbedEmbedder
from src.ingestion.indexer import index_repository
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever, format_context

embedder = FastEmbedEmbedder()
store = VectorStore("chroma_db", embedding_model=embedder.name)
index_repository("path/to/repository", embedder, store)

result = Retriever(embedder, store).retrieve("How does authentication work?")
if result.found:
    print(format_context(result))
else:
    print("No chunks passed the relevance threshold.")
```

## Project layout

```text
ask.py                       Retrieve evidence and generate an answer with Groq
src/ingestion/
  file_discovery.py          Find eligible repository files
  loader.py                  Read eligible text files safely
  chunker.py                 Create line-based chunks with citation metadata
  embedder.py                FastEmbed and hashing embedding implementations
  vector_store.py            Persistent Chroma storage and similarity search
  indexer.py                 CLI and end-to-end indexing pipeline
src/retrieval/
  retriever.py               Similarity thresholding and evidence formatting CLI
  generator.py               Groq-based answer generation
tests/                       Unit and integration tests
docs/                        Architecture, chunking, and RAG design notes
```

## Tests

Run the test suite with:

```bash
pytest
```

Tests use temporary repositories and the hashing embedder, so they do not need the FastEmbed model download or Groq API access.

## Design notes

- [Architecture](docs/module-1-architecture.md)
- [Chunking](docs/module-2-chunking.md)
- [RAG implementation reference](docs/Rag_implementation.md)

The current pipeline uses full index rebuilds rather than incremental updates. Hybrid keyword/vector retrieval, structure-aware chunking, retrieval evaluation, and deployment remain areas for future work.
