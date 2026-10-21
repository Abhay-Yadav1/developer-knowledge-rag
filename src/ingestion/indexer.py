"""Indexer: run the whole ingestion pipeline for one repository.

    discover -> load -> chunk -> embed -> store      (this file ties them together)

Usage from Python:

    from src.ingestion.embedder import FastEmbedEmbedder
    from src.ingestion.vector_store import VectorStore
    from src.ingestion.indexer import index_repository

    embedder = FastEmbedEmbedder()
    store = VectorStore("chroma_db", embedding_model=embedder.name)
    index_repository("path/to/some/repository", embedder, store)

Usage from the terminal: see the bottom of this file.
"""

import argparse
import sys

from src.ingestion.chunker import chunk_repository
from src.ingestion.embedder import FastEmbedEmbedder, HashingEmbedder
from src.ingestion.vector_store import VectorStore

# How many chunks we embed per call. Small batches keep memory use low and let
# us show progress.
_EMBED_BATCH_SIZE = 64


def index_repository(repo_root, embedder, store, rebuild=True):
    """Chunk, embed and store every eligible file in `repo_root`.

    Args:
        repo_root: path to the repository folder.
        embedder: any object with embed_documents(texts) (see embedder.py).
        store: a VectorStore.
        rebuild: if True (default), the store is emptied first, so it ends up
            exactly matching the repository. That also removes chunks of files
            that were edited or deleted since the last run. (Smarter updates
            that only touch changed files are a planned later feature.)

    Returns:
        The number of chunks stored.
    """
    # Chunk FIRST. If repo_root is wrong this raises an error here, before we
    # wipe an existing index.
    chunks = chunk_repository(repo_root)

    if rebuild:
        store.clear()

    for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[start:start + _EMBED_BATCH_SIZE]
        vectors = embedder.embed_documents([chunk.text for chunk in batch])
        store.add_chunks(batch, vectors)

    return len(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Index a repository into a local Chroma database and optionally search it."
    )
    parser.add_argument("repo", help="path to the repository folder to index")
    parser.add_argument("--db", default="chroma_db", help="database folder (default: chroma_db)")
    parser.add_argument("--query", help="after indexing, search for this question")
    parser.add_argument("--k", type=int, default=3, help="number of results to show (default: 3)")
    parser.add_argument("--no-index", action="store_true",
                        help="skip indexing; only search an existing database")
    parser.add_argument("--demo-embedder", action="store_true",
                        help="use the simple word-hashing embedder (no model download, not semantic)")
    args = parser.parse_args(argv)

    embedder = HashingEmbedder() if args.demo_embedder else FastEmbedEmbedder()
    store = VectorStore(args.db, embedding_model=embedder.name)

    if not args.no_index:
        count = index_repository(args.repo, embedder, store)
        print(f"Indexed {count} chunks from {args.repo} into '{args.db}'")

    print(f"Database '{args.db}' holds {store.count()} chunks (model: {embedder.name})")

    if args.query:
        print(f"\nQuestion: {args.query}")
        results = store.search(embedder.embed_query(args.query), k=args.k)
        for rank, result in enumerate(results, start=1):
            preview = result.text[:200].replace("\n", "\n    ")
            print(f"\n#{rank}  score {result.score:.3f}  "
                  f"{result.file_path} L{result.start_line}-L{result.end_line}")
            print(f"    {preview}")


if __name__ == "__main__":
    sys.exit(main())