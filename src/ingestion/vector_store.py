"""Vector store: save chunk vectors (plus text and metadata) and search them.

We use Chroma, a small database that saves to a folder on disk. For each
chunk it keeps three things together:

    * the VECTOR    (used to find similar chunks)
    * the TEXT      (what we will later show to the LLM)
    * the METADATA  (file path, start/end line, file type: used for citations)

This class is a thin wrapper. The rest of the project talks to OUR small
interface (add_chunks / search / count / clear), not to Chroma directly, so
Chroma could be swapped for another database without touching other code.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

DEFAULT_COLLECTION_NAME = "code_chunks"

# Chroma limits how many items one call may contain, so we send big lists in
# batches. 500 is far below the limit.
_BATCH_SIZE = 500


@dataclass(frozen=True)
class SearchResult:
    """One search hit: the chunk's data plus how similar it was to the query."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    file_type: str
    text: str
    score: float  # cosine similarity: 1.0 = same direction, 0 = unrelated


class VectorStore:
    """A persistent collection of embedded chunks."""

    def __init__(self, path, embedding_model, collection_name=DEFAULT_COLLECTION_NAME):
        """Open (or create) a store.

        Args:
            path: folder where Chroma saves its data (created if missing).
            embedding_model: name of the embedder that makes the vectors
                (for example "BAAI/bge-small-en-v1.5"). It is recorded in the
                store, so we can refuse to mix vectors from different models.
            collection_name: name of the collection inside the database. At
                least 3 characters: letters, digits, dot, dash, underscore.
        """
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        # Telemetry is switched off so nothing is sent anywhere.
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()

    def _open_collection(self):
        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            # "cosine" means: compare the DIRECTION of vectors (the usual
            # choice for text embeddings).
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedding_model},
            # None = "we supply our own vectors; do not embed text for us".
            embedding_function=None,
        )

        # Vectors from different models are NOT comparable (they may even have
        # a different number of dimensions). If this collection already exists,
        # it keeps the model name it was created with, so we can check it.
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model != self._embedding_model:
            raise ValueError(
                f"Collection '{self._collection_name}' was built with embedding model "
                f"'{stored_model}', but you are using '{self._embedding_model}'. "
                "Delete the database folder or use a different collection_name, "
                "then index again."
            )
        return collection

    def add_chunks(self, chunks, embeddings):
        """Save chunks with their vectors. Same chunk_id again = overwrite.

        Args:
            chunks: list of Chunk objects.
            embeddings: list of vectors, one per chunk, in the same order.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i:i + _BATCH_SIZE]
            # "upsert" = update if the id exists, insert otherwise. This makes
            # re-running indexing safe: it never creates duplicates.
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[i:i + _BATCH_SIZE],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "file_path": c.file_path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "file_type": c.file_type,
                    }
                    for c in batch
                ],
            )

    def search(self, query_embedding, k=5):
        """Return the `k` most similar chunks, best match first."""
        if k <= 0:
            raise ValueError("k must be greater than 0")

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,  # asking for more than exist is fine
            include=["documents", "metadatas", "distances"],
        )

        results = []
        # Chroma supports several queries at once, so every field is a list of
        # lists. We sent exactly one query, hence the [0].
        for chunk_id, text, meta, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            results.append(SearchResult(
                chunk_id=chunk_id,
                file_path=meta["file_path"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                file_type=meta["file_type"],
                text=text,
                # Chroma returns cosine DISTANCE (0 = identical).
                # Similarity is easier to read: 1 - distance.
                score=1.0 - distance,
            ))
        return results

    def count(self):
        """Number of chunks currently stored."""
        return self._collection.count()

    def clear(self):
        """Remove every chunk (keeps the same collection name and settings)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._open_collection()