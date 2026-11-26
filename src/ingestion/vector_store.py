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
from dataclasses import dataclass, field

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
    # repr=False taaki print karne par 384 numbers screen na bhar dein;
    # compare=False taaki purane tests jahan embedding None hai break na hon.
    embedding: tuple[float, ...] | None = field(default=None, repr=False, compare=False)


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

    def search(self, query_embedding, k=5, include_embeddings=False):
        """Return the `k` most similar chunks, best match first."""
        if k <= 0:
            raise ValueError("k must be greater than 0")

        # MMR ko chunk ke vectors chahiye hote hain, par normal search ko nahi.
        # Isliye sirf zaroorat padne par hi "embeddings" mangwate hain.
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,  # asking for more than exist is fine
            include=include,
        )

        # Agar embeddings nahi maange, toh har result ke liye None rakh do.
        ids = response["ids"][0]
        embeddings = response["embeddings"][0] if include_embeddings else [None] * len(ids)

        results = []
        for chunk_id, text, meta, distance, emb in zip(
            ids,
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
            embeddings,
        ):
            results.append(SearchResult(
                chunk_id=chunk_id,
                file_path=meta["file_path"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                file_type=meta["file_type"],
                text=text,
                score=max(-1.0, min(1.0, 1.0 - distance)),
                # Chroma numpy array de sakta hai, usko clean Python tuple bana dete hain.
                embedding=tuple(float(x) for x in emb) if emb is not None else None,
            ))
        return results

    def count(self):
        """Number of chunks currently stored."""
        return self._collection.count()
    
    def get_all_chunks(self):
        """Chroma mein stored saare chunks wapas laata hai (BM25 index banane ke liye)."""
        response = self._collection.get(include=["documents", "metadatas"])
        results = []
        for chunk_id, text, meta in zip(
            response["ids"],
            response["documents"],
            response["metadatas"],
        ):
            results.append(SearchResult(
                chunk_id=chunk_id,
                file_path=meta["file_path"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                file_type=meta["file_type"],
                text=text,
                score=0.0,  # Abhi koi query nahi chali, isliye default score 0.0
            ))
        return results
    
    def clear(self):
        """Remove every chunk (keeps the same collection name and settings)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._open_collection()