"""Embedders: turn text into vectors (lists of numbers).

Pipeline so far:

    discover -> load -> chunk -> EMBED -> (store) -> (later) retrieve

An embedding model reads a piece of text and returns a vector (for the model
we use, 384 numbers). Texts with similar MEANING get vectors that point in
similar directions, so we can later find chunks "close" to a question.

Every embedder here has the same small interface:

    embedder.name                 -> a string identifying the model
    embedder.embed_documents(texts) -> one vector per text   (used for chunks)
    embedder.embed_query(text)      -> one vector             (used for questions)

Keeping the interface tiny means the rest of the project does not care WHICH
model is used. Switching models later only changes one line.

Why two methods? Many models treat a short QUESTION and a longer PASSAGE a bit
differently (for example bge models add a hidden instruction to queries).
Using the right method for each gives better search results.
"""

import hashlib
import math
import re
from pathlib import Path

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # small (about 130 MB), 384 dims


class FastEmbedEmbedder:
    """Real embeddings, computed locally on your CPU with the fastembed library.

    The first time you create one, fastembed downloads the model once
    (about 100-150 MB). After that it works fully offline.
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, cache_dir=None):
        # Imported here (not at the top of the file) so that the rest of the
        # project, and its tests, can be used without fastembed installed.
        from fastembed import TextEmbedding

        if cache_dir is None:
            # fastembed's default cache lives in the system temp folder, which
            # Windows can clean up, causing a surprise re-download. A folder in
            # your home directory is stable and stays outside the repository.
            cache_dir = Path.home() / ".cache" / "fastembed"

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts):
        """Embed chunks of text. Returns a list of vectors (lists of floats)."""
        if not texts:
            return []
        # fastembed yields numpy arrays; we convert them to plain Python lists
        # of floats because that is what the vector database accepts.
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text):
        """Embed one question. Returns one vector."""
        return next(iter(self._model.query_embed(text))).tolist()


class HashingEmbedder:
    """A tiny, model-free embedder for TESTS and offline demos.

    It is NOT semantic: it only knows about shared WORDS. Each word is hashed
    to one of `dimensions` slots, and the vector counts the words per slot.
    Texts that share words get similar vectors. That is enough to check that
    our storage and search plumbing works without downloading any model.
    """

    name = "hashing-embedder"

    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        # Lowercase words made of letters, digits and underscores.
        for word in re.findall(r"[a-z0-9_]+", text.lower()):
            # md5 gives the same number for the same word on every run
            # (Python's built-in hash() changes between runs, so we avoid it).
            slot = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[slot] += 1.0

        length = math.sqrt(sum(v * v for v in vector))
        if length == 0:
            vector[0] = 1.0  # text with no words: avoid an all-zero vector
            return vector
        # Scale to length 1 so cosine similarity compares direction only.
        return [v / length for v in vector]