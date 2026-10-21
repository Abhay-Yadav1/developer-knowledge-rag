"""Tests for src/ingestion/embedder.py"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.embedder import FastEmbedEmbedder, HashingEmbedder


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ---- HashingEmbedder (the model-free embedder used in tests and demos) ----

def test_same_text_gives_same_vector_every_time():
    embedder = HashingEmbedder()

    assert embedder.embed_query("reset the password") == embedder.embed_query("reset the password")


def test_vectors_have_requested_size_and_unit_length():
    embedder = HashingEmbedder(dimensions=64)
    vector = embedder.embed_query("some words in a sentence")

    assert len(vector) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_texts_sharing_words_are_more_similar_than_unrelated_texts():
    embedder = HashingEmbedder()
    query = embedder.embed_query("how do users reset a password")
    related = embedder.embed_query("reset the user password and log out")
    unrelated = embedder.embed_query("render the invoice as a pdf file")

    assert cosine(query, related) > cosine(query, unrelated)


def test_text_without_words_still_gives_a_usable_vector():
    vector = HashingEmbedder().embed_query("!!! ??? ...")

    assert any(v != 0 for v in vector)


def test_embed_documents_returns_one_vector_per_text():
    vectors = HashingEmbedder().embed_documents(["one", "two", "three"])

    assert len(vectors) == 3


# ---- FastEmbedEmbedder (tested with a stand-in, so no model is downloaded) ----

class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding. Records how it is used."""

    created_with = None

    def __init__(self, model_name, cache_dir):
        FakeTextEmbedding.created_with = {"model_name": model_name, "cache_dir": cache_dir}
        self.calls = []

    def passage_embed(self, texts):
        self.calls.append(("passage", list(texts)))
        return (np.array([float(len(t)), 1.0], dtype=np.float32) for t in texts)

    def query_embed(self, text):
        self.calls.append(("query", text))
        return iter([np.array([9.0, 9.0], dtype=np.float32)])


@pytest.fixture
def fake_fastembed(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))


def test_fastembed_wrapper_returns_plain_python_lists(fake_fastembed):
    embedder = FastEmbedEmbedder(model_name="some/model")

    vectors = embedder.embed_documents(["ab", "abcd"])

    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    assert type(vectors[0]) is list and type(vectors[0][0]) is float


def test_fastembed_wrapper_uses_query_method_for_questions(fake_fastembed):
    embedder = FastEmbedEmbedder(model_name="some/model")

    assert embedder.embed_query("what is this?") == [9.0, 9.0]
    assert embedder._model.calls == [("query", "what is this?")]


def test_fastembed_wrapper_handles_empty_list_without_calling_model(fake_fastembed):
    embedder = FastEmbedEmbedder(model_name="some/model")

    assert embedder.embed_documents([]) == []
    assert embedder._model.calls == []


def test_fastembed_wrapper_exposes_model_name_and_stable_cache_folder(fake_fastembed):
    embedder = FastEmbedEmbedder(model_name="some/model")

    assert embedder.name == "some/model"
    cache_dir = Path(FakeTextEmbedding.created_with["cache_dir"])
    assert cache_dir == Path.home() / ".cache" / "fastembed"


def test_fastembed_wrapper_accepts_custom_cache_folder(fake_fastembed, tmp_path):
    FastEmbedEmbedder(model_name="some/model", cache_dir=tmp_path)

    assert FakeTextEmbedding.created_with["cache_dir"] == str(tmp_path)