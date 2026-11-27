"""Offline tests for src/retrieval/multi_query.py (zero Groq API calls)."""

import pytest

from src.retrieval.multi_query import generate_queries


def test_generate_queries_includes_original_and_cleans_bullets():
    def fake_llm(query, n):
        return (
            "1. AgentState TypedDict definition\n"
            "- LangGraph state schema fields\n"
            "3) workflow state class in graphs\n"
        )

    queries = generate_queries("How is graph state defined?", n=3, llm_fn=fake_llm)

    assert queries == [
        "How is graph state defined?",
        "AgentState TypedDict definition",
        "LangGraph state schema fields",
        "workflow state class in graphs",
    ]


def test_generate_queries_deduplicates_and_caps_at_n():
    def fake_llm(query, n):
        return (
            "How is graph state defined?\n"  # original ka duplicate
            "state schema\n"
            "STATE SCHEMA\n"                 # case-insensitive duplicate
            "extra query 1\n"
            "extra query 2\n"
        )

    queries = generate_queries("How is graph state defined?", n=2, llm_fn=fake_llm)

    assert queries == ["How is graph state defined?", "state schema", "extra query 1"]


def test_generate_queries_falls_back_to_original_on_error():
    def broken_llm(query, n):
        raise ValueError("API key missing")

    queries = generate_queries("How is graph state defined?", n=3, llm_fn=broken_llm)

    assert queries == ["How is graph state defined?"]


def test_blank_query_raises_value_error():
    with pytest.raises(ValueError):
        generate_queries("   ")