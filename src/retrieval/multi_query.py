"""Multi-Query Retriever: ek user question ko multiple search queries mein expand karna."""

import re
from groq import APIConnectionError, APIError, RateLimitError

from src.retrieval.generator import _get_client

# System prompt: LLM ko strict bolte hain ki sirf N clean search queries de, koi extra baat nahi.
_MULTI_QUERY_PROMPT = (
    "You are an AI search assistant for a software codebase. "
    "Generate {n} different search queries that rephrase the user's question using "
    "alternative technical terms, code identifiers, or file/class concepts to improve code retrieval. "
    "Return ONLY the queries, one per line, with no numbering, bullets, or explanation."
)


def _call_groq(query, n, model):
    """Groq API ko call karke raw text (har line par ek query) laata hai."""
    client = _get_client()
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": _MULTI_QUERY_PROMPT.format(n=n)},
            {"role": "user", "content": query},
        ],
        model=model,
        temperature=0.3,  # Thodi si creativity (0.3) taaki alag-alag wordings milein
    )
    return response.choices[0].message.content or ""


def _parse_lines(raw_text):
    """LLM ke output se clean queries nikalta hai (numbering/bullets hata kar)."""
    queries = []
    for line in raw_text.splitlines():
        # Agar LLM galti se "1. " ya "- " laga de, toh regex se hata do
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            queries.append(cleaned)
    return queries


def generate_queries(query, n=3, model="openai/gpt-oss-20b", llm_fn=None):
    """Original query + up to `n` alternative queries ki list return karta hai.

    Args:
        query: user ka original question.
        n: kitni extra variations banani hain (default 3).
        model: Groq model name.
        llm_fn: optional fake function (query, n) -> str, taaki pytest bina
                API call kiye offline test kar sake.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if n <= 0:
        return [query.strip()]

    base_query = query.strip()

    try:
        raw = llm_fn(base_query, n) if llm_fn is not None else _call_groq(base_query, n, model)
    except (RateLimitError, APIConnectionError, APIError, ValueError):
        # Graceful fallback: agar Groq fail ho jaye, toh crash mat karo—sirf original query se kaam chalao!
        return [base_query]

    variants = _parse_lines(raw)

    # Original query hamesha index 0 par rakho, aur duplicate queries hata do (case-insensitive)
    seen = {base_query.lower()}
    result = [base_query]
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            result.append(v)
        if len(result) >= n + 1:
            break

    return result