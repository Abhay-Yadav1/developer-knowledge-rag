# Learning and Code Change Audit

Use this file as a running set of revision notes while building the project. Each entry explains what changed, why it changed, how the code works, and what to remember when evaluating it. Append each new entry after the previous one so the audit remains in chronological order.

## Entry template

Copy this section for each meaningful code change:

```markdown
## YYYY-MM-DD — Short change title

### What changed
- Files and behavior changed.

### Why
- The problem or learning goal behind the change.

### How it works
- The key implementation idea, in your own words.

### What I learned
- Concepts, trade-offs, or debugging lessons to remember.

### Limitations and next steps
- What this change does not solve and what to explore next.

### Verification
- Commands or manual checks run, and the outcome. Write “Not run” if nothing was checked.
```

Keep entries focused on behavior and reasoning rather than copying a full diff. Mention important functions or files so you can find the implementation later. Record only checks that were actually run.

## 2026-11-02 — Retrieval quality: lightweight lexical reranking

### What changed

- `src/retrieval/retriever.py` requests a wider candidate pool from the vector store (four times the requested `k`).
- It refines candidate order using exact query terms, code identifiers such as `MAX_RETRIES`, and matching terms in file paths.
- It still returns at most `k` chunks.
- The original cosine similarity score is unchanged, and `min_score` is still applied to that score.

### Why

Embeddings are useful for matching meaning, but can rank exact function names, configuration keys, or error identifiers poorly. A wider candidate pool plus a small lexical ranking signal can improve those exact matches without adding a model or runtime dependency.

### How it works

The query and chunk text are tokenized into lowercase words, with underscore-separated identifiers kept together. Candidates receive a small ranking bonus based on query-token coverage, with a smaller bonus for matching terms in the source path. The cosine score remains the relevance threshold score.

### What I learned

- Dense vector search and lexical matching provide different signals: semantic similarity helps with paraphrases, while exact token matches help with identifiers.
- Reranking only changes the order of retrieved candidates; it cannot find a relevant chunk that was absent from the candidate pool.
- A ranking bonus and the original similarity score serve different purposes. Here, ranking can change without changing threshold eligibility.

### Limitations and next steps

This is not full hybrid search. It only reorders candidates found by vector search. A next experiment is to add a separate keyword retriever such as BM25, combine its results with vector results, and compare both approaches using the same evaluation questions and index.

### Verification

Tests were not run after this change. To evaluate it, keep the repository index, question set, and evaluation settings the same, then compare hit rate, MRR, precision@k, and per-question results.

## 2026-11-23 - Retrieval diversity: Maximal Marginal Relevance (MMR)

### What changed

- `src/retrieval/retriever.py` adds an optional MMR selection path, disabled by default through `use_mmr=False`.
- When enabled, retrieval gathers a candidate pool of `4 × k`, applies the existing lexical reranker, then selects up to `k` chunks using query relevance and redundancy against already selected chunks.
- `src/ingestion/vector_store.py` can include each result's stored embedding on request. The retriever asks for vectors only when MMR is enabled; ordinary retrieval avoids the extra vector payload.
- `src/evaluation/runner.py` adds `--mmr` and `--mmr-lambda` so MMR can be evaluated and its relevance/diversity balance adjusted.
- `tests/test_retriever.py` covers reducing duplicate results and rejecting lambda values outside 0 to 1.

### Why

Top-k vector search can return several nearly identical chunks, using result slots that could provide evidence from other relevant parts of the repository. MMR tries to keep results relevant while reducing redundancy.

### How it works

`_mmr_select` keeps the first candidate as the initial result. For each later slot, it chooses the remaining candidate with the highest score:

`lambda × query_similarity - (1 - lambda) × max_similarity_to_selected`

Query similarity is the candidate's cosine score, and redundancy is the highest cosine similarity between that candidate's embedding and any selected chunk. The default lambda is `0.7`, weighting query relevance more heavily than diversity. A lambda of `1` removes the redundancy penalty; a lambda of `0` gives the penalty full weight after the first result. MMR selection happens before the existing `min_score` filter, which continues to use the original query cosine score.

### What I learned

- MMR adds diversity by considering relationships between results, not just each result's similarity to the query.
- It needs candidate embeddings to measure redundancy, so retrieving vectors can be conditional on MMR being enabled.
- The first result is fixed to the top candidate after lexical reranking. Later choices use cosine query scores and embedding similarity in the MMR formula.
- A wider candidate pool gives MMR more choices, but it cannot select relevant chunks that vector search did not retrieve.

### Limitations and next steps

MMR diversifies the vector-search candidate pool; it does not add a separate sparse keyword retriever. Compare MMR on and off using the same index, question set, `k`, and threshold. Try multiple lambda values and inspect aggregate metrics and per-question results. Sparse retrieval such as BM25 remains a possible next technique.

### Verification

MMR-focused tests are present in `tests/test_retriever.py`, but no test or evaluation run was performed while recording this entry.
