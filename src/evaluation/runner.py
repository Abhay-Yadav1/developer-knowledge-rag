"""Evaluation runner: test retrieval quality on a file of known questions.

    questions.json -> retrieve for each -> score -> report + threshold sweep

Usage (the database must contain the repository the questions are about):

    python -m src.evaluation.runner evaluation/questions_intelligent_db_query_agent.json
    python -m src.evaluation.runner <questions.json> --min-score 0.55 --output results.json

Saving results with --output lets you compare runs before and after a change
(different chunking, a new embedding model, hybrid search...).
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.metrics import (
    Candidate,
    QuestionResult,
    first_hit_rank,
    hit_rate,
    mrr,
    precision_at_k,
    recommend_threshold,
    refusal_accuracy,
    sweep,
)
from src.ingestion.embedder import FastEmbedEmbedder, HashingEmbedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import DEFAULT_K, DEFAULT_MIN_SCORE, Retriever


def load_questions(path):
    """Read and validate a questions file. Returns a list of question dicts.

    Each question needs: "id", "question", and "expected_files" (a list of file
    paths relative to the repository root; an EMPTY list marks the question as
    unanswerable). "category" is optional.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"Questions file not found: {path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Questions file is not valid JSON: {error}")

    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError('The file must contain a non-empty "questions" list.')

    seen_ids = set()
    for index, q in enumerate(questions, start=1):
        label = f"question #{index}"
        if not isinstance(q, dict):
            raise ValueError(f"{label} must be an object.")
        for field in ("id", "question", "expected_files"):
            if field not in q:
                raise ValueError(f'{label} is missing "{field}".')
        if not isinstance(q["question"], str) or not q["question"].strip():
            raise ValueError(f'{label} has an empty "question".')
        if not isinstance(q["expected_files"], list):
            raise ValueError(f'{label} ("{q["id"]}"): "expected_files" must be a list.')
        if q["id"] in seen_ids:
            raise ValueError(f'Duplicate question id "{q["id"]}".')
        seen_ids.add(q["id"])

    if not any(q["expected_files"] for q in questions):
        raise ValueError("At least one question needs expected_files (an answerable question).")
    return questions


def run_retrieval(questions, retriever):
    """Retrieve for every question. The retriever must use min_score=-1 so that
    ALL candidates are returned; thresholds are applied later, when scoring."""
    results = []
    for q in questions:
        retrieved = retriever.retrieve(q["question"])
        candidates = tuple(
            Candidate(c.file_path, c.start_line, c.end_line, c.score)
            for c in retrieved.chunks + retrieved.rejected
        )
        results.append(QuestionResult(
            id=q["id"],
            question=q["question"],
            category=q.get("category", "general"),
            expected_files=tuple(q["expected_files"]),
            candidates=candidates,
        ))
    return results


def question_status(result, min_score):
    """One-line verdict for a question at the given threshold."""
    returned = [c for c in result.candidates if c.score >= min_score]
    if result.answerable:
        rank = first_hit_rank(result, min_score)
        if rank is not None:
            return f"HIT @{rank}"
        return "MISS (wrong files)" if returned else "MISS (nothing passed)"
    return "WRONGLY ANSWERED" if returned else "refused OK"


def summarize(results, k, min_score):
    """All headline numbers at one threshold, as a dict (also saved to JSON)."""
    return {
        "questions": len(results),
        "answerable": sum(1 for r in results if r.answerable),
        "unanswerable": sum(1 for r in results if not r.answerable),
        "k": k,
        "min_score": min_score,
        "hit_rate_at_1": hit_rate(results, 1, min_score),
        "hit_rate_at_3": hit_rate(results, min(3, k), min_score),
        "hit_rate_at_k": hit_rate(results, k, min_score),
        "mrr": mrr(results, min_score),
        "precision_at_k": precision_at_k(results, k, min_score),
        "refusal_accuracy": refusal_accuracy(results, min_score),
    }


def _pct(value):
    return "  n/a" if value is None else f"{value * 100:4.0f}%"


def _num(value):
    return " n/a" if value is None else f"{value:.2f}"


def format_report(results, k, min_score):
    summary = summarize(results, k, min_score)
    lines = [
        f"Evaluation: {summary['questions']} questions "
        f"({summary['answerable']} answerable, {summary['unanswerable']} unanswerable)",
        f"k = {k}   threshold = {min_score}",
        "",
        f"{'id':<8}{'result':<24}{'best':>6}  detail",
    ]

    for r in results:
        best = max((c.score for c in r.candidates), default=None)
        status = question_status(r, min_score)
        if r.answerable:
            top_files = ", ".join(dict.fromkeys(c.file_path for c in r.candidates[:3]))
            detail = f"expected {', '.join(r.expected_files)} | top: {top_files}"
        else:
            detail = f"top: {r.candidates[0].file_path}" if r.candidates else ""
        lines.append(f"{r.id:<8}{status:<24}{_num(best):>6}  {detail}")

    lines += [
        "",
        f"Summary at threshold {min_score}:",
        f"  hit rate @1   {_pct(summary['hit_rate_at_1'])}",
        *([f"  hit rate @3   {_pct(summary['hit_rate_at_3'])}"] if k > 3 else []),
        f"  hit rate @{k:<2}  {_pct(summary['hit_rate_at_k'])}" if k > 1 else "",
        f"  MRR            {_num(summary['mrr'])}",
        f"  precision @{k:<2} {_pct(summary['precision_at_k'])}",
        f"  refusal acc.  {_pct(summary['refusal_accuracy'])}   (unanswerable questions correctly refused)",
        "",
        "Threshold sweep:",
        "  thr    hit@k    MRR  refusal  balanced",
    ]

    rows = sweep(results, k)
    for row in rows:
        lines.append(f"  {row['threshold']:.2f}  {_pct(row['hit_rate'])}  {_num(row['mrr'])}  "
                     f"{_pct(row['refusal_accuracy'])}  {_pct(row['balanced'])}")

    lines = [line for line in lines if line is not None]
    best = recommend_threshold(rows)
    if best:
        lines += ["", f"Suggested threshold: {best['threshold']:.2f} "
                      f"(balanced score {_pct(best['balanced']).strip()}). "
                      "With few questions, look at the whole table before trusting this."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a set of known questions.")
    parser.add_argument("questions", help="path to the questions JSON file")
    parser.add_argument("--db", default="chroma_db", help="database folder (default: chroma_db)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"candidates per question (default: {DEFAULT_K})")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"threshold for the main report (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--output", help="also save the numbers to this JSON file")
    parser.add_argument("--demo-embedder", action="store_true",
                        help="use the word-hashing embedder (must match how the db was indexed)")
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.questions)
        embedder = HashingEmbedder() if args.demo_embedder else FastEmbedEmbedder()
        store = VectorStore(args.db, embedding_model=embedder.name)
        if store.count() == 0:
            print(f"The database '{args.db}' is empty. Run src.ingestion.indexer first.")
            return 1
        # min_score=-1 keeps every candidate; thresholds are applied when scoring.
        retriever = Retriever(embedder, store, k=args.k, min_score=-1.0)
        results = run_retrieval(questions, retriever)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    print(format_report(results, args.k, args.min_score))

    if args.output:
        report = {
            "summary": summarize(results, args.k, args.min_score),
            "sweep": sweep(results, args.k),
            "questions": [
                {"id": r.id, "category": r.category, "status": question_status(r, args.min_score),
                 "expected_files": list(r.expected_files),
                 "top_files": [c.file_path for c in r.candidates[:args.k]]}
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved numbers to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())