"""Retrieval metrics: numbers that say how good our search is.

For every test question we know which file(s) SHOULD answer it
("expected_files"). We retrieve candidates, then ask: did the right file come
back, and how high was it ranked?

Questions with NO expected files are "unanswerable" (the repository does not
contain the answer). For those, the right behaviour is to return nothing, so
we measure whether the similarity threshold correctly rejected everything.

All functions take a `min_score` (the similarity threshold). Candidates scoring
below it count as "not returned". That lets us re-score the same retrieval run
at many thresholds without searching again (see `sweep`).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One retrieved chunk (only the fields metrics need)."""

    file_path: str
    start_line: int
    end_line: int
    score: float


@dataclass(frozen=True)
class QuestionResult:
    """A test question plus what retrieval returned for it, best first."""

    id: str
    question: str
    category: str
    expected_files: tuple   # empty tuple = the question is unanswerable
    candidates: tuple       # Candidate objects, best score first

    @property
    def answerable(self):
        return len(self.expected_files) > 0


def _returned(result, min_score):
    """Candidates that pass the threshold (these are what the user would get)."""
    return [c for c in result.candidates if c.score >= min_score]


def _answerable(results):
    return [r for r in results if r.answerable]


def _unanswerable(results):
    return [r for r in results if not r.answerable]


def first_hit_rank(result, min_score=-1.0):
    """Rank (1 = best) of the first returned chunk from an expected file, else None."""
    for rank, candidate in enumerate(_returned(result, min_score), start=1):
        if candidate.file_path in result.expected_files:
            return rank
    return None


def hit_rate(results, k, min_score=-1.0):
    """Fraction of answerable questions with a correct chunk in the top k.

    (Also called recall@k.) Returns None if there are no answerable questions.
    """
    answerable = _answerable(results)
    if not answerable:
        return None
    hits = 0
    for r in answerable:
        rank = first_hit_rank(r, min_score)
        if rank is not None and rank <= k:
            hits += 1
    return hits / len(answerable)


def mrr(results, min_score=-1.0):
    """Mean reciprocal rank: average of 1/rank of the first correct chunk.

    Rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25, a miss scores 0.
    Rewards putting the right answer near the top, not just somewhere in the list.
    """
    answerable = _answerable(results)
    if not answerable:
        return None
    total = 0.0
    for r in answerable:
        rank = first_hit_rank(r, min_score)
        total += 1.0 / rank if rank else 0.0
    return total / len(answerable)


def precision_at_k(results, k, min_score=-1.0):
    """Average share of the top-k returned chunks that come from an expected file.

    A question that returns nothing counts as 0.
    """
    answerable = _answerable(results)
    if not answerable:
        return None
    total = 0.0
    for r in answerable:
        top = _returned(r, min_score)[:k]
        if top:
            total += sum(1 for c in top if c.file_path in r.expected_files) / len(top)
    return total / len(answerable)


def refusal_accuracy(results, min_score=-1.0):
    """Fraction of unanswerable questions where NOTHING passed the threshold."""
    unanswerable = _unanswerable(results)
    if not unanswerable:
        return None
    refused = sum(1 for r in unanswerable if not _returned(r, min_score))
    return refused / len(unanswerable)


def default_thresholds():
    """0.30, 0.35, ... 0.80"""
    return [round(0.30 + 0.05 * i, 2) for i in range(11)]


def sweep(results, k, thresholds=None):
    """Score the same retrieval run at many thresholds.

    Returns a list of dicts with threshold, hit_rate, mrr, refusal_accuracy and
    balanced (the average of hit_rate and refusal_accuracy, or just hit_rate if
    there are no unanswerable questions).
    """
    rows = []
    for threshold in (thresholds if thresholds is not None else default_thresholds()):
        hits = hit_rate(results, k, threshold)
        refusal = refusal_accuracy(results, threshold)
        parts = [value for value in (hits, refusal) if value is not None]
        rows.append({
            "threshold": threshold,
            "hit_rate": hits,
            "mrr": mrr(results, threshold),
            "refusal_accuracy": refusal,
            "balanced": sum(parts) / len(parts) if parts else None,
        })
    return rows


def recommend_threshold(rows):
    """Pick the row with the best balanced score (ties: the lowest threshold).

    Treat this as a SUGGESTION. With a small question set, look at the whole
    table and prefer a threshold in the middle of a good range.
    """
    scored = [row for row in rows if row["balanced"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: (row["balanced"], -row["threshold"]))