"""Tests for src/evaluation/metrics.py (pure functions, hand-checkable numbers)."""

import pytest

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


def cand(path, score):
    return Candidate(path, 1, 5, score)


def answerable(qid, expected, candidates):
    return QuestionResult(qid, "q?", "general", tuple(expected), tuple(candidates))


def unanswerable(qid, candidates):
    return QuestionResult(qid, "q?", "unanswerable", (), tuple(candidates))


@pytest.fixture
def results():
    return [
        # right file ranked 1st
        answerable("a", ["auth.py"], [cand("auth.py", 0.80), cand("db.py", 0.50)]),
        # right file ranked 3rd, with a low score
        answerable("b", ["db.py"], [cand("x.py", 0.70), cand("y.py", 0.60), cand("db.py", 0.45)]),
        # right file never retrieved
        answerable("c", ["api.py"], [cand("x.py", 0.55), cand("y.py", 0.40)]),
        # unanswerable, best match is weak
        unanswerable("u1", [cand("x.py", 0.35), cand("y.py", 0.30)]),
        # unanswerable, but a fairly strong (misleading) match exists
        unanswerable("u2", [cand("x.py", 0.62)]),
    ]


def test_first_hit_rank(results):
    assert first_hit_rank(results[0]) == 1
    assert first_hit_rank(results[1]) == 3
    assert first_hit_rank(results[2]) is None


def test_threshold_removes_candidates_before_ranking(results):
    # With threshold 0.5, question b's correct chunk (0.45) is no longer returned.
    assert first_hit_rank(results[1], min_score=0.5) is None
    # Removing weak candidates can also IMPROVE rank: here the first two stay.
    assert first_hit_rank(results[0], min_score=0.5) == 1


def test_hit_rate_counts_only_answerable_questions(results):
    assert hit_rate(results, k=1) == pytest.approx(1 / 3)   # only "a"
    assert hit_rate(results, k=3) == pytest.approx(2 / 3)   # "a" and "b"
    assert hit_rate(results, k=3, min_score=0.5) == pytest.approx(1 / 3)


def test_mrr_averages_reciprocal_ranks(results):
    # a: 1/1, b: 1/3, c: 0  ->  (1 + 0.333 + 0) / 3
    assert mrr(results) == pytest.approx((1 + 1 / 3) / 3)


def test_precision_at_k(results):
    # a: 1 of 2 correct = 0.5 ; b: 1 of 3 = 0.333 ; c: 0 of 2 = 0
    assert precision_at_k(results, k=5) == pytest.approx((0.5 + 1 / 3 + 0) / 3)
    # nothing returned counts as 0, not as an error
    assert precision_at_k(results, k=5, min_score=0.99) == 0.0


def test_refusal_accuracy(results):
    assert refusal_accuracy(results, min_score=0.3) == pytest.approx(0.0)   # both still return something
    assert refusal_accuracy(results, min_score=0.5) == pytest.approx(0.5)   # u1 refused, u2 (0.62) not
    assert refusal_accuracy(results, min_score=0.7) == pytest.approx(1.0)


def test_metrics_return_none_when_there_is_nothing_to_measure():
    only_unanswerable = [unanswerable("u", [cand("x.py", 0.4)])]
    only_answerable = [answerable("a", ["x.py"], [cand("x.py", 0.9)])]

    assert hit_rate(only_unanswerable, 3) is None
    assert mrr(only_unanswerable) is None
    assert refusal_accuracy(only_answerable) is None


def test_sweep_has_one_row_per_threshold_with_balanced_score(results):
    rows = sweep(results, k=3, thresholds=[0.3, 0.5, 0.7])

    assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.7]
    row = rows[1]
    assert row["hit_rate"] == pytest.approx(1 / 3)
    assert row["refusal_accuracy"] == pytest.approx(0.5)
    assert row["balanced"] == pytest.approx((1 / 3 + 0.5) / 2)


def test_default_sweep_covers_a_sensible_range(results):
    rows = sweep(results, k=3)

    assert rows[0]["threshold"] == 0.30 and rows[-1]["threshold"] == 0.80
    assert len(rows) == 11


def test_recommend_threshold_picks_best_balance_and_prefers_lower_on_ties():
    rows = [
        {"threshold": 0.4, "balanced": 0.60},
        {"threshold": 0.5, "balanced": 0.80},
        {"threshold": 0.6, "balanced": 0.80},
        {"threshold": 0.7, "balanced": 0.50},
    ]

    assert recommend_threshold(rows)["threshold"] == 0.5


def test_recommend_threshold_with_no_data_returns_none():
    assert recommend_threshold([{"threshold": 0.5, "balanced": None}]) is None