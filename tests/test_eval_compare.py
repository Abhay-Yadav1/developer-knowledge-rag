"""Tests for src/evaluation/compare.py"""

import json

import pytest

from src.evaluation.compare import compare_reports, load_report, main


def report(statuses, hit_k=0.5, mrr=0.4, refusal=0.5, best_balanced=0.6, best_threshold=0.5):
    """Build a minimal runner-style report. statuses: {question_id: status}."""
    return {
        "summary": {"min_score": 0.5, "hit_rate_at_1": hit_k, "hit_rate_at_k": hit_k, "mrr": mrr,
                    "precision_at_k": 0.3, "refusal_accuracy": refusal},
        "sweep": [{"threshold": best_threshold, "hit_rate": hit_k, "mrr": mrr,
                   "refusal_accuracy": refusal, "balanced": best_balanced}],
        "questions": [
            {"id": qid, "status": status,
             "expected_files": [] if qid.startswith("u") else ["x.py"]}
            for qid, status in statuses.items()
        ],
    }


def test_shows_metric_changes_and_best_threshold_of_each_run():
    before = report({"q1": "HIT @3"}, hit_k=0.50, mrr=0.40, best_balanced=0.60, best_threshold=0.50)
    after = report({"q1": "HIT @1"}, hit_k=0.70, mrr=0.60, best_balanced=0.80, best_threshold=0.60)

    text = compare_reports(before, after)

    assert "+20 pts" in text            # hit rate rose from 50% to 70%
    assert "+0.20" in text              # MRR
    assert "before: 60% at threshold 0.50" in text
    assert "after:  80% at threshold 0.60" in text


def test_classifies_improved_and_worse_questions():
    before = report({"q1": "MISS (wrong files)", "q2": "HIT @1", "q3": "HIT @4",
                     "u1": "WRONGLY ANSWERED", "u2": "refused OK"})
    after = report({"q1": "HIT @2", "q2": "MISS (nothing passed)", "q3": "HIT @2",
                    "u1": "refused OK", "u2": "WRONGLY ANSWERED"})

    text = compare_reports(before, after)
    improved = text.split("Improved:")[1].split("Got worse:")[0]
    worse = text.split("Got worse:")[1]

    assert "q1" in improved and "q3" in improved and "u1" in improved
    assert "q2" in worse and "u2" in worse


def test_unchanged_questions_are_not_listed():
    both = report({"q1": "HIT @1"})

    text = compare_reports(both, both)

    assert "No per-question changes" in text
    assert "same" in text


def test_warns_when_question_sets_differ():
    before = report({"q1": "HIT @1"})
    after = report({"q1": "HIT @1", "q2": "HIT @1"})

    assert "only in the new file (q2)" in compare_reports(before, after)


def test_load_report_validates_the_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_report(tmp_path / "missing.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{ nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_report(bad)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="does not look like"):
        load_report(wrong)


def test_command_line_tool_prints_comparison(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(report({"q1": "MISS (wrong files)"})))
    b.write_text(json.dumps(report({"q1": "HIT @1"}, hit_k=0.9)))

    code = main([str(a), str(b)])

    assert code == 0
    assert "Improved:" in capsys.readouterr().out
    assert main([str(a), str(tmp_path / "nope.json")]) == 1


def test_warns_when_the_two_runs_used_different_thresholds():
    before = report({"q1": "HIT @1"})
    after = report({"q1": "HIT @1"})
    after["summary"]["min_score"] = 0.65

    text = compare_reports(before, after)

    assert "WARNING" in text and "0.5 before, 0.65 after" in text


def test_no_warning_when_thresholds_match():
    same = report({"q1": "HIT @1"})

    assert "WARNING" not in compare_reports(same, same)