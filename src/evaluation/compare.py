"""Compare two saved evaluation files: did my change help?

    python -m src.evaluation.compare evaluation/results/baseline.json results_new.json

Workflow for every experiment:
    1. change something (chunking, embedding input, search method...)
    2. rebuild the index
    3. python -m src.evaluation.runner <questions> --output results_new.json
    4. run this tool to see exactly what improved and what got worse
"""
import argparse
import json
import sys
from pathlib import Path

def load_report(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"Results file not found: {path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Results file is not valid JSON: {error}")
    for key in ("summary", "sweep", "questions"):
        if key not in data:
            raise ValueError(f'{path} does not look like a runner output (missing "{key}").')
    return data


def _best_row(report):
    rows = [r for r in report["sweep"] if r.get("balanced") is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: (r["balanced"], -r["threshold"]))


def _fmt(value, pct=False):
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%" if pct else f"{value:.2f}"


def _delta(old, new, pct=False):
    if old is None or new is None:
        return ""
    diff = new - old
    if abs(diff) < 1e-9:
        return "same"
    return f"{diff * 100:+.0f} pts" if pct else f"{diff:+.2f}"


def _rank(status):
    """Turn 'HIT @3' into 3, a miss into None, so changes can be classified."""
    if status.startswith("HIT @"):
        return int(status.split("@")[1])
    return None


def compare_reports(old, new):
    lines = []
    s_old, s_new = old["summary"], new["summary"]

    if s_old.get("min_score") != s_new.get("min_score"):
        lines += [
            f"WARNING: the two runs used different thresholds "
            f"({s_old.get('min_score')} before, {s_new.get('min_score')} after).",
            "Per-question statuses and the refusal row are NOT like-for-like. Compare the",
            "'best balanced score' below, or rerun one of them with the same --min-score.",
            "",
        ]

    lines.append(f"{'metric':<22}{'before':>9}{'after':>9}  change")
    for label, key, pct in [
        ("hit rate @1", "hit_rate_at_1", True),
        ("hit rate @k", "hit_rate_at_k", True),
        ("MRR", "mrr", False),
        ("precision @k", "precision_at_k", True),
        (f"refusal acc. (t={s_new['min_score']})", "refusal_accuracy", True),
    ]:
        lines.append(f"{label:<22}{_fmt(s_old.get(key), pct):>9}{_fmt(s_new.get(key), pct):>9}  "
                     f"{_delta(s_old.get(key), s_new.get(key), pct)}")

    # Each experiment has its own best threshold, so compare the best of each.
    b_old, b_new = _best_row(old), _best_row(new)
    if b_old and b_new:
        lines += ["",
                  "Best balanced score over the threshold sweep (fair comparison):",
                  f"  before: {_fmt(b_old['balanced'], True)} at threshold {b_old['threshold']:.2f}",
                  f"  after:  {_fmt(b_new['balanced'], True)} at threshold {b_new['threshold']:.2f}   "
                  f"({_delta(b_old['balanced'], b_new['balanced'], True)})"]

    old_q = {q["id"]: q for q in old["questions"]}
    improved, worse, moved = [], [], []
    for q in new["questions"]:
        before = old_q.get(q["id"])
        if before is None or before["status"] == q["status"]:
            continue
        line = f"  {q['id']}: {before['status']}  ->  {q['status']}"
        r_old, r_new = _rank(before["status"]), _rank(q["status"])
        answerable = bool(q["expected_files"])
        if answerable:
            found_now = r_old is None and r_new is not None
            lost_now = r_old is not None and r_new is None
            ranked_higher = r_old is not None and r_new is not None and r_new < r_old
            ranked_lower = r_old is not None and r_new is not None and r_new > r_old
            if found_now or ranked_higher:
                improved.append(line)
            elif lost_now or ranked_lower:
                worse.append(line)
            else:
                moved.append(line)
        else:
            (improved if q["status"] == "refused OK" else worse).append(line)

    for title, group in (("Improved", improved), ("Got worse", worse), ("Changed (neither better nor worse)", moved)):
        if group:
            lines += ["", f"{title}:"] + group
    if not (improved or worse or moved):
        lines += ["", "No per-question changes at the saved thresholds."]

    new_ids = set(q["id"] for q in new["questions"]) - set(old_q)
    if new_ids:
        lines += ["", f"Note: {len(new_ids)} question(s) exist only in the new file "
                      f"({', '.join(sorted(new_ids))}); the overall numbers are not like-for-like."]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two evaluation result files.")
    parser.add_argument("before", help="older results JSON (for example the baseline)")
    parser.add_argument("after", help="newer results JSON")
    args = parser.parse_args(argv)
    try:
        print(compare_reports(load_report(args.before), load_report(args.after)))
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())