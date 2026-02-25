"""Read manual review verdicts and print summary stats."""
import json
import sys
from pathlib import Path

REVIEW_FILE = Path(__file__).parent / "results" / "review.json"


def summarize():
    if not REVIEW_FILE.exists():
        print(f"No review file found at {REVIEW_FILE}")
        print("Run benchmark.py first, then fill in the verdicts.")
        sys.exit(1)

    with open(REVIEW_FILE) as f:
        review = json.load(f)

    unreviewed = [q for q in review if q["semantic_hit"] is None or q["keyword_hit"] is None]
    if unreviewed:
        ids = [str(q["id"]) for q in unreviewed]
        print(f"Queries not yet reviewed: {', '.join(ids)}")
        print("Fill in semantic_hit and keyword_hit (true/false) for all queries first.")
        sys.exit(1)

    categories = {}
    for q in review:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "semantic": 0, "keyword": 0}
        categories[cat]["count"] += 1
        if q["semantic_hit"]:
            categories[cat]["semantic"] += 1
        if q["keyword_hit"]:
            categories[cat]["keyword"] += 1

    total_sem = sum(c["semantic"] for c in categories.values())
    total_kw = sum(c["keyword"] for c in categories.values())
    total = sum(c["count"] for c in categories.values())

    print(f"\n{'Category':<20} {'Semantic':>10} {'Keyword':>10}")
    print("-" * 42)
    for cat in ["terminology_gap", "natural_language", "hard"]:
        if cat in categories:
            c = categories[cat]
            print(f"{cat:<20} {c['semantic']:>5}/{c['count']:<4} {c['keyword']:>5}/{c['count']:<4}")
    print("-" * 42)
    print(f"{'Total':<20} {total_sem:>5}/{total:<4} {total_kw:>5}/{total:<4}")

    # Per-query detail
    print(f"\n{'ID':>3}  {'Sem':>3}  {'Key':>3}  Query")
    print("-" * 60)
    for q in review:
        sem = "Y" if q["semantic_hit"] else "N"
        kw = "Y" if q["keyword_hit"] else "N"
        print(f"{q['id']:>3}   {sem:>3}  {kw:>3}  {q['query']}")


if __name__ == "__main__":
    summarize()
