"""Print the latest saved evaluation as a one-screen terminal summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BAR_WIDTH = 24


def score_row(label: str, value: float, ideal: float) -> str:
    gap = value - ideal
    if abs(gap) < 0.0005:
        result = "PERFECT"
    elif ideal >= value:
        result = f"-{abs(gap) * 100:.1f}pp short"
    else:
        result = f"+{abs(gap) * 100:.1f}pp over"
    filled = round(BAR_WIDTH * min(max(value, 0.0), 1.0))
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    return f"  {label:<28}{value:>7.3f}  [{bar}]  ideal {ideal:.3f}  {result}"


def latest_result() -> Path:
    candidates = sorted(
        (ROOT / "results").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No evaluation JSON found in results/.")
    return candidates[0]


def main() -> None:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_result()
    report = json.loads(path.read_text(encoding="utf-8"))
    meta = report["meta"]
    retrieval = report["retrieval"]
    faithful = report["faithfulness"]
    correct = report["correctness"]
    reliable = report["reliability"]
    latency = report["latency"]

    print("=" * 76)
    print("RAG LOCAL EVAL LOOP - COMPACT FINAL RESULTS")
    print("=" * 76)
    print(f"Result:  {path.name}")
    print(f"Dataset: ai4bharat/MSMARCO-XI ({meta['language']}, {meta['split']})")
    print(
        f"Sample:  {meta['num_answerable']} answerable + "
        f"{meta['num_unanswerable']} unanswerable | seed={meta['seed']} | "
        f"top_k={meta['top_k']}"
    )
    print(
        f"Index:   {meta['num_chunks']} chunks from {meta['num_examples']} examples | "
        f"errored={meta['num_errored']}"
    )

    print(f"\nRETRIEVAL ({retrieval['num_evaluated']} answerable queries evaluated)")
    for key, label in (
        ("cross_lingual", "cross-lingual (either language is a hit)"),
        ("same_language", "same-language only"),
    ):
        values = retrieval[key]
        recall = values["recall_at_k"]
        print(f"\n  {label}:")
        print(score_row("Recall@1", recall["1"], 1.0))
        print(score_row("Recall@3", recall["3"], 1.0))
        print(score_row("Recall@5", recall["5"], 1.0))
        print(score_row("MRR", values["mrr"], 1.0))

    print("\nFAITHFULNESS / HALLUCINATION")
    print(f"  {faithful['num_evaluated']} answers evaluated")
    print(score_row("Faithful rate", faithful["faithful_rate"], 1.0))
    print(score_row("Hallucination rate", faithful["hallucination_rate"], 0.0))
    print(score_row("Self-report precision", faithful["self_report_precision"], 1.0))

    print("\nCORRECTNESS")
    print(f"  {correct['num_evaluated']} answerable-query answers evaluated")
    print(score_row("Correct rate", correct["correct_rate"], 1.0))

    print('\nRELIABILITY / "LYING FACTOR"')
    print(score_row("False refusal rate", reliable["false_refusal_rate"], 0.0))
    print(score_row("False confidence rate", reliable["false_confidence_rate"], 0.0))

    print("\nLATENCY (milliseconds)")
    print(f"  {'Stage':<17}{'Avg':>8}{'P50':>8}{'P95':>8}{'P99':>8}")
    for stage in ("embed", "search", "retrieval_total", "generation"):
        values = latency[stage]
        print(
            f"  {stage:<17}{values['avg_ms']:>8.2f}{values['p50_ms']:>8.2f}"
            f"{values['p95_ms']:>8.2f}{values['p99_ms']:>8.2f}"
        )
    retrieval_status = "PASS" if latency["retrieval_within_budget"] else "OVER BUDGET"
    generation_status = "PASS" if latency["generation_within_target"] else "OVER TARGET"
    print(
        f"  Retrieval P95 {latency['retrieval_total']['p95_ms']:.2f} vs "
        f"{latency['retrieval_latency_budget_ms']:.1f} budget: {retrieval_status}"
    )
    print(
        f"  Generation P95 {latency['generation']['p95_ms']:.2f} vs "
        f"{latency['generation_latency_target_ms']:.1f} target: {generation_status}"
    )
    print("=" * 76)
    print("Full examples and audit data remain unchanged in the JSON result above.")


if __name__ == "__main__":
    main()
