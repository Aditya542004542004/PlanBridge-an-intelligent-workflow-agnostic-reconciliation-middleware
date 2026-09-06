"""
test_benchmark.py
====================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 8 DELIVERABLE — End-to-End Safety Benchmark
--------------------------------------------------------
Runs every synthetic DPR in ``data/dprs.json`` (Phase 1's labeled benchmark
dataset — 100+ cases spanning EASY_MATCH, AMBIGUOUS_MEDIUM, UNIT_MISMATCH,
QA_CLEARANCE, and UNMATCHED_NOISE) through the full matching pipeline:

    IngestionEngine -> UnitNormalizer -> EntityExtractor ->
    CandidateNarrower -> VectorRanker -> ScoringEngine -> ConfidenceGate

...and reports the one metric that matters most for a CVC/CAG-audited PSU
system: the FALSE AUTO-ACCEPT RATE — the fraction of ambiguous/noise cases
the pipeline wrongly committed to automatically. This is expected to be
exactly 0.00%; the benchmark exits non-zero if it isn't, so it can be wired
into CI as a hard safety gate, not just an informational report.

Usage
-----
    python test_benchmark.py
    python test_benchmark.py --verbose
    python test_benchmark.py --dprs-path data/dprs.json --activities-path data/activities.json
    pytest test_benchmark.py        # also runnable as a single pytest case

Ground-truth correctness model
-------------------------------------
Every DPR carries an ``expected_activity_id`` (or ``null``) and a
``case_type`` from Phase 1. Correctness is judged per case_type:

  * EASY_MATCH / UNIT_MISMATCH / QA_CLEARANCE (a real match exists):
      CORRECT             — AUTO_ACCEPT with the right activity_id
      FALSE_ACCEPT_WRONG   — AUTO_ACCEPT with the WRONG activity_id (unsafe)
      SUBOPTIMAL           — HUMAN_REVIEW or UNMATCHED (safe, just not ideal)
      NO_OBSERVATION       — no quantity extracted at all (never reached matching)

  * AMBIGUOUS_MEDIUM (deliberately hard to resolve automatically):
      CORRECT             — HUMAN_REVIEW (routed for human judgement, as intended)
      FALSE_ACCEPT         — AUTO_ACCEPT (unsafe — this is exactly what the
                              confidence gate exists to prevent)
      SUBOPTIMAL           — UNMATCHED (safe, just overly conservative)
      NO_OBSERVATION       — no quantity extracted at all

  * UNMATCHED_NOISE (no real activity exists to match):
      CORRECT             — UNMATCHED or NO_OBSERVATION (nothing false committed)
      FALSE_ACCEPT         — AUTO_ACCEPT (unsafe — a real false positive)
      SUBOPTIMAL           — HUMAN_REVIEW (safe, just wastes a planner's time)

The spec's literal "FALSE AUTO-ACCEPT RATE" is computed over the
AMBIGUOUS_MEDIUM + UNMATCHED_NOISE cases specifically (per the task
description). This module additionally reports a stricter supplementary
metric — AUTO_ACCEPT decisions that picked the wrong activity even on
real-match case types — since silently corrupting a real activity's
progress is arguably just as dangerous as accepting pure noise.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from candidate_narrower import CandidateNarrower
from confidence_gate import ConfidenceGate
from entity_extractor import EntityExtractor
from ingestion import IngestionEngine
from schemas import ReportEvent
from scoring_engine import ScoringEngine
from unit_normalizer import UnitNormalizer
from vector_ranker import VectorRanker

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ACTIVITIES_PATH = BASE_DIR / "data" / "activities.json"
DEFAULT_DPRS_PATH = BASE_DIR / "data" / "dprs.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "benchmark_results.json"

# Case-type groupings used for correctness classification.
REAL_MATCH_CASE_TYPES = {"EASY_MATCH", "UNIT_MISMATCH", "QA_CLEARANCE"}
AMBIGUOUS_CASE_TYPES = {"AMBIGUOUS_MEDIUM"}
NOISE_CASE_TYPES = {"UNMATCHED_NOISE"}

logging.getLogger().setLevel(logging.CRITICAL)  # keep pipeline component logging out of benchmark output


@dataclass
class CaseResult:
    """Per-DPR benchmark outcome, one entry per test case."""
    report_id: str
    case_type: str
    expected_activity_id: Optional[str]
    raw_content: str
    observations_processed: int
    decision_type: Optional[str]
    selected_activity_id: Optional[str]
    final_confidence_score: Optional[float]
    outcome: str          # CORRECT | FALSE_ACCEPT | FALSE_ACCEPT_WRONG | SUBOPTIMAL | NO_OBSERVATION
    latency_ms: float


class BenchmarkPipeline:
    """
    Thin orchestration wrapper around the real Phase 2-4 components, built
    once and reused across every test case (VectorRanker's model load is
    expensive and must not be repeated per-case, which would both slow the
    benchmark down and corrupt the latency measurement with one-time
    warm-up cost).
    """

    def __init__(self, activities: list[dict[str, Any]]) -> None:
        self.ingestion = IngestionEngine(UnitNormalizer())
        self.extractor = EntityExtractor()
        self.narrower = CandidateNarrower(activities)
        self.ranker = VectorRanker()
        self.scorer = ScoringEngine(vector_ranker=self.ranker)
        self.gate = ConfidenceGate()

    def process(self, report: ReportEvent) -> tuple[int, Optional[dict[str, Any]]]:
        """
        Run one report through the full pipeline. Returns
        (observations_processed, first_decision_dict_or_None) — mirroring
        the same "take the first observation as representative" convention
        established by Phase 6's IngestResponse convenience fields, since
        Phase 1's benchmark DPRs are constructed to carry at most one
        quantity claim each.
        """
        observations = self.ingestion.process_report(report)
        if not observations:
            return 0, None

        obs = observations[0]
        entities = self.extractor.extract(obs.raw_phrase)
        shortlist = self.narrower.narrow_candidates(obs, entities)
        scored = self.scorer.evaluate_candidates(obs, shortlist)
        decision = self.gate.make_decision(obs, scored)
        return len(observations), {
            "decision_type": decision.decision_type,
            "selected_activity_id": decision.selected_activity_id,
            "final_confidence_score": decision.final_confidence_score,
        }


def classify_outcome(case_type: str, expected_activity_id: Optional[str], observations_processed: int, decision: Optional[dict[str, Any]]) -> str:
    """
    Judge one case's outcome against the ground-truth model described in
    this module's docstring. Never raises — an unrecognized case_type
    falls back to the same logic as a real-match case type, since that is
    the safer (stricter) default.
    """
    if observations_processed == 0 or decision is None:
        return "NO_OBSERVATION"

    decision_type = decision["decision_type"]
    selected = decision["selected_activity_id"]

    if case_type in AMBIGUOUS_CASE_TYPES:
        if decision_type == "AUTO_ACCEPT":
            return "FALSE_ACCEPT"
        if decision_type == "HUMAN_REVIEW":
            return "CORRECT"
        return "SUBOPTIMAL"  # UNMATCHED

    if case_type in NOISE_CASE_TYPES:
        if decision_type == "AUTO_ACCEPT":
            return "FALSE_ACCEPT"
        if decision_type == "UNMATCHED":
            return "CORRECT"
        return "SUBOPTIMAL"  # HUMAN_REVIEW

    # REAL_MATCH_CASE_TYPES and any unrecognized case_type (safer default).
    if decision_type == "AUTO_ACCEPT":
        return "CORRECT" if selected == expected_activity_id else "FALSE_ACCEPT_WRONG"
    return "SUBOPTIMAL"  # HUMAN_REVIEW or UNMATCHED


def run_benchmark(activities_path: Path, dprs_path: Path, verbose: bool = False) -> dict[str, Any]:
    """
    Execute the full benchmark and return a JSON-serializable results dict
    (the same structure written to ``data/benchmark_results.json``).

    Raises
    ------
    FileNotFoundError : if the activities or DPR dataset is missing —
        callers (including __main__ below) should catch this and print a
        clear instruction to run synthetic_generator.py first, rather than
        letting a bare traceback surface.
    """
    if not activities_path.exists():
        raise FileNotFoundError(f"Activities dataset not found at {activities_path}. Run synthetic_generator.py first.")
    if not dprs_path.exists():
        raise FileNotFoundError(f"DPR dataset not found at {dprs_path}. Run synthetic_generator.py first.")

    with activities_path.open("r", encoding="utf-8") as f:
        activities = json.load(f)
    with dprs_path.open("r", encoding="utf-8") as f:
        dprs = json.load(f)

    print(f"Loading pipeline components ({len(activities)} activities)...")
    warmup_start = time.perf_counter()
    pipeline = BenchmarkPipeline(activities)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    print(f"Pipeline ready in {warmup_ms:.0f} ms (VectorRanker backend: {pipeline.ranker.backend_name}).\n")

    results: list[CaseResult] = []
    print(f"Processing {len(dprs)} test cases...")

    for i, dpr in enumerate(dprs, start=1):
        report = ReportEvent(
            report_id=dpr["report_id"],
            source_type=dpr["source_type"],
            submitted_by=dpr.get("submitted_by", "Benchmark Harness"),
            submission_timestamp=dpr["submission_timestamp"],
            raw_content=dpr["raw_content"],
        )

        start = time.perf_counter()
        try:
            observations_processed, decision = pipeline.process(report)
            error: Optional[str] = None
        except Exception as exc:  # a pipeline crash on any single case must not abort the whole benchmark
            observations_processed, decision, error = 0, None, str(exc)
        latency_ms = (time.perf_counter() - start) * 1000

        if error is not None:
            outcome = "PIPELINE_ERROR"
        else:
            outcome = classify_outcome(dpr["case_type"], dpr.get("expected_activity_id"), observations_processed, decision)

        results.append(CaseResult(
            report_id=dpr["report_id"],
            case_type=dpr["case_type"],
            expected_activity_id=dpr.get("expected_activity_id"),
            raw_content=dpr["raw_content"],
            observations_processed=observations_processed,
            decision_type=decision["decision_type"] if decision else None,
            selected_activity_id=decision["selected_activity_id"] if decision else None,
            final_confidence_score=decision["final_confidence_score"] if decision else None,
            outcome=outcome,
            latency_ms=round(latency_ms, 3),
        ))

        if verbose and outcome in ("FALSE_ACCEPT", "FALSE_ACCEPT_WRONG", "PIPELINE_ERROR"):
            print(f"  [{outcome}] {dpr['report_id']} ({dpr['case_type']}): {dpr['raw_content'][:70]!r}")

        if i % 20 == 0 or i == len(dprs):
            print(f"  ...{i}/{len(dprs)} processed")

    print()
    return build_report(results, warmup_ms, str(activities_path), str(dprs_path), pipeline.ranker.backend_name)


def build_report(
    results: list[CaseResult], warmup_ms: float, activities_path: str, dprs_path: str, backend_name: str
) -> dict[str, Any]:
    """Aggregate per-case results into the full summary report structure."""
    total = len(results)

    def subset(case_types: set[str]) -> list[CaseResult]:
        return [r for r in results if r.case_type in case_types]

    def pct(numerator: int, denominator: int) -> Optional[float]:
        return round(100.0 * numerator / denominator, 2) if denominator else None

    easy = subset({"EASY_MATCH"})
    ambiguous = subset(AMBIGUOUS_CASE_TYPES)
    noise = subset(NOISE_CASE_TYPES)
    unit_mismatch = subset({"UNIT_MISMATCH"})
    qa_clearance = subset({"QA_CLEARANCE"})

    def decided(case_results: list[CaseResult]) -> list[CaseResult]:
        """Cases that actually reached a gate decision -- excludes
        NO_OBSERVATION cases, which never got a chance to be right or
        wrong (that's a data characteristic of the report text, not a
        matching-quality failure) and would otherwise unfairly deflate
        the "correctly queued/rejected" percentages below."""
        return [r for r in case_results if r.outcome != "NO_OBSERVATION"]

    easy_decided = decided(easy)
    ambiguous_decided = decided(ambiguous)
    noise_decided = decided(noise)

    easy_correct = sum(1 for r in easy_decided if r.outcome == "CORRECT")
    ambiguous_correct = sum(1 for r in ambiguous_decided if r.outcome == "CORRECT")
    noise_correct = sum(1 for r in noise_decided if r.outcome == "CORRECT")

    # The spec's literal metric: ambiguous/noise cases wrongly AUTO_ACCEPTed.
    ambiguous_noise = ambiguous + noise
    false_accepts_ambiguous_noise = sum(1 for r in ambiguous_noise if r.outcome == "FALSE_ACCEPT")
    false_auto_accept_rate = pct(false_accepts_ambiguous_noise, len(ambiguous_noise))

    # Supplementary, stricter safety metric: ANY auto-accept that picked
    # the wrong activity, including on real-match case types.
    wrong_match_auto_accepts = sum(1 for r in results if r.outcome == "FALSE_ACCEPT_WRONG")
    total_auto_accepts = sum(1 for r in results if r.decision_type == "AUTO_ACCEPT")
    any_false_accept_rate = pct(false_accepts_ambiguous_noise + wrong_match_auto_accepts, total_auto_accepts) if total_auto_accepts else 0.0

    pipeline_errors = sum(1 for r in results if r.outcome == "PIPELINE_ERROR")
    no_observation = sum(1 for r in results if r.outcome == "NO_OBSERVATION")

    latencies = [r.latency_ms for r in results if r.outcome != "PIPELINE_ERROR"]
    avg_latency = round(statistics.mean(latencies), 3) if latencies else 0.0
    p95_latency = round(statistics.quantiles(latencies, n=20)[18], 3) if len(latencies) >= 20 else (round(max(latencies), 3) if latencies else 0.0)

    def case_type_breakdown(case_results: list[CaseResult]) -> dict[str, Any]:
        n = len(case_results)
        by_outcome: dict[str, int] = {}
        for r in case_results:
            by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
        return {"count": n, "by_outcome": by_outcome}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"activities_path": activities_path, "dprs_path": dprs_path, "total_cases": total},
        "engine": {"vector_ranker_backend": backend_name, "warmup_ms": round(warmup_ms, 1)},
        "headline_metrics": {
            "total_test_cases_processed": total,
            "easy_cases_correctly_auto_accepted_pct": pct(easy_correct, len(easy_decided)),
            "easy_cases_n": f"{easy_correct}/{len(easy_decided)}",
            "ambiguous_cases_correctly_queued_pct": pct(ambiguous_correct, len(ambiguous_decided)),
            "ambiguous_cases_n": f"{ambiguous_correct}/{len(ambiguous_decided)}",
            "noise_cases_correctly_rejected_pct": pct(noise_correct, len(noise_decided)),
            "noise_cases_n": f"{noise_correct}/{len(noise_decided)}",
            "false_auto_accept_rate_pct": false_auto_accept_rate,
            "false_auto_accept_n": f"{false_accepts_ambiguous_noise}/{len(ambiguous_noise)}",
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
        },
        "headline_metrics_denominator_note": (
            "easy/ambiguous/noise percentages above are computed over cases that actually reached a "
            "gate decision (observations_processed > 0); cases with no extractable quantity never reach "
            "matching and are reported separately under no_observation_extracted, not folded into these rates."
        ),
        "supplementary_safety_metrics": {
            "wrong_activity_auto_accepts_on_real_match_cases": wrong_match_auto_accepts,
            "any_false_accept_rate_pct_of_all_auto_accepts": any_false_accept_rate,
            "total_auto_accepts": total_auto_accepts,
            "pipeline_errors": pipeline_errors,
            "no_observation_extracted": no_observation,
        },
        "case_type_breakdown": {
            "EASY_MATCH": case_type_breakdown(easy),
            "AMBIGUOUS_MEDIUM": case_type_breakdown(ambiguous),
            "UNIT_MISMATCH": case_type_breakdown(unit_mismatch),
            "QA_CLEARANCE": case_type_breakdown(qa_clearance),
            "UNMATCHED_NOISE": case_type_breakdown(noise),
        },
        "safety_gate_passed": false_accepts_ambiguous_noise == 0,
        "per_case_results": [asdict(r) for r in results],
    }
    return report


# --------------------------------------------------------------------------
# Terminal report printing
# --------------------------------------------------------------------------
def print_report(report: dict[str, Any]) -> None:
    W = 80
    def rule(char: str = "-") -> None: print(char * W)
    def title(text: str) -> None: print(f" {text}")
    def fmt_pct(value: Optional[float]) -> str:
        return "   N/A" if value is None else f"{value:>6.2f}%"

    print("=" * W)
    title("PLANBRIDGE BENCHMARK REPORT")
    title(f"Generated: {report['generated_at']}")
    print("=" * W)
    print(f" Dataset:          {report['dataset']['dprs_path']} ({report['dataset']['total_cases']} cases)")
    print(f" Activities:       {report['dataset']['activities_path']}")
    print(f" Semantic backend: {report['engine']['vector_ranker_backend']}  (warm-up {report['engine']['warmup_ms']} ms, excluded from latency stats)")
    print()

    rule()
    title("HEADLINE SAFETY & PERFORMANCE METRICS")
    rule()
    m = report["headline_metrics"]
    print(f" Total Test Cases Processed .......................... {m['total_test_cases_processed']}")
    print(f" Easy Cases Correctly Auto-Accepted ................... {fmt_pct(m['easy_cases_correctly_auto_accepted_pct'])}  ({m['easy_cases_n']})")
    print(f" Ambiguous Cases Correctly Queued for Human Review .... {fmt_pct(m['ambiguous_cases_correctly_queued_pct'])}  ({m['ambiguous_cases_n']})")
    print(f" Noise Cases Correctly Rejected ........................ {fmt_pct(m['noise_cases_correctly_rejected_pct'])}  ({m['noise_cases_n']})")
    status = "PASS" if report["safety_gate_passed"] else "FAIL"
    print(f" FALSE AUTO-ACCEPT RATE (ambiguous+noise) ............. {fmt_pct(m['false_auto_accept_rate_pct'])}  ({m['false_auto_accept_n']})  [{status}]")
    print(f" Average End-to-End Latency per Report ................ {m['average_latency_ms']:.2f} ms  (p95: {m['p95_latency_ms']:.2f} ms)")
    print(f" (% rates above are over cases that reached a decision -- see NO_OBSERVATION notes below)")
    print()

    rule()
    title("SUPPLEMENTARY SAFETY METRICS")
    rule()
    s = report["supplementary_safety_metrics"]
    print(f" Wrong-activity auto-accepts on real-match cases ...... {s['wrong_activity_auto_accepts_on_real_match_cases']}")
    print(f" Any-false-accept rate (of all AUTO_ACCEPT decisions) . {s['any_false_accept_rate_pct_of_all_auto_accepts']:.2f}%  ({s['total_auto_accepts']} total auto-accepts)")
    print(f" Pipeline errors (exceptions during processing) ....... {s['pipeline_errors']}")
    print(f" Cases with no extractable quantity (no observation) .. {s['no_observation_extracted']}")
    print()

    rule()
    title("PER-CASE-TYPE BREAKDOWN")
    rule()
    header = f" {'Case Type':<18}{'N':>5}   Outcomes"
    print(header)
    for case_type, data in report["case_type_breakdown"].items():
        outcomes_str = ", ".join(f"{k}={v}" for k, v in sorted(data["by_outcome"].items()))
        print(f" {case_type:<18}{data['count']:>5}   {outcomes_str}")
    print()

    rule("=")
    title("SAFETY VERDICT")
    rule("=")
    if report["safety_gate_passed"]:
        print(f" [PASS] False auto-accept rate is {fmt_pct(m['false_auto_accept_rate_pct']).strip()} -- zero unsafe automatic")
        print("        matches across all ambiguous and noise test cases.")
    else:
        print(f" [FAIL] False auto-accept rate is {fmt_pct(m['false_auto_accept_rate_pct']).strip()} -- unsafe automatic matches")
        print("        were made on ambiguous or noise cases. This MUST be investigated")
        print("        before this system is trusted with real progress updates.")
    if s["wrong_activity_auto_accepts_on_real_match_cases"] > 0:
        print(f" [WARN] {s['wrong_activity_auto_accepts_on_real_match_cases']} case(s) auto-accepted a real-match report")
        print("        against the WRONG activity_id -- see per_case_results in the")
        print("        saved JSON report for details.")
    if s["pipeline_errors"] > 0:
        print(f" [WARN] {s['pipeline_errors']} case(s) raised an exception during processing --")
        print("        treated as failures, not silently skipped. See saved JSON report.")
    if s["no_observation_extracted"] > 0:
        print(f" [INFO] {s['no_observation_extracted']} case(s) had no extractable quantity and never")
        print("        reached the matching stage (excluded from decision-based accuracy")
        print("        denominators above) -- this is expected for purely descriptive text.")
    print("=" * W)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="PlanBridge end-to-end safety benchmark.")
    parser.add_argument("--activities-path", type=Path, default=DEFAULT_ACTIVITIES_PATH)
    parser.add_argument("--dprs-path", type=Path, default=DEFAULT_DPRS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--verbose", action="store_true", help="Print each FALSE_ACCEPT / error case as it's found.")
    args = parser.parse_args()

    try:
        report = run_benchmark(args.activities_path, args.dprs_path, verbose=args.verbose)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(report)

    try:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report (including every per-case result) saved to: {args.output_path}")
    except OSError as exc:
        print(f"\nWARNING: could not save benchmark report to {args.output_path}: {exc}", file=sys.stderr)

    return 0 if report["safety_gate_passed"] else 1


# --------------------------------------------------------------------------
# pytest entry point — `pytest test_benchmark.py` runs this as a single
# hard safety-gate test, in addition to `python test_benchmark.py` giving
# the full human-readable report.
# --------------------------------------------------------------------------
def test_false_auto_accept_rate_is_zero() -> None:
    report = run_benchmark(DEFAULT_ACTIVITIES_PATH, DEFAULT_DPRS_PATH, verbose=False)
    print_report(report)
    assert report["safety_gate_passed"], (
        f"False auto-accept rate is {report['headline_metrics']['false_auto_accept_rate_pct']}% "
        f"({report['headline_metrics']['false_auto_accept_n']}) -- expected 0.00%."
    )


if __name__ == "__main__":
    sys.exit(main())
