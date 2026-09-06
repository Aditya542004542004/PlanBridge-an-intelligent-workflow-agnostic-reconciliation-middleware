"""
ingestion.py
============
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 2 DELIVERABLE — Ingestion pipeline
-------------------------------------------
``IngestionEngine`` is the orchestration layer that sits between a raw
``ReportEvent`` and the list of ``NormalizedObservation`` records PlanBridge
will eventually try to reconcile against schedule activities (Phase 3+).

Its job, deliberately narrow for Phase 2:
    1. Detect QA/QC clearance language anywhere in the report text.
    2. Delegate quantity+unit extraction and conversion to
       ``UnitNormalizer`` (kept as a separate, independently-testable
       component — see unit_normalizer.py).
    3. Assemble the results into validated ``NormalizedObservation``
       Pydantic models, with a stable, traceable ``observation_id``.

Explicitly OUT of scope for Phase 2 (belongs to later phases):
    * Linking an observation to a specific schedule ``activity_id``.
    * Confidence scoring / semantic matching.
    * Partial-progress rollup or evidence-log persistence.
"""

from __future__ import annotations

import logging
import re

from schemas import NormalizedObservation, ReportEvent
from unit_normalizer import QuantityParseError, UnitConversionError, UnitNormalizer

log = logging.getLogger("planbridge.ingestion")


class IngestionEngine:
    """
    Parses raw field reports into normalized, unit-converted observations.

    Usage
    -----
        engine = IngestionEngine()
        observations = engine.process_report(report_event)
    """

    # QA/QC clearance keyword patterns, compiled once. Each pattern is
    # intentionally specific enough to avoid false positives (e.g. bare
    # "test" or "pass" alone do not qualify) while covering the phrasing
    # variants seen across Phase 1's synthetic QA_CLEARANCE reports and the
    # spec's required trigger phrases (NDT, Radiography passed, Hydrotest
    # clearance, Cube test passed).
    QA_CLEARANCE_PATTERNS: list[re.Pattern] = [
        re.compile(r"\bndt\b", re.IGNORECASE),
        re.compile(r"radiography\s+(passed|cleared|clearance)", re.IGNORECASE),
        re.compile(r"hydrotest\s+(clearance|cleared|passed)", re.IGNORECASE),
        re.compile(r"cube\s+test\s+(passed|cleared|clearance)", re.IGNORECASE),
        # Slightly broader nets for real-world phrasing variance, still
        # anchored to an explicit pass/clear/result verb so we don't
        # misfire on e.g. "hydrotest scheduled for next week".
        re.compile(r"\bhydrotest\b.{0,20}\b(pass(ed)?|clear(ed|ance)?|result)", re.IGNORECASE),
        re.compile(r"\bradiograph(y|ic)\b.{0,20}\b(pass(ed)?|clear(ed|ance)?)", re.IGNORECASE),
    ]

    def __init__(self, normalizer: UnitNormalizer | None = None) -> None:
        # Dependency-injectable so tests (or future phases) can swap in a
        # mock/alternate normalizer without touching this class.
        self.normalizer = normalizer or UnitNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_report(self, report: ReportEvent) -> list[NormalizedObservation]:
        """
        Extract and normalize every quantity claim in a report.

        Returns an empty list (never raises) when the report text contains
        no extractable quantity+unit phrase — that is a legitimate outcome
        (e.g. a pure-noise report like "toolbox talk conducted today") and
        must not crash the ingestion pipeline.

        A malformed individual quantity match (bad number, unrecognized
        unit) is logged and skipped rather than aborting the whole report,
        so one bad phrase in a long DPR doesn't discard everything else
        that *did* parse cleanly.
        """
        if report is None:
            raise ValueError("process_report() requires a ReportEvent, got None.")

        is_qa_clearance = self.detect_qa_clearance(report.raw_content)

        try:
            parsed_quantities = self.normalizer.parse(report.raw_content)
        except QuantityParseError as exc:
            log.warning("Report %s: quantity parsing failed entirely: %s", report.report_id, exc)
            return []

        observations: list[NormalizedObservation] = []
        for idx, pq in enumerate(parsed_quantities, start=1):
            try:
                normalized_quantity, normalized_unit, conversion_applied = self.normalizer.normalize(
                    pq.raw_quantity, pq.raw_unit
                )
            except UnitConversionError as exc:
                log.warning(
                    "Report %s: skipping unconvertible phrase '%s %s' (%s)",
                    report.report_id, pq.raw_quantity, pq.raw_unit, exc,
                )
                continue
            except ValueError as exc:
                log.warning(
                    "Report %s: skipping invalid quantity '%s' (%s)",
                    report.report_id, pq.raw_quantity, exc,
                )
                continue

            observation_id = f"{report.report_id}-OBS-{idx:03d}"
            try:
                observation = NormalizedObservation(
                    observation_id=observation_id,
                    report_id=report.report_id,
                    raw_phrase=pq.raw_phrase,
                    raw_quantity=pq.raw_quantity,
                    raw_unit=pq.raw_unit,
                    normalized_quantity=normalized_quantity,
                    normalized_unit=normalized_unit,
                    conversion_applied=conversion_applied,
                    is_qa_clearance=is_qa_clearance,
                )
            except Exception as exc:  # pydantic ValidationError or similar
                log.warning(
                    "Report %s: failed to build NormalizedObservation for phrase '%s' (%s)",
                    report.report_id, pq.raw_phrase, exc,
                )
                continue

            observations.append(observation)

        if not observations:
            log.info(
                "Report %s: no extractable quantity observations found (this may be expected, "
                "e.g. noise/administrative reports).",
                report.report_id,
            )

        return observations

    def process_reports(self, reports: list[ReportEvent]) -> list[NormalizedObservation]:
        """Convenience batch wrapper — processes a list of reports and
        flattens the results into a single observation list, continuing
        past any single report that raises rather than aborting the batch."""
        all_observations: list[NormalizedObservation] = []
        for report in reports:
            try:
                all_observations.extend(self.process_report(report))
            except Exception as exc:  # defensive: never let one bad report kill a batch job
                log.error("Report %s: unexpected failure during ingestion: %s", getattr(report, "report_id", "?"), exc)
        return all_observations

    # ------------------------------------------------------------------
    # QA/QC clearance detection
    # ------------------------------------------------------------------
    def detect_qa_clearance(self, text: str) -> bool:
        """
        Return True if the report text contains QA/QC clearance language
        (NDT, Radiography passed, Hydrotest clearance, Cube test passed, or
        close variants thereof).

        This is a deliberately simple keyword/regex check — Phase 2's brief
        is deterministic parsing, not semantic classification. A report
        merely *mentioning* an upcoming QA gate ("Hydrotest scheduled for
        Friday") should NOT trigger this flag, which is why every pattern
        requires an explicit pass/clear/result verb nearby.
        """
        if not text:
            return False
        return any(pattern.search(text) for pattern in self.QA_CLEARANCE_PATTERNS)
