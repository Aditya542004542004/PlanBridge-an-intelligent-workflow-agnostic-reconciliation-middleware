"""
confidence_gate.py
=====================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 4 DELIVERABLE — Three-Way Confidence Gate
--------------------------------------------------------
``ConfidenceGate`` takes the ranked, scored candidates from
``ScoringEngine.evaluate_candidates()`` and turns them into a final,
auditable ``MatchDecision`` by applying three confidence thresholds:

    final_confidence_score >= 0.85              -> AUTO_ACCEPT
    0.60 <= final_confidence_score < 0.85        -> HUMAN_REVIEW
    final_confidence_score < 0.60                -> UNMATCHED

This is deliberately the LAST, simplest stage in the pipeline — all the
actual judgement happened upstream in Stage 1 (entity narrowing) and Stage 2
(semantic + quantity scoring). The gate's only job is to apply a clear,
auditable business rule to a number that's already been computed, and to
explain that rule's outcome in plain language for whoever reads it (a
planner reviewing the HUMAN_REVIEW queue, or an auditor checking why an
AUTO_ACCEPT was made).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from schemas import MatchDecision, NormalizedObservation

log = logging.getLogger("planbridge.confidence_gate")

HIGH_THRESHOLD = 0.85   # >= this -> AUTO_ACCEPT
MEDIUM_THRESHOLD = 0.60  # >= this (and < HIGH_THRESHOLD) -> HUMAN_REVIEW
                          # < this -> UNMATCHED


class ConfidenceGate:
    """
    Applies the three-way confidence gate to a scored candidate list and
    produces the final, auditable ``MatchDecision``.

    Usage
    -----
        gate = ConfidenceGate()
        decision = gate.make_decision(observation, evaluated_candidates)
    """

    def __init__(self, high_threshold: float = HIGH_THRESHOLD, medium_threshold: float = MEDIUM_THRESHOLD) -> None:
        if not (0.0 <= medium_threshold < high_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: expected 0.0 <= medium_threshold ({medium_threshold}) "
                f"< high_threshold ({high_threshold}) <= 1.0."
            )
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def make_decision(
        self, observation: NormalizedObservation, evaluated_candidates: list[dict[str, Any]]
    ) -> MatchDecision:
        """
        Produce the final ``MatchDecision`` for one observation.

        Never raises solely because ``evaluated_candidates`` is empty — an
        observation with no viable candidates (e.g. pure noise/admin text
        that Stage 1 could not narrow meaningfully) correctly resolves to
        UNMATCHED with an all-zero score breakdown, not an exception.
        """
        match_id = self._generate_match_id(observation.observation_id)

        if not evaluated_candidates:
            reasoning = (
                f"No viable candidate activities were found for observation "
                f"'{observation.observation_id}' — the DPR text did not carry enough "
                f"schedule-relevant signal to narrow or score any candidates. "
                f"Marked UNMATCHED for manual triage."
            )
            log.info("ConfidenceGate: %s -> UNMATCHED (no candidates).", observation.observation_id)
            return MatchDecision(
                match_id=match_id,
                observation_id=observation.observation_id,
                selected_activity_id=None,
                entity_score=0.0,
                semantic_score=0.0,
                quantity_score=0.0,
                final_confidence_score=0.0,
                decision_type="UNMATCHED",
                reasoning=reasoning,
                candidate_scores=[],
            )

        # evaluated_candidates is expected sorted descending by ScoringEngine,
        # but we don't trust that blindly — re-sort defensively so the gate
        # is correct even if called with an unsorted list.
        ranked = sorted(evaluated_candidates, key=lambda c: c.get("final_score", 0.0), reverse=True)
        top = ranked[0]
        top_score = float(top.get("final_score", 0.0))

        decision_type, selected_activity_id = self._classify(top_score, top.get("activity_id"))
        reasoning = self._build_reasoning(observation, ranked, top, top_score, decision_type)

        log.info(
            "ConfidenceGate: %s -> %s (top candidate %s, score=%.3f).",
            observation.observation_id, decision_type, top.get("activity_id"), top_score,
        )

        return MatchDecision(
            match_id=match_id,
            observation_id=observation.observation_id,
            selected_activity_id=selected_activity_id,
            entity_score=float(top.get("entity_score", 0.0)),
            semantic_score=float(top.get("semantic_score", 0.0)),
            quantity_score=float(top.get("quantity_score", 0.0)),
            final_confidence_score=top_score,
            decision_type=decision_type,
            reasoning=reasoning,
            candidate_scores=self._sanitize_candidate_scores(ranked),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _classify(self, top_score: float, top_activity_id: Optional[str]) -> tuple[str, Optional[str]]:
        """Apply the three-way threshold rule. UNMATCHED deliberately does
        NOT carry a selected_activity_id forward — a sub-60% score means
        the pipeline is not confident enough to even tentatively commit to
        a candidate; that's a job for a human starting from scratch, not a
        pre-filled (likely wrong) suggestion."""
        if top_score >= self.high_threshold:
            return "AUTO_ACCEPT", top_activity_id
        if top_score >= self.medium_threshold:
            return "HUMAN_REVIEW", top_activity_id
        return "UNMATCHED", None

    def _build_reasoning(
        self,
        observation: NormalizedObservation,
        ranked: list[dict[str, Any]],
        top: dict[str, Any],
        top_score: float,
        decision_type: str,
    ) -> str:
        """Compose a clear, human-readable explanation of the decision,
        citing the actual score breakdown so a reviewer or auditor doesn't
        have to reverse-engineer the number."""
        breakdown = (
            f"entity={top.get('entity_score', 0.0):.1%}, "
            f"semantic={top.get('semantic_score', 0.0):.1%}, "
            f"quantity={top.get('quantity_score', 0.0):.1%}"
        )
        activity_id = top.get("activity_id", "?")
        activity_name = top.get("activity_name", "?")

        if decision_type == "AUTO_ACCEPT":
            base = (
                f"Top candidate {activity_id} ('{activity_name}') scored {top_score:.1%} confidence "
                f"({breakdown}), clearing the {self.high_threshold:.0%} auto-accept threshold. "
                f"Auto-matched without human review."
            )
        elif decision_type == "HUMAN_REVIEW":
            base = (
                f"Top candidate {activity_id} ('{activity_name}') scored {top_score:.1%} confidence "
                f"({breakdown}) — above the {self.medium_threshold:.0%} minimum but below the "
                f"{self.high_threshold:.0%} auto-accept bar. Routed to the planner review queue "
                f"for manual confirmation."
            )
        else:  # UNMATCHED
            base = (
                f"Best candidate {activity_id} ('{activity_name}') only scored {top_score:.1%} confidence "
                f"({breakdown}) — below the {self.medium_threshold:.0%} minimum threshold. "
                f"No confident match could be made; flagged UNMATCHED for manual triage."
            )

        if len(ranked) > 1:
            runner_up = ranked[1]
            base += (
                f" (Runner-up: {runner_up.get('activity_id', '?')} at "
                f"{float(runner_up.get('final_score', 0.0)):.1%}.)"
            )

        if top.get("unit_match") is False:
            base += (
                " Note: the claimed quantity's unit did not match this candidate's planned unit, "
                "which weighed against it in the quantity score."
            )

        return base

    @staticmethod
    def _sanitize_candidate_scores(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip the bulky full `activity` record out of the candidate
        breakdown before it goes into MatchDecision.candidate_scores — that
        field is meant to be a lightweight audit trail (scores per
        candidate), not a duplicate of the full activities.json payload."""
        sanitized = []
        for candidate in ranked:
            sanitized.append({k: v for k, v in candidate.items() if k != "activity"})
        return sanitized

    @staticmethod
    def _generate_match_id(observation_id: str) -> str:
        """Deterministic, reproducible match_id derived from the
        observation_id (same observation always yields the same match_id —
        useful for idempotent reprocessing/debugging), formatted in the
        'MATCH-####' style from the spec example."""
        digest = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()
        numeric = int(digest[:8], 16) % 9000 + 1000
        return f"MATCH-{numeric}"
