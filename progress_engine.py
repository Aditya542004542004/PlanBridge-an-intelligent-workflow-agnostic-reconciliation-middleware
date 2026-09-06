"""
progress_engine.py
=====================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 5 DELIVERABLE — Dual Progress Tracking Engine
--------------------------------------------------------
``ProgressEngine`` maintains the current ``ProgressState`` of every
Primavera P6 schedule activity, and is the single place where a matched
observation actually changes a number that matters — everything upstream
(Phases 2-4) is about deciding *whether* and *what* to match; this is where
that decision becomes tracked progress.

The core design decision this module encodes: PHYSICAL claims and QA-VERIFIED
earned progress are two different numbers, updated by two different kinds of
evidence, and neither can be conflated into the other:

    * A site supervisor's DPR ("150m HDD drilling finished") claims physical
      execution happened. It moves ``physical_claimed_quantity``.
    * A QA/QC report ("NDT Radiography passed") does NOT claim new physical
      work — it *certifies* that previously-claimed physical work is now
      independently verified. It moves ``verified_earned_quantity``, and is
      capped so it can never exceed what has actually been physically
      claimed (QA cannot verify work nobody has claimed doing).

This separation is exactly what a CVC/CAG audit of a PSU like Oil India
would expect to see: "what the site says is done" and "what's independently
verified" are not allowed to silently become the same number.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from schemas import MatchDecision, NormalizedObservation, ProgressState
from unit_normalizer import convert_for_comparison

log = logging.getLogger("planbridge.progress_engine")

# Sentinel used for ProgressState.qa_gate_type when an activity has no QA
# gate requirement — the schema field is a plain (non-Optional) str per
# spec, so we need a concrete value rather than None.
NO_QA_GATE_SENTINEL = "NONE"


class ProgressEngine:
    """
    Tracks the dual (physical vs QA-verified) progress of every schedule
    activity, applying matched observations as they arrive.

    Usage
    -----
        with open("data/activities.json") as f:
            activities = json.load(f)
        engine = ProgressEngine(activities)
        state = engine.apply_match_decision(decision, observation)
        # state.physical_progress_pct / state.verified_progress_pct now reflect the update
    """

    def __init__(self, activities: list[dict[str, Any]]) -> None:
        """
        Builds the initial ``ProgressState`` for every activity, all
        starting at 0% progress. ``qa_gate_status`` is seeded from each
        activity's ``requires_qa_gate`` flag: "PENDING_QA" if a gate is
        required, "NOT_REQUIRED" otherwise.
        """
        self._states: dict[str, ProgressState] = {}
        for activity in activities:
            activity_id = activity.get("activity_id")
            if not activity_id:
                log.warning("ProgressEngine: skipping activity record with no activity_id: %r", activity)
                continue
            requires_qa_gate = bool(activity.get("requires_qa_gate"))
            qa_gate_type = activity.get("qa_gate_type") or NO_QA_GATE_SENTINEL
            planned_quantity = activity.get("planned_quantity")
            if not planned_quantity or planned_quantity <= 0:
                log.warning(
                    "ProgressEngine: activity %s has invalid planned_quantity=%r; skipping.",
                    activity_id, planned_quantity,
                )
                continue

            self._states[activity_id] = ProgressState(
                activity_id=activity_id,
                planned_quantity=float(planned_quantity),
                unit=str(activity.get("unit", "")),
                physical_claimed_quantity=0.0,
                physical_progress_pct=0.0,
                verified_earned_quantity=0.0,
                verified_progress_pct=0.0,
                qa_gate_type=qa_gate_type,
                qa_gate_status="PENDING_QA" if requires_qa_gate else "NOT_REQUIRED",
            )

        log.info("ProgressEngine: initialized progress state for %d activities.", len(self._states))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_match_decision(
        self, decision: MatchDecision, observation: NormalizedObservation
    ) -> ProgressState:
        """
        Apply a matched observation's quantity to its selected activity's
        progress, updating either the physical claim or the QA-verified
        earned progress depending on ``observation.is_qa_clearance``.

        Parameters
        ----------
        decision    : a ``MatchDecision`` with a resolved
                      ``selected_activity_id`` (i.e. AUTO_ACCEPT, or a
                      HUMAN_REVIEW decision a human has since confirmed).
                      Callers are responsible for that confirmation step —
                      ``ProgressEngine`` itself only requires a resolved
                      activity_id and does not re-examine ``decision_type``.
        observation : the ``NormalizedObservation`` the decision was made
                      for, supplying the quantity/unit/QA-clearance flag.

        Returns
        -------
        The activity's updated ``ProgressState``.

        Raises
        ------
        ValueError : ``decision.selected_activity_id`` is None (an
                     UNMATCHED decision carries no activity to update — the
                     caller should never route an UNMATCHED decision here),
                     or the activity_id is not tracked by this engine.

        Unit mismatches (observation.normalized_unit != the activity's
        tracked unit) are logged as a warning and the update is skipped —
        returning the activity's unchanged current state — rather than
        silently corrupting the progress figures with an invalid addition.
        """
        if not decision.selected_activity_id:
            raise ValueError(
                f"apply_match_decision: decision '{decision.match_id}' has no selected_activity_id "
                f"(decision_type={decision.decision_type!r}) — nothing to update. "
                f"Only AUTO_ACCEPT or human-confirmed HUMAN_REVIEW decisions should reach this method."
            )

        activity_id = decision.selected_activity_id
        state = self._states.get(activity_id)
        if state is None:
            raise ValueError(f"apply_match_decision: unknown activity_id '{activity_id}' — not tracked by this engine.")

        comparable_quantity = convert_for_comparison(
            observation.normalized_quantity, observation.normalized_unit, state.unit
        )
        if comparable_quantity is None:
            log.warning(
                "apply_match_decision: incompatible units for activity %s (observation unit=%r, activity unit=%r) — "
                "skipping quantity update for observation %s.",
                activity_id, observation.normalized_unit, state.unit, observation.observation_id,
            )
            return state

        if observation.is_qa_clearance:
            updated_state = self._apply_qa_verification(state, comparable_quantity)
        else:
            updated_state = self._apply_physical_claim(state, comparable_quantity)

        self._states[activity_id] = updated_state
        return updated_state

    def get_progress_state(self, activity_id: str) -> Optional[ProgressState]:
        """Read-only lookup of an activity's current progress state, or
        None if the activity_id isn't tracked by this engine."""
        return self._states.get(activity_id)

    def get_all_progress_states(self) -> list[ProgressState]:
        """Snapshot of every tracked activity's current progress state."""
        return list(self._states.values())

    # ------------------------------------------------------------------
    # Internal update logic
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_physical_claim(state: ProgressState, comparable_quantity: float) -> ProgressState:
        """
        A non-QA observation claims new physical execution. The claimed
        quantity accumulates cumulatively across every DPR reporting
        against this activity (each report describes incremental work done
        "today", not a cumulative-to-date total).

        ``comparable_quantity`` is the observation's quantity already
        converted into this activity's own unit (see
        ``unit_normalizer.convert_for_comparison``) — this method never
        looks at the observation's original unit directly.

        The raw ``physical_claimed_quantity`` is intentionally NOT capped at
        ``planned_quantity`` — a cumulative claim exceeding planned scope is
        itself meaningful audit information (potential scope overrun or an
        inflated claim) and should never be silently discarded. Only the
        *percentage* is capped at 100%, exactly per the spec formula.
        """
        new_claimed = state.physical_claimed_quantity + comparable_quantity
        new_pct = min(100.0, (new_claimed / state.planned_quantity) * 100.0)

        return state.model_copy(update={
            "physical_claimed_quantity": round(new_claimed, 4),
            "physical_progress_pct": round(new_pct, 4),
        })

    @staticmethod
    def _apply_qa_verification(state: ProgressState, comparable_quantity: float) -> ProgressState:
        """
        A QA-clearance observation certifies previously-claimed physical
        work as verified — it does NOT introduce new physical claim
        quantity of its own. ``verified_earned_quantity`` accumulates but is
        capped at ``physical_claimed_quantity``: QA cannot verify more work
        than has actually been claimed as physically done, by design — this
        is the core anti-fraud/anti-error guarantee of the dual-tracking
        model.

        ``comparable_quantity`` is the observation's quantity already
        converted into this activity's own unit.
        """
        proposed_verified = state.verified_earned_quantity + comparable_quantity
        new_verified = min(state.physical_claimed_quantity, proposed_verified)

        if proposed_verified > state.physical_claimed_quantity:
            log.warning(
                "Activity %s: QA verification claim (%.4f) exceeds physical claimed quantity (%.4f) — "
                "capping verified_earned_quantity at the physical claim. This may indicate the QA report "
                "arrived before the corresponding physical DPR, or a data quality issue worth reviewing.",
                state.activity_id, proposed_verified, state.physical_claimed_quantity,
            )

        new_pct = min(100.0, (new_verified / state.planned_quantity) * 100.0)

        return state.model_copy(update={
            "verified_earned_quantity": round(new_verified, 4),
            "verified_progress_pct": round(new_pct, 4),
            "qa_gate_status": "VERIFIED_PASSED",
        })
