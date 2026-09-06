"""
scoring_engine.py
====================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 4 DELIVERABLE — Hybrid Scoring Formula
---------------------------------------------------
``ScoringEngine`` combines three independent signals into one final
confidence score per candidate activity:

    S_final = w_entity * S_entity + w_semantic * S_semantic + w_quantity * S_quantity

    w_entity   = 0.50   — how well Stage 1's extracted entities
                          (KP, facility, discipline, action) match this
                          specific candidate.
    w_semantic = 0.35   — Sentence-BERT (or fallback) semantic similarity
                          between the DPR's raw text and the candidate's
                          activity name.
    w_quantity = 0.15   — whether the claimed quantity is physically
                          plausible against the candidate's planned scope.

Entity scoring is deliberately re-derived *per candidate* here (rather than
reusing Stage 1's single narrowing_stage label for the whole shortlist),
because Stage 1's job was only to shrink 500+ activities down to 3-5 — it
does not rank *within* that shortlist. ``ScoringEngine`` is what actually
discriminates between the surviving candidates.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from candidate_narrower import ACTION_TO_ACTIVITY_KEYWORDS, CandidateNarrower
from schemas import CandidateShortlist, ExtractedEntities, NormalizedObservation
from unit_normalizer import convert_for_comparison
from vector_ranker import VectorRanker

log = logging.getLogger("planbridge.scoring_engine")

# Default hybrid formula weights, per the PS26122 Stage 2 spec.
DEFAULT_W_ENTITY = 0.50
DEFAULT_W_SEMANTIC = 0.35
DEFAULT_W_QUANTITY = 0.15

# Sub-weights within the entity-score dimension itself. Only the entity
# fields that were actually EXTRACTED for a given observation contribute to
# both the numerator and denominator of the normalized entity_score — a
# field that wasn't extracted is excluded entirely rather than counted as a
# failed match, since "no signal" and "signal that disagrees" are different
# things and shouldn't be scored the same way.
KP_SUBWEIGHT = 0.40
FACILITY_SUBWEIGHT = 0.25
DISCIPLINE_SUBWEIGHT = 0.20
ACTION_SUBWEIGHT = 0.15

# A KP match within this many kilometres of the extracted chainage still
# earns partial credit, decaying linearly to 0 at the window edge. Exact
# matches (distance 0) always score 1.0.
KP_PROXIMITY_WINDOW_KM = 5.0


class ScoringEngine:
    """
    Computes the weighted hybrid confidence score for each candidate in a
    Stage 1 shortlist.

    Usage
    -----
        engine = ScoringEngine()
        ranked = engine.evaluate_candidates(observation, shortlist)
        # ranked[0] is the best-scoring candidate (final_score descending)
    """

    def __init__(
        self,
        w_entity: float = DEFAULT_W_ENTITY,
        w_semantic: float = DEFAULT_W_SEMANTIC,
        w_quantity: float = DEFAULT_W_QUANTITY,
        vector_ranker: Optional[VectorRanker] = None,
    ) -> None:
        """
        Parameters
        ----------
        w_entity, w_semantic, w_quantity : hybrid formula weights. Must sum
            to (approximately) 1.0 — a mismatch is logged as a warning but
            does not raise, since a caller may deliberately want to
            over/under-weight a dimension for experimentation.
        vector_ranker : an existing ``VectorRanker`` instance to reuse
            (recommended — model loading is expensive). A new default
            instance is created if not provided.
        """
        total_weight = w_entity + w_semantic + w_quantity
        if abs(total_weight - 1.0) > 1e-6:
            log.warning(
                "ScoringEngine: weights sum to %.4f, not 1.0 (w_entity=%.2f, w_semantic=%.2f, w_quantity=%.2f).",
                total_weight, w_entity, w_semantic, w_quantity,
            )
        self.w_entity = w_entity
        self.w_semantic = w_semantic
        self.w_quantity = w_quantity
        self.vector_ranker = vector_ranker or VectorRanker()

    # ------------------------------------------------------------------
    # Quantity plausibility scoring
    # ------------------------------------------------------------------
    def compute_quantity_score(self, claimed_qty: float, target_qty: float) -> float:
        """
        Score how plausible a claimed quantity is against a candidate
        activity's planned quantity.

        Rules
        -----
        * claimed_qty <= 0                -> 0.0  (no real claim / invalid data)
        * target_qty is missing or <= 0   -> 0.5  (can't evaluate; neutral,
                                                     not penalized — this is
                                                     a data-quality gap, not
                                                     evidence against the
                                                     candidate)
        * 0 < claimed_qty <= target_qty   -> 1.0  (within planned scope)
        * claimed_qty > target_qty        -> linearly penalized, reaching
                                              0.0 once the claim is double
                                              the planned quantity or more.
                                              e.g. 10% over -> 0.90,
                                                   50% over -> 0.50,
                                                   100%+ over -> 0.00
        """
        if claimed_qty is None or claimed_qty <= 0:
            return 0.0
        if target_qty is None or target_qty <= 0:
            log.warning(
                "compute_quantity_score: invalid target_qty=%r for claimed_qty=%r; returning neutral score.",
                target_qty, claimed_qty,
            )
            return 0.5
        if claimed_qty <= target_qty:
            return 1.0

        overage_fraction = (claimed_qty - target_qty) / target_qty
        return round(max(0.0, 1.0 - overage_fraction), 4)

    # ------------------------------------------------------------------
    # Entity scoring (per-candidate, within the Stage 1 shortlist)
    # ------------------------------------------------------------------
    def _compute_entity_score(self, activity: dict[str, Any], entities: ExtractedEntities) -> float:
        """
        Score how well one candidate activity matches the entities
        extracted from the observation's text, normalized over only the
        entity dimensions that were actually extracted.

        Returns 0.0 if no entities were extracted at all (no basis for a
        judgement — this mirrors Stage 1 falling to FALLBACK_GENERIC).
        """
        weighted_sum = 0.0
        weight_total = 0.0

        if entities.location_kp:
            weight_total += KP_SUBWEIGHT
            weighted_sum += KP_SUBWEIGHT * self._kp_match_score(entities.location_kp, activity.get("location_kp"))

        if entities.facility:
            weight_total += FACILITY_SUBWEIGHT
            act_facility = str(activity.get("facility", "")).lower()
            match = CandidateNarrower._fuzzy_substring_match(entities.facility.lower(), act_facility)
            weighted_sum += FACILITY_SUBWEIGHT * (1.0 if match else 0.0)

        if entities.discipline and entities.action_verb != "QA Inspection":
            # QA Inspection is intentionally discipline-agnostic here — see
            # the matching comment in candidate_narrower.py's
            # _filter_by_action_verb: real QA-gated activities are Piping/
            # Civil/Mechanical, never "HSE", so treating discipline=="HSE"
            # as a hard signal against otherwise-correct candidates would
            # be penalizing them for a label mismatch that isn't actually
            # evidence of a wrong match.
            weight_total += DISCIPLINE_SUBWEIGHT
            match = str(activity.get("discipline", "")).lower() == entities.discipline.lower()
            weighted_sum += DISCIPLINE_SUBWEIGHT * (1.0 if match else 0.0)

        if entities.action_verb:
            weight_total += ACTION_SUBWEIGHT
            weighted_sum += ACTION_SUBWEIGHT * self._action_match_score(entities.action_verb, activity)

        if weight_total == 0.0:
            return 0.0
        return round(weighted_sum / weight_total, 4)

    @staticmethod
    def _kp_match_score(extracted_kp: str, activity_kp: Optional[str]) -> float:
        """1.0 for an exact KP match, decaying linearly to 0.0 at
        KP_PROXIMITY_WINDOW_KM away — rewards "close but not exact" KP
        mentions (common when a field report rounds chainage) rather than
        scoring every non-exact KP identically at 0."""
        if not activity_kp:
            return 0.0
        extracted_value = CandidateNarrower._kp_to_float(extracted_kp)
        activity_value = CandidateNarrower._kp_to_float(activity_kp)
        if extracted_value is None or activity_value is None:
            return 0.0
        distance = abs(extracted_value - activity_value)
        if distance == 0:
            return 1.0
        return round(max(0.0, 1.0 - (distance / KP_PROXIMITY_WINDOW_KM)), 4)

    @staticmethod
    def _action_match_score(action_verb: str, activity: dict[str, Any]) -> float:
        """1.0 if the candidate's activity_name contains a keyword
        associated with the extracted action_verb (same mapping Stage 1
        uses for its ACTION_VERB filter, imported for consistency), with a
        special case for QA Inspection also matching on requires_qa_gate."""
        keywords = ACTION_TO_ACTIVITY_KEYWORDS.get(action_verb, ())
        name = str(activity.get("activity_name", ""))
        matched = any(re.search(rf"\b{re.escape(kw)}", name, re.IGNORECASE) for kw in keywords)
        if action_verb == "QA Inspection":
            matched = matched or bool(activity.get("requires_qa_gate"))
        return 1.0 if matched else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_candidates(
        self, observation: NormalizedObservation, shortlist: CandidateShortlist
    ) -> list[dict[str, Any]]:
        """
        Score every candidate in a Stage 1 shortlist using the hybrid
        formula, returning a list of score-breakdown dicts sorted by
        ``final_score`` descending.

        Returns an empty list (never raises) if the shortlist has no
        candidates — a legitimate outcome for noise/unmatched observations.

        Each returned dict has the shape:
            {
                "activity_id": str,
                "activity_name": str,
                "entity_score": float,
                "semantic_score": float,
                "quantity_score": float,
                "unit_match": bool,
                "final_score": float,
                "activity": dict,   # the full source activity record
            }
        """
        candidates = shortlist.candidate_activities
        if not candidates:
            log.info(
                "evaluate_candidates: empty shortlist for observation %s; nothing to score.",
                observation.observation_id,
            )
            return []

        entities = shortlist.extracted_entities
        candidate_names = [str(act.get("activity_name", "")) for act in candidates]

        try:
            semantic_scores = self.vector_ranker.calculate_semantic_similarity(
                observation.raw_phrase, candidate_names
            )
        except Exception as exc:  # defensive — a ranker failure shouldn't crash scoring
            log.error("evaluate_candidates: semantic similarity computation failed (%s); using 0.0 for all.", exc)
            semantic_scores = [0.0] * len(candidates)

        results: list[dict[str, Any]] = []
        for activity, semantic_score in zip(candidates, semantic_scores):
            entity_score = self._compute_entity_score(activity, entities)

            activity_unit = activity.get("unit")
            comparable_claimed_qty = convert_for_comparison(
                observation.normalized_quantity, observation.normalized_unit, activity_unit
            )
            unit_match = comparable_claimed_qty is not None
            if unit_match:
                quantity_score = self.compute_quantity_score(
                    comparable_claimed_qty, activity.get("planned_quantity")
                )
            else:
                # Genuinely incompatible physical dimensions (e.g. a KM
                # claim against a JOINTS-measured activity) — comparing
                # them is meaningless and is itself a signal this
                # candidate is likely the wrong one.
                quantity_score = 0.0

            final_score = round(
                self.w_entity * entity_score
                + self.w_semantic * semantic_score
                + self.w_quantity * quantity_score,
                4,
            )

            results.append({
                "activity_id": activity.get("activity_id"),
                "activity_name": activity.get("activity_name"),
                "entity_score": entity_score,
                "semantic_score": round(semantic_score, 4),
                "quantity_score": quantity_score,
                "unit_match": unit_match,
                "final_score": final_score,
                "activity": activity,
            })

        results.sort(key=lambda r: r["final_score"], reverse=True)
        return results
