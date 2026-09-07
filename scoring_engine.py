"""
scoring_engine.py
PlanBridge — Stage 2: Hybrid Scoring Engine
"""
from __future__ import annotations
import logging
import re
from typing import Any, Optional
from schemas import CandidateShortlist, ExtractedEntities, NormalizedObservation
from vector_ranker import VectorRanker

log = logging.getLogger("planbridge.scoring_engine")

DEFAULT_WEIGHT_ENTITY = 0.55
DEFAULT_WEIGHT_SEMANTIC = 0.30
DEFAULT_WEIGHT_QUANTITY = 0.15

UNIT_CONVERSION_FACTORS: dict[str, dict[str, float]] = {
    "M": {"KM": 0.001, "M": 1.0},
    "KM": {"M": 1000.0, "KM": 1.0},
    "KG": {"TONNES": 0.001, "KG": 1.0},
    "TONNES": {"KG": 1000.0, "TONNES": 1.0},
    "JOINTS": {"JOINTS": 1.0},
    "SPOOLS": {"SPOOLS": 1.0},
    "PIT": {"PIT": 1.0},
}

class ScoringEngine:
    def __init__(
        self,
        vector_ranker: Optional[VectorRanker] = None,
        w_entity: float = DEFAULT_WEIGHT_ENTITY,
        w_semantic: float = DEFAULT_WEIGHT_SEMANTIC,
        w_quantity: float = DEFAULT_WEIGHT_QUANTITY,
    ) -> None:
        self.vector_ranker = vector_ranker or VectorRanker()
        self.w_entity = w_entity
        self.w_semantic = w_semantic
        self.w_quantity = w_quantity

    def evaluate_candidates(
        self, observation: NormalizedObservation, shortlist: CandidateShortlist
    ) -> list[dict[str, Any]]:
        candidates = shortlist.candidate_activities
        if not candidates:
            return []

        entities = shortlist.extracted_entities
        clean_text = self._clean_query_text(observation.raw_phrase)
        candidate_names = [str(c.get("activity_name", "")) for c in candidates]

        semantic_scores = self.vector_ranker.calculate_semantic_similarity(
            clean_text, candidate_names
        )

        results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            activity_id = str(candidate.get("activity_id", ""))
            activity_name = str(candidate.get("activity_name", ""))
            planned_qty = float(candidate.get("planned_quantity", 0.0))
            planned_unit = str(candidate.get("unit", "")).upper()

            entity_score = self._compute_entity_score(candidate, entities)
            semantic_score = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0

            quantity_score, unit_match = self._compute_quantity_score(
                observation.normalized_quantity,
                observation.normalized_unit,
                planned_qty,
                planned_unit,
            )

            final_score = (
                (self.w_entity * entity_score)
                + (self.w_semantic * semantic_score)
                + (self.w_quantity * quantity_score)
            )
            final_score = round(min(1.0, max(0.0, final_score)), 4)

            results.append({
                "activity_id": activity_id,
                "activity_name": activity_name,
                "entity_score": round(entity_score, 4),
                "semantic_score": round(semantic_score, 4),
                "quantity_score": round(quantity_score, 4),
                "unit_match": unit_match,
                "final_score": final_score,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def _compute_entity_score(
        self, candidate: dict[str, Any], entities: ExtractedEntities
    ) -> float:
        extracted_act_id = getattr(entities, "activity_id", None)
        if extracted_act_id and str(candidate.get("activity_id", "")).upper() == str(extracted_act_id).upper():
            return 1.0

        score = 0.0
        total_weight = 0.0

        # 1. Location KP
        if entities.location_kp:
            total_weight += 0.40
            cand_kp = str(candidate.get("location_kp", ""))
            if self._normalize_kp(entities.location_kp) == self._normalize_kp(cand_kp):
                score += 0.40
            else:
                ev_val = self._kp_to_float(entities.location_kp)
                cand_val = self._kp_to_float(cand_kp)
                if ev_val is not None and cand_val is not None:
                    dist = abs(ev_val - cand_val)
                    if dist <= 5.0:
                        score += 0.40 * max(0.0, 1.0 - (dist / 5.0))

        # 2. Facility
        if entities.facility:
            total_weight += 0.25
            cand_facility = str(candidate.get("facility", "")).lower()
            ef_lower = entities.facility.lower()
            if ef_lower in cand_facility or cand_facility in ef_lower:
                score += 0.25

        # 3. Discipline
        if entities.discipline:
            total_weight += 0.20
            cand_disc = str(candidate.get("discipline", "")).lower()
            if entities.discipline.lower() == cand_disc:
                score += 0.20

        # 4. Action Verb — Smart Multi-Word & Substring Matching
        if entities.action_verb:
            total_weight += 0.15
            cand_name = str(candidate.get("activity_name", "")).lower()
            act_lower = entities.action_verb.lower()

            if act_lower in cand_name:
                score += 0.15
            else:
                action_words = [w for w in re.split(r"[\s\-_/]+", act_lower) if len(w) > 2]
                if any(w in cand_name for w in action_words):
                    score += 0.15
                elif entities.action_verb == "QA Inspection" and candidate.get("requires_qa_gate"):
                    score += 0.15

        return min(1.0, score / total_weight) if total_weight > 0.0 else 0.0

    def _compute_quantity_score(
        self,
        claimed_qty: float,
        claimed_unit: str,
        planned_qty: float,
        planned_unit: str,
    ) -> tuple[float, bool]:
        if claimed_qty <= 0 or planned_qty <= 0:
            return 0.5, True

        c_unit = claimed_unit.upper().strip()
        p_unit = planned_unit.upper().strip()

        factor_map = UNIT_CONVERSION_FACTORS.get(c_unit, {})
        factor = factor_map.get(p_unit)

        if factor is None:
            return 0.0, False

        comparable_qty = claimed_qty * factor
        if comparable_qty <= planned_qty:
            return 1.0, True

        overage = (comparable_qty - planned_qty) / planned_qty
        score = max(0.0, 1.0 - overage)
        return round(score, 4), True

    @staticmethod
    def _clean_query_text(text: str) -> str:
        t = re.sub(r'["\'`]', '', text)
        t = re.sub(r'^(today\'s update\s*[-—:]*|progress\s*[-—:]*)', '', t, flags=re.IGNORECASE)
        return t.strip()

    @staticmethod
    def _normalize_kp(kp: Optional[str]) -> str:
        if not kp:
            return ""
        digits = re.search(r"\d+(?:\+\d+)?", kp)
        return digits.group(0) if digits else kp.strip().lower()

    @staticmethod
    def _kp_to_float(kp: Optional[str]) -> Optional[float]:
        if not kp:
            return None
        match = re.search(r"(\d+)(?:\+(\d+))?", kp)
        if not match:
            return None
        km = float(match.group(1))
        m = float(match.group(2)) if match.group(2) else 0.0
        return km + (m / 1000.0)
