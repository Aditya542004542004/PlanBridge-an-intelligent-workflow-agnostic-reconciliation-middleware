"""
candidate_narrower.py
========================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 3 DELIVERABLE — Stage 1: Candidate Narrowing
---------------------------------------------------------
``CandidateNarrower`` takes the entities ``EntityExtractor`` pulled out of a
DPR phrase and uses them to cut OIL's 500+ Primavera P6 activities down to a
short, human-reviewable list of 3-5 candidates — purely through hard,
deterministic entity filtering. No embeddings, no semantic similarity: that
is Phase 4's job. Stage 1's entire purpose is to shrink the search space
*before* the (much more expensive) semantic re-ranking pass ever runs.

Filter cascade (per the spec, tried in order until >=3 candidates survive):

    0. ACTIVITY_ID_EXACT  — exact match on explicit task code (e.g. "PIP-L5-044-036").
    1. KP_EXACT           — exact match on ``location_kp``.
    2. FACILITY_DISCIPLINE — facility name (substring, case-insensitive)
                              AND/OR discipline match.
    3. ACTION_VERB         — activity name contains keywords associated
                              with the extracted action, optionally
                              narrowed further by discipline.
    4. FALLBACK_GENERIC    — no usable entity signal at all; return a
                              generic top-N slice so the reviewer always
                              has *something* to look at rather than a
                              silent empty shortlist.

Whichever stage first produces >=3 matches is used as the final shortlist
source; if a stage produces more than 5, it is ranked down to the best 5
(by KP numeric proximity when possible, else by stable input order) before
being returned.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from schemas import CandidateShortlist, ExtractedEntities, NormalizedObservation

log = logging.getLogger("planbridge.candidate_narrower")

MIN_SHORTLIST_SIZE = 3
MAX_SHORTLIST_SIZE = 5

# Maps an extracted action_verb back to the keywords we'd expect to find in
# a matching activity_name — the inverse of EntityExtractor's action rules,
# used for the ACTION_VERB fallback filter stage.
ACTION_TO_ACTIVITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "HDD Drilling": ("HDD",),
    "Tie-in Welding": ("Tie-in", "Welding", "Spool"),
    "Excavation": ("Excavation", "Trenching", "Valve Pit", "Backfilling", "ROW Clearing", "Access Road"),
    "QA Inspection": ("Hydrotest", "NDT", "Radiography"),  # supplemented by requires_qa_gate below
}


class CandidateNarrower:
    """
    Narrows the full activity list down to a 3-5 item candidate shortlist
    for a given observation, using ``ExtractedEntities`` as hard filters.

    Usage
    -----
        with open("data/activities.json") as f:
            activities = json.load(f)
        narrower = CandidateNarrower(activities)
        shortlist = narrower.narrow_candidates(observation, entities)
    """

    def __init__(self, activities: list[dict[str, Any]]) -> None:
        if not activities:
            log.warning("CandidateNarrower initialized with an empty activities list.")
        self.activities: list[dict[str, Any]] = activities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def narrow_candidates(
        self, observation: NormalizedObservation, entities: ExtractedEntities
    ) -> CandidateShortlist:
        """
        Produce a ``CandidateShortlist`` for one observation.

        Never raises solely because entities are sparse or absent — that is
        the expected shape for noise/administrative reports, and the
        FALLBACK_GENERIC stage guarantees a well-formed (if weak) shortlist
        is always returned so the pipeline never breaks on missing signal.
        """
        if not self.activities:
            return CandidateShortlist(
                observation_id=observation.observation_id,
                extracted_entities=entities,
                candidate_activities=[],
                initial_candidate_count=0,
                narrowing_stage=None,
            )

        # -- Stage 0: exact Activity ID match (e.g. "PIP-L5-044-036") ---
        extracted_act_id = getattr(entities, "activity_id", None)
        if extracted_act_id:
            matches = [
                act for act in self.activities
                if str(act.get("activity_id", "")).strip().upper() == str(extracted_act_id).strip().upper()
            ]
            if matches:
                # Pad to minimum shortlist size if needed so 3-5 candidate contract holds
                if len(matches) < MIN_SHORTLIST_SIZE:
                    disc_hint = matches[0].get("discipline") if matches else entities.discipline
                    matches = self._pad_to_minimum(matches, entities.location_kp, disc_hint)
                return self._build_shortlist(observation, entities, matches, "ACTIVITY_ID_EXACT", entities.location_kp)

        # -- Stage 1: exact KP match -----------------------------------
        if entities.location_kp:
            matches = self._filter_by_kp(entities.location_kp)
            if len(matches) >= MIN_SHORTLIST_SIZE:
                return self._build_shortlist(observation, entities, matches, "KP_EXACT")

        # -- Stage 2: facility + discipline ------------------------------
        if entities.facility:
            matches = self._filter_by_facility_and_discipline(entities.facility, entities.discipline)
            if len(matches) >= MIN_SHORTLIST_SIZE:
                return self._build_shortlist(observation, entities, matches, "FACILITY_DISCIPLINE", entities.location_kp)

        # -- Stage 3: action verb (optionally + discipline) --------------
        if entities.action_verb or entities.discipline:
            matches = self._filter_by_action_verb(entities.action_verb, entities.discipline)
            if len(matches) >= MIN_SHORTLIST_SIZE:
                return self._build_shortlist(observation, entities, matches, "ACTION_VERB", entities.location_kp)

        # -- Stage 4: generic fallback — always produces a result --------
        best_partial = self._best_partial_match(entities)
        if best_partial:
            matches, stage = best_partial
        else:
            matches, stage = self.activities, "FALLBACK_GENERIC"

        if len(matches) < MIN_SHORTLIST_SIZE:
            discipline_hint = matches[0].get("discipline") if matches else entities.discipline
            matches = self._pad_to_minimum(matches, entities.location_kp, discipline_hint)

        return self._build_shortlist(observation, entities, matches, stage, entities.location_kp)

    # ------------------------------------------------------------------
    # Filter stage implementations
    # ------------------------------------------------------------------
    def _filter_by_kp(self, location_kp: str) -> list[dict[str, Any]]:
        target = self._normalize_kp_for_compare(location_kp)
        return [
            act for act in self.activities
            if self._normalize_kp_for_compare(act.get("location_kp", "")) == target
        ]

    def _filter_by_facility_and_discipline(
        self, facility: Optional[str], discipline: Optional[str]
    ) -> list[dict[str, Any]]:
        if not facility:
            return []
        results = []
        facility_norm = facility.lower()
        for act in self.activities:
            act_facility = str(act.get("facility", "")).lower()
            if not self._fuzzy_substring_match(facility_norm, act_facility):
                continue
            if discipline and str(act.get("discipline", "")).lower() != discipline.lower():
                continue
            results.append(act)
        return results

    def _filter_by_action_verb(
        self, action_verb: Optional[str], discipline: Optional[str]
    ) -> list[dict[str, Any]]:
        results = []
        keywords = ACTION_TO_ACTIVITY_KEYWORDS.get(action_verb, ()) if action_verb else ()
        for act in self.activities:
            name = str(act.get("activity_name", ""))
            action_ok = True
            if keywords:
                action_ok = any(re.search(rf"\b{re.escape(kw)}", name, re.IGNORECASE) for kw in keywords)
                if action_verb == "QA Inspection":
                    action_ok = action_ok or bool(act.get("requires_qa_gate"))
            discipline_ok = True
            if discipline:
                discipline_ok = str(act.get("discipline", "")).lower() == discipline.lower()
            if action_ok and discipline_ok and (keywords or discipline):
                results.append(act)
        return results

    def _best_partial_match(
        self, entities: ExtractedEntities
    ) -> Optional[tuple[list[dict[str, Any]], str]]:
        candidates: list[tuple[list[dict[str, Any]], str]] = []

        extracted_act_id = getattr(entities, "activity_id", None)
        if extracted_act_id:
            act_matches = [
                act for act in self.activities
                if str(act.get("activity_id", "")).strip().upper() == str(extracted_act_id).strip().upper()
            ]
            if act_matches:
                candidates.append((act_matches, "ACTIVITY_ID_EXACT"))

        if entities.location_kp:
            kp_matches = self._filter_by_kp(entities.location_kp)
            if kp_matches:
                candidates.append((kp_matches, "KP_EXACT"))

        if entities.facility:
            fd_matches = self._filter_by_facility_and_discipline(entities.facility, entities.discipline)
            if fd_matches:
                candidates.append((fd_matches, "FACILITY_DISCIPLINE"))

        if entities.action_verb or entities.discipline:
            av_matches = self._filter_by_action_verb(entities.action_verb, entities.discipline)
            if av_matches:
                candidates.append((av_matches, "ACTION_VERB"))

        if not candidates:
            return None
        return max(candidates, key=lambda pair: len(pair[0]))

    def _pad_to_minimum(
        self,
        matches: list[dict[str, Any]],
        reference_kp: Optional[str],
        discipline_hint: Optional[str],
    ) -> list[dict[str, Any]]:
        if len(matches) >= MIN_SHORTLIST_SIZE:
            return matches

        existing_ids = {act.get("activity_id") for act in matches}
        pool = [act for act in self.activities if act.get("activity_id") not in existing_ids]

        ref_value = self._kp_to_float(reference_kp) if reference_kp else None
        if ref_value is not None:
            def sort_key(act: dict[str, Any]) -> tuple[bool, float]:
                act_value = self._kp_to_float(act.get("location_kp"))
                return (act_value is None, abs(act_value - ref_value) if act_value is not None else float("inf"))

            pool.sort(key=sort_key)
        elif discipline_hint:
            pool.sort(key=lambda act: str(act.get("discipline", "")).lower() != discipline_hint.lower())

        needed = MIN_SHORTLIST_SIZE - len(matches)
        return matches + pool[:needed]

    # ------------------------------------------------------------------
    # Shortlist assembly & ranking
    # ------------------------------------------------------------------
    def _build_shortlist(
        self,
        observation: NormalizedObservation,
        entities: ExtractedEntities,
        matches: list[dict[str, Any]],
        stage: str,
        reference_kp: Optional[str] = None,
    ) -> CandidateShortlist:
        initial_count = len(matches)
        top_matches = self._rank_and_truncate(matches, reference_kp)
        return CandidateShortlist(
            observation_id=observation.observation_id,
            extracted_entities=entities,
            candidate_activities=top_matches,
            initial_candidate_count=initial_count,
            narrowing_stage=stage,
        )

    def _rank_and_truncate(
        self, matches: list[dict[str, Any]], reference_kp: Optional[str]
    ) -> list[dict[str, Any]]:
        ref_value = self._kp_to_float(reference_kp) if reference_kp else None
        if ref_value is not None and len(matches) > 1:
            def distance(act: dict[str, Any]) -> float:
                act_value = self._kp_to_float(act.get("location_kp"))
                return abs(act_value - ref_value) if act_value is not None else float("inf")

            matches = sorted(matches, key=distance)

        return matches[:MAX_SHORTLIST_SIZE]

    # ------------------------------------------------------------------
    # KP comparison helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_kp_for_compare(kp: Optional[str]) -> str:
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
        km_part = float(match.group(1))
        m_part = float(match.group(2)) if match.group(2) else 0.0
        return km_part + (m_part / 1000.0)

    @staticmethod
    def _fuzzy_substring_match(a: str, b: str) -> bool:
        if not a or not b:
            return False
        return a in b or b in a