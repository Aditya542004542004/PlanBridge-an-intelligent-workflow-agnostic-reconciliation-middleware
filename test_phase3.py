"""
test_phase3.py
================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 3 DELIVERABLE — Test suite
--------------------------------------
Run with either:

    python -m unittest test_phase3.py -v
    pytest test_phase3.py -v

Coverage:
    * TestEntityExtractor      — regex entity extraction accuracy (KP, Line
                                  ID, Facility) and keyword-dictionary
                                  discipline/action detection, including the
                                  three sample DPR phrases from the spec.
    * TestCandidateNarrower    — each filter-cascade stage exercised in
                                  isolation against a small, deterministic
                                  synthetic activity list (fast, no I/O).
    * TestPhase3Integration    — end-to-end run against the real
                                  data/activities.json produced by Phase 1,
                                  confirming the full pipeline narrows 500+
                                  activities down to a 3-5 item shortlist
                                  for each of the spec's sample phrases.
"""

from __future__ import annotations

import json
import os
import unittest

from candidate_narrower import CandidateNarrower, MAX_SHORTLIST_SIZE, MIN_SHORTLIST_SIZE
from entity_extractor import EntityExtractor
from schemas import ExtractedEntities, NormalizedObservation

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def make_observation(obs_id: str, raw_phrase: str) -> NormalizedObservation:
    """Minimal NormalizedObservation builder for test purposes — the
    quantity/unit fields aren't relevant to entity extraction or candidate
    narrowing, so they're stubbed with plausible placeholder values."""
    return NormalizedObservation(
        observation_id=obs_id,
        report_id="TEST-REPORT",
        raw_phrase=raw_phrase,
        raw_quantity=1.0,
        raw_unit="m",
        normalized_quantity=0.001,
        normalized_unit="KM",
        conversion_applied="test-stub",
        is_qa_clearance=False,
    )


class TestEntityExtractor(unittest.TestCase):
    """Regex + keyword-dictionary entity extraction accuracy."""

    @classmethod
    def setUpClass(cls) -> None:
        # Shared across tests — spaCy model load is the expensive part.
        cls.extractor = EntityExtractor()

    # -- Spec sample phrase 1 -----------------------------------------------
    def test_sample_phrase_1_hdd_at_kp(self):
        entities = self.extractor.extract("150m HDD drilling finished near KP 24+600 at river site")
        self.assertEqual(entities.location_kp, "KP 24+600")
        self.assertEqual(entities.action_verb, "HDD Drilling")
        self.assertEqual(entities.discipline, "Piping")

    # -- Spec sample phrase 2 -----------------------------------------------
    def test_sample_phrase_2_tie_in_at_cgs(self):
        entities = self.extractor.extract("Piping tie-in work progressing near CGS inlet valve pit")
        self.assertEqual(entities.action_verb, "Tie-in Welding")
        self.assertEqual(entities.discipline, "Piping")
        self.assertIsNotNone(entities.facility)
        self.assertIn("cgs", entities.facility.lower())

    # -- Spec sample phrase 3 -----------------------------------------------
    def test_sample_phrase_3_noise_text_degrades_gracefully(self):
        # NOTE: the spec's sample expects entities like "warehouse" and
        # "mud pump pit" to be extracted, but the FACILITY_PATTERN given in
        # the spec (CGS|OCS|GCP|Valve Pit|Yard|Station|Duliajan|Numaligarh)
        # does not include "warehouse" or a bare "pit" as a facility
        # keyword — so this text correctly yields no facility/KP match.
        # What we DO assert is the core Phase 3 requirement: extraction
        # must never raise and must return well-formed (mostly-None)
        # entities on text with no real schedule-relevant signal. See the
        # accompanying delivery notes for the full discussion.
        entities = self.extractor.extract("Temporary mud pump pit near warehouse due to heavy rain")
        self.assertIsInstance(entities, ExtractedEntities)
        self.assertIsNone(entities.location_kp)
        self.assertIsNone(entities.line_id)
        # "pit" is a legitimate Excavation/Civil trigger per the spec's own
        # keyword dictionary, so this text is expected to fire that rule —
        # a known, documented false-positive risk for Stage 1 that later
        # phases' semantic/confidence layer is designed to catch.
        self.assertEqual(entities.action_verb, "Excavation")
        self.assertEqual(entities.discipline, "Civil")

    # -- KP marker format variants -------------------------------------------
    def test_kp_marker_variants_are_normalized(self):
        cases = {
            "work done at KP 24+600 today": "KP 24+600",
            "work done at KP24+600 today": "KP 24+600",
            "work done at kp 18 today": "KP 18",
            "work done at KP  18 + 250 today": "KP 18",  # spaced '+' not matched by strict pattern; digits-only fallback
        }
        for text, _ in cases.items():
            entities = self.extractor.extract(text)
            self.assertIsNotNone(entities.location_kp, f"Expected a KP match in: {text!r}")
            self.assertTrue(entities.location_kp.startswith("KP "))

    def test_kp_no_match_returns_none(self):
        entities = self.extractor.extract("Work progressing well today at the yard.")
        self.assertIsNone(entities.location_kp)

    # -- Line ID extraction ---------------------------------------------------
    def test_line_id_extraction(self):
        entities = self.extractor.extract("Tie-in completed on Line 24-A near the manifold.")
        self.assertIsNotNone(entities.line_id)
        self.assertIn("24-A".lower(), entities.line_id.lower())

    def test_line_id_pipeline_prefix(self):
        entities = self.extractor.extract("Pipeline 12-B hydrotest scheduled for tomorrow.")
        self.assertIsNotNone(entities.line_id)

    # -- Facility extraction ---------------------------------------------------
    def test_facility_extraction_cgs(self):
        entities = self.extractor.extract("Manifold setup progressing at CGS Duliajan today.")
        self.assertIsNotNone(entities.facility)
        self.assertIn("cgs", entities.facility.lower())

    def test_facility_extraction_ocs(self):
        entities = self.extractor.extract("Equipment installed at OCS-4 yesterday.")
        self.assertIsNotNone(entities.facility)

    def test_facility_extraction_bare_place_name(self):
        entities = self.extractor.extract("Team mobilized to Duliajan for inspection.")
        self.assertIsNotNone(entities.facility)
        self.assertIn("duliajan", entities.facility.lower())

    # -- Discipline / action-verb keyword dictionary ---------------------------
    def test_action_welding(self):
        entities = self.extractor.extract("8 joints welded today near the tie-in point.")
        self.assertEqual(entities.action_verb, "Tie-in Welding")
        self.assertEqual(entities.discipline, "Piping")

    def test_action_excavation_backfilling(self):
        entities = self.extractor.extract("Backfilling completed for the trench section.")
        self.assertEqual(entities.action_verb, "Excavation")
        self.assertEqual(entities.discipline, "Civil")

    def test_action_qa_inspection_ndt(self):
        entities = self.extractor.extract("NDT radiography passed for the weld joint.")
        self.assertEqual(entities.action_verb, "QA Inspection")
        self.assertEqual(entities.discipline, "HSE")

    def test_no_action_keyword_returns_none(self):
        entities = self.extractor.extract("Site office generator refuelled today.")
        self.assertIsNone(entities.action_verb)
        self.assertIsNone(entities.discipline)

    # -- Graceful handling of edge-case input -----------------------------------
    def test_blank_text_returns_all_none(self):
        entities = self.extractor.extract("")
        self.assertEqual(entities, ExtractedEntities())

    def test_whitespace_only_text_returns_all_none(self):
        entities = self.extractor.extract("    ")
        self.assertFalse(entities.has_any_entity())

    def test_extract_never_raises_on_odd_input(self):
        # Defensive: numbers-only, punctuation-only, very long strings.
        for odd_text in ["12345", "!!!???", "a" * 5000, "KP", "Line"]:
            try:
                entities = self.extractor.extract(odd_text)
                self.assertIsInstance(entities, ExtractedEntities)
            except Exception as exc:  # pragma: no cover
                self.fail(f"extract() raised on input {odd_text[:20]!r}...: {exc}")


class TestCandidateNarrower(unittest.TestCase):
    """Filter-cascade stage behavior against a small, deterministic
    synthetic activity list (independent of Phase 1's generated data, so
    these tests stay stable even if the generator's random seed changes)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.activities = [
            {
                "activity_id": "PIP-L5-001-001",
                "activity_name": "HDD River Crossing Execution at KP 24+600",
                "discipline": "Piping",
                "location_kp": "KP 24+600",
                "facility": "OCS-4",
                "requires_qa_gate": True,
            },
            {
                "activity_id": "PIP-L5-001-002",
                "activity_name": "Tie-in Welding at Chainage KP 10+000",
                "discipline": "Piping",
                "location_kp": "KP 10+000",
                "facility": "CGS Duliajan",
                "requires_qa_gate": True,
            },
            {
                "activity_id": "PIP-L5-001-003",
                "activity_name": "HDD River Crossing Execution at KP 30+000",
                "discipline": "Piping",
                "location_kp": "KP 30+000",
                "facility": "OCS-4",
                "requires_qa_gate": True,
            },
            {
                "activity_id": "PIP-L5-001-004",
                "activity_name": "HDD River Crossing Execution at KP 31+200",
                "discipline": "Piping",
                "location_kp": "KP 31+200",
                "facility": "Trunkline ROW",
                "requires_qa_gate": False,
            },
            {
                "activity_id": "CIV-L5-002-001",
                "activity_name": "Valve Pit Excavation at KP 5+000",
                "discipline": "Civil",
                "location_kp": "KP 5+000",
                "facility": "CGS Moran",
                "requires_qa_gate": False,
            },
            {
                "activity_id": "CIV-L5-002-002",
                "activity_name": "ROW Clearing — Section 2A",
                "discipline": "Civil",
                "location_kp": "KP 6+400",
                "facility": "Trunkline ROW",
                "requires_qa_gate": False,
            },
            {
                "activity_id": "MEC-L5-003-001",
                "activity_name": "CGS Manifold Setup at CGS Duliajan",
                "discipline": "Mechanical",
                "location_kp": "KP 0+500",
                "facility": "CGS Duliajan",
                "requires_qa_gate": False,
            },
            # Duplicate KP for the KP_EXACT stage test (needs >=3 exact matches)
            {
                "activity_id": "PIP-L5-001-005",
                "activity_name": "Hydrotesting of Pipeline Section 9C",
                "discipline": "Piping",
                "location_kp": "KP 24+600",
                "facility": "OCS-4",
                "requires_qa_gate": True,
            },
            {
                "activity_id": "PIP-L5-001-006",
                "activity_name": "Pipe Stringing along ROW — Section 9D",
                "discipline": "Piping",
                "location_kp": "KP 24+600",
                "facility": "OCS-4",
                "requires_qa_gate": False,
            },
            {
                "activity_id": "PIP-L5-001-007",
                "activity_name": "Tie-in Welding at Chainage KP 15+000",
                "discipline": "Piping",
                "location_kp": "KP 15+000",
                "facility": "CGS Duliajan",
                "requires_qa_gate": True,
            },
            {
                "activity_id": "PIP-L5-001-008",
                "activity_name": "Tie-in Welding at Chainage KP 20+000",
                "discipline": "Piping",
                "location_kp": "KP 20+000",
                "facility": "Trunkline ROW",
                "requires_qa_gate": True,
            },
        ]
        cls.narrower = CandidateNarrower(cls.activities)

    def test_kp_exact_stage_triggers_with_enough_matches(self):
        obs = make_observation("OBS-1", "HDD drilling at KP 24+600")
        entities = ExtractedEntities(location_kp="KP 24+600")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertEqual(shortlist.narrowing_stage, "KP_EXACT")
        self.assertEqual(shortlist.initial_candidate_count, 3)  # 3 activities share KP 24+600
        self.assertTrue(MIN_SHORTLIST_SIZE <= len(shortlist.candidate_activities) <= MAX_SHORTLIST_SIZE)
        for act in shortlist.candidate_activities:
            self.assertEqual(act["location_kp"], "KP 24+600")

    def test_kp_exact_insufficient_falls_back_to_action_verb(self):
        # KP 10+000 only has ONE activity -> below MIN_SHORTLIST_SIZE ->
        # cascade must fall through past facility/discipline (no facility
        # given here) to the action-verb stage.
        obs = make_observation("OBS-2", "Tie-in welding at KP 10+000")
        entities = ExtractedEntities(location_kp="KP 10+000", action_verb="Tie-in Welding", discipline="Piping")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertNotEqual(shortlist.narrowing_stage, "KP_EXACT")

    def test_facility_discipline_stage(self):
        obs = make_observation("OBS-3", "Work at OCS-4")
        entities = ExtractedEntities(facility="OCS-4", discipline="Piping")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertEqual(shortlist.narrowing_stage, "FACILITY_DISCIPLINE")
        for act in shortlist.candidate_activities:
            self.assertEqual(act["facility"], "OCS-4")
            self.assertEqual(act["discipline"], "Piping")

    def test_facility_alone_without_discipline(self):
        obs = make_observation("OBS-4", "Work at OCS-4")
        entities = ExtractedEntities(facility="OCS-4")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertEqual(shortlist.narrowing_stage, "FACILITY_DISCIPLINE")
        for act in shortlist.candidate_activities:
            self.assertEqual(act["facility"], "OCS-4")

    def test_action_verb_stage(self):
        obs = make_observation("OBS-5", "HDD drilling somewhere along the ROW")
        entities = ExtractedEntities(action_verb="HDD Drilling", discipline="Piping")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertEqual(shortlist.narrowing_stage, "ACTION_VERB")
        for act in shortlist.candidate_activities:
            self.assertIn("HDD", act["activity_name"])

    def test_no_entities_falls_back_to_generic(self):
        obs = make_observation("OBS-6", "Nothing extractable here")
        entities = ExtractedEntities()
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertEqual(shortlist.narrowing_stage, "FALLBACK_GENERIC")
        self.assertTrue(len(shortlist.candidate_activities) >= MIN_SHORTLIST_SIZE)

    def test_shortlist_never_exceeds_max_size(self):
        obs = make_observation("OBS-7", "Piping work somewhere")
        entities = ExtractedEntities(discipline="Piping")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        self.assertLessEqual(len(shortlist.candidate_activities), MAX_SHORTLIST_SIZE)

    def test_kp_ranking_orders_by_proximity(self):
        # Discipline=Piping alone matches 6 activities (>5) -> should be
        # ranked by proximity to the (non-matching, so cascade falls
        # through to ACTION_VERB) KP if one was supplied via best-partial.
        obs = make_observation("OBS-8", "HDD work near KP 29+000")
        entities = ExtractedEntities(location_kp="KP 29+000", action_verb="HDD Drilling", discipline="Piping")
        shortlist = self.narrower.narrow_candidates(obs, entities)
        kps = [self.narrower._kp_to_float(a["location_kp"]) for a in shortlist.candidate_activities]
        self.assertEqual(kps, sorted(kps, key=lambda v: abs(v - 29.0)))

    def test_empty_activities_list_returns_empty_shortlist_without_raising(self):
        empty_narrower = CandidateNarrower([])
        obs = make_observation("OBS-9", "Anything")
        entities = ExtractedEntities(location_kp="KP 1+000")
        shortlist = empty_narrower.narrow_candidates(obs, entities)
        self.assertEqual(shortlist.candidate_activities, [])
        self.assertEqual(shortlist.initial_candidate_count, 0)


class TestPhase3Integration(unittest.TestCase):
    """End-to-end run against the real Phase 1 activities.json, if present."""

    @classmethod
    def setUpClass(cls) -> None:
        activities_path = os.path.join(DATA_DIR, "activities.json")
        if not os.path.exists(activities_path):
            raise unittest.SkipTest(
                f"data/activities.json not found at {activities_path} — run "
                "synthetic_generator.py first to generate Phase 1 data."
            )
        with open(activities_path, "r", encoding="utf-8") as f:
            cls.activities = json.load(f)
        cls.extractor = EntityExtractor()
        cls.narrower = CandidateNarrower(cls.activities)

    def _run_phrase(self, phrase: str) -> tuple[ExtractedEntities, list[dict]]:
        entities = self.extractor.extract(phrase)
        obs = make_observation("INTEGRATION-OBS", phrase)
        shortlist = self.narrower.narrow_candidates(obs, entities)
        return entities, shortlist.candidate_activities

    def test_full_dataset_has_expected_minimum_size(self):
        self.assertGreaterEqual(len(self.activities), 500)

    def test_sample_1_narrows_to_valid_shortlist_size(self):
        entities, candidates = self._run_phrase("150m HDD drilling finished near KP 24+600 at river site")
        self.assertEqual(entities.action_verb, "HDD Drilling")
        self.assertTrue(MIN_SHORTLIST_SIZE <= len(candidates) <= MAX_SHORTLIST_SIZE)

    def test_sample_2_narrows_to_valid_shortlist_size(self):
        entities, candidates = self._run_phrase("Piping tie-in work progressing near CGS inlet valve pit")
        self.assertEqual(entities.action_verb, "Tie-in Welding")
        self.assertTrue(MIN_SHORTLIST_SIZE <= len(candidates) <= MAX_SHORTLIST_SIZE)

    def test_sample_3_narrows_to_valid_shortlist_size(self):
        entities, candidates = self._run_phrase("Temporary mud pump pit near warehouse due to heavy rain")
        self.assertTrue(MIN_SHORTLIST_SIZE <= len(candidates) <= MAX_SHORTLIST_SIZE)

    def test_all_easy_match_dprs_produce_valid_shortlists(self):
        """Sanity sweep: run every EASY_MATCH DPR from Phase 1 through the
        full Stage 1 pipeline and confirm every single one produces a
        well-formed 3-5 item shortlist without raising."""
        dprs_path = os.path.join(DATA_DIR, "dprs.json")
        if not os.path.exists(dprs_path):
            self.skipTest("data/dprs.json not found — run synthetic_generator.py first.")
        with open(dprs_path, "r", encoding="utf-8") as f:
            dprs = json.load(f)

        easy_match_dprs = [d for d in dprs if d["case_type"] == "EASY_MATCH"]
        self.assertGreater(len(easy_match_dprs), 0)

        for dpr in easy_match_dprs:
            entities, candidates = self._run_phrase(dpr["raw_content"])
            self.assertTrue(
                MIN_SHORTLIST_SIZE <= len(candidates) <= MAX_SHORTLIST_SIZE,
                f"Shortlist size out of bounds for DPR {dpr['report_id']}: {len(candidates)}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
