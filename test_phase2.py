"""
test_phase2.py
===============
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 2 DELIVERABLE — Unit test suite
------------------------------------------
Written with the standard-library ``unittest`` framework (zero extra
dependency required to run it) but the classes/methods are fully
pytest-discoverable too — both of these work:

    python -m unittest test_phase2.py -v
    pytest test_phase2.py -v

Coverage:
    * TestUnitNormalizer   — the five spec-mandated conversion cases, plus
                              edge cases (unknown unit, negative quantity,
                              feet, mismatched target_unit override).
    * TestIngestionEngine  — end-to-end ReportEvent -> NormalizedObservation
                              flow, QA clearance detection (positive and
                              negative), multi-quantity reports, and
                              zero-quantity ("noise") reports.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from ingestion import IngestionEngine
from schemas import NormalizedObservation, ReportEvent
from unit_normalizer import UnitConversionError, UnitNormalizer


def make_report(report_id: str, raw_content: str, source_type: str = "FREE_TEXT") -> ReportEvent:
    """Small helper to keep test cases short and readable."""
    return ReportEvent(
        report_id=report_id,
        source_type=source_type,
        submitted_by="Rakesh Sharma (Site Inspector)",
        submission_timestamp=datetime(2026, 8, 24, 10, 30, tzinfo=timezone.utc),
        raw_content=raw_content,
    )


class TestUnitNormalizer(unittest.TestCase):
    """Direct tests of the deterministic conversion engine, independent of
    the ingestion/report layer."""

    def setUp(self) -> None:
        self.normalizer = UnitNormalizer()

    # -- Spec-mandated conversion accuracy cases --------------------------
    def test_meters_to_km(self):
        matches = self.normalizer.parse("150m HDD drilling finished")
        self.assertEqual(len(matches), 1)
        qty, unit, audit = self.normalizer.normalize(matches[0].raw_quantity, matches[0].raw_unit)
        self.assertAlmostEqual(qty, 0.150, places=3)
        self.assertEqual(unit, "KM")
        self.assertIn("div", audit.lower())

    def test_km_passthrough(self):
        matches = self.normalizer.parse("2.5 km pipeline trenching")
        self.assertEqual(len(matches), 1)
        qty, unit, _ = self.normalizer.normalize(matches[0].raw_quantity, matches[0].raw_unit)
        self.assertAlmostEqual(qty, 2.500, places=3)
        self.assertEqual(unit, "KM")

    def test_kg_to_tonnes(self):
        matches = self.normalizer.parse("500 kg welding electrode used")
        self.assertEqual(len(matches), 1)
        qty, unit, _ = self.normalizer.normalize(matches[0].raw_quantity, matches[0].raw_unit)
        self.assertAlmostEqual(qty, 0.500, places=3)
        self.assertEqual(unit, "TONNES")

    def test_joints_preserved(self):
        matches = self.normalizer.parse("8 joints welded today")
        self.assertEqual(len(matches), 1)
        qty, unit, _ = self.normalizer.normalize(matches[0].raw_quantity, matches[0].raw_unit)
        self.assertAlmostEqual(qty, 8.0, places=3)
        self.assertEqual(unit, "JOINTS")

    def test_qa_clearance_flag_via_ingestion(self):
        # is_qa_clearance lives on NormalizedObservation, which is assembled
        # by IngestionEngine, not UnitNormalizer directly — exercised here
        # end-to-end for the spec's fifth required case.
        engine = IngestionEngine()
        report = make_report("DPR-TEST-005", "NDT radiography passed for 150m section")
        observations = engine.process_report(report)
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].is_qa_clearance)
        self.assertAlmostEqual(observations[0].normalized_quantity, 0.150, places=3)
        self.assertEqual(observations[0].normalized_unit, "KM")

    # -- Additional conversion rules ---------------------------------------
    def test_feet_to_meters(self):
        matches = self.normalizer.parse("10 ft trench depth recorded")
        self.assertEqual(len(matches), 1)
        qty, unit, _ = self.normalizer.normalize(matches[0].raw_quantity, matches[0].raw_unit)
        self.assertAlmostEqual(qty, 3.048, places=3)
        self.assertEqual(unit, "M")

    def test_spools_and_pit_preserved(self):
        qty, unit, _ = self.normalizer.normalize(4, "spools")
        self.assertEqual((qty, unit), (4.0, "SPOOLS"))
        qty, unit, _ = self.normalizer.normalize(1, "pit")
        self.assertEqual((qty, unit), (1.0, "PIT"))

    def test_unit_alias_case_insensitivity(self):
        qty, unit, _ = self.normalizer.normalize(3.5, "TONNES")
        self.assertEqual((qty, unit), (3.5, "TONNES"))
        qty, unit, _ = self.normalizer.normalize(3.5, "Tonne")
        self.assertEqual((qty, unit), (3.5, "TONNES"))

    # -- Edge cases / error handling ---------------------------------------
    def test_unknown_unit_raises(self):
        with self.assertRaises(UnitConversionError):
            self.normalizer.normalize(100.0, "furlongs")

    def test_negative_quantity_raises(self):
        with self.assertRaises(ValueError):
            self.normalizer.normalize(-5.0, "m")

    def test_mismatched_target_unit_raises(self):
        with self.assertRaises(UnitConversionError):
            self.normalizer.normalize(100.0, "m", target_unit="TONNES")

    def test_no_quantity_in_text_returns_empty(self):
        matches = self.normalizer.parse("Piping work progressing near CGS inlet valve pit.")
        # "pit" here is a unit-alias word but has no attached number, so no
        # match should be produced — this guards against false positives
        # on bare unit words with no numeric quantity.
        self.assertEqual(matches, [])

    def test_blank_text_returns_empty(self):
        self.assertEqual(self.normalizer.parse(""), [])
        self.assertEqual(self.normalizer.parse("   "), [])

    def test_does_not_false_match_inside_longer_word(self):
        # "5mm" should not be misread as "5m" + stray "m" — the negative
        # lookahead in the trailing pattern guards against mid-word matches.
        matches = self.normalizer.parse("Clearance gap measured at 5mm on the flange.")
        for m in matches:
            self.assertNotEqual(m.raw_unit, "m")


class TestIngestionEngine(unittest.TestCase):
    """End-to-end tests: ReportEvent -> list[NormalizedObservation]."""

    def setUp(self) -> None:
        self.engine = IngestionEngine()

    def test_single_observation_report(self):
        report = make_report("DPR-TEST-001", "150m HDD drilling completed today near KP 24+600 at river site.")
        observations = self.engine.process_report(report)
        self.assertEqual(len(observations), 1)
        obs = observations[0]
        self.assertIsInstance(obs, NormalizedObservation)
        self.assertEqual(obs.report_id, "DPR-TEST-001")
        self.assertEqual(obs.observation_id, "DPR-TEST-001-OBS-001")
        self.assertAlmostEqual(obs.normalized_quantity, 0.150, places=3)
        self.assertEqual(obs.normalized_unit, "KM")
        self.assertFalse(obs.is_qa_clearance)

    def test_unit_mismatch_report_converts_correctly(self):
        report = make_report("DPR-TEST-002", "750 meters trenching completed on Section 4B.")
        observations = self.engine.process_report(report)
        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations[0].normalized_quantity, 0.750, places=3)
        self.assertEqual(observations[0].normalized_unit, "KM")

    def test_multi_quantity_report_produces_multiple_observations(self):
        report = make_report(
            "DPR-TEST-003",
            "500 kg welding electrode used and 8 joints welded today at the tie-in point.",
        )
        observations = self.engine.process_report(report)
        self.assertEqual(len(observations), 2)
        units = {obs.normalized_unit for obs in observations}
        self.assertEqual(units, {"TONNES", "JOINTS"})

    def test_noise_report_produces_zero_observations(self):
        report = make_report(
            "DPR-TEST-004",
            "Constructed temporary mud pump pit near yard due to heavy rain.",
        )
        observations = self.engine.process_report(report)
        self.assertEqual(observations, [])

    def test_qa_clearance_negative_case_not_flagged(self):
        report = make_report("DPR-TEST-006", "150m HDD drilling completed today near KP 24+600.")
        observations = self.engine.process_report(report)
        self.assertEqual(len(observations), 1)
        self.assertFalse(observations[0].is_qa_clearance)

    def test_qa_clearance_scheduled_not_yet_passed_not_flagged(self):
        # Guards against over-eager keyword matching: mentioning a future
        # QA gate should NOT be treated as a clearance event.
        report = make_report("DPR-TEST-007", "Hydrotest scheduled for next Friday at KP 12+000.")
        observations = self.engine.process_report(report)
        self.assertFalse(self.engine.detect_qa_clearance(report.raw_content))

    def test_hydrotest_clearance_phrasing(self):
        report = make_report("DPR-TEST-008", "Hydrotest clearance obtained for 2 km pipeline section.")
        observations = self.engine.process_report(report)
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].is_qa_clearance)

    def test_cube_test_passed_phrasing(self):
        report = make_report("DPR-TEST-009", "Cube test passed for foundation works, 12 tonnes concrete poured.")
        observations = self.engine.process_report(report)
        self.assertTrue(all(obs.is_qa_clearance for obs in observations))
        self.assertEqual(len(observations), 1)  # 12 tonnes
        self.assertEqual(observations[0].normalized_unit, "TONNES")

    def test_none_report_raises(self):
        with self.assertRaises(ValueError):
            self.engine.process_report(None)  # type: ignore[arg-type]

    def test_batch_processing_continues_past_empty_reports(self):
        reports = [
            make_report("DPR-BATCH-001", "150m HDD drilling completed today."),
            make_report("DPR-BATCH-002", "Toolbox talk conducted for all contractor staff."),
            make_report("DPR-BATCH-003", "2.5 km pipeline trenching completed."),
        ]
        observations = self.engine.process_reports(reports)
        self.assertEqual(len(observations), 2)  # BATCH-002 contributes none


if __name__ == "__main__":
    unittest.main(verbosity=2)
