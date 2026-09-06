"""
test_phase5.py
================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 5 DELIVERABLE — Test suite
--------------------------------------
Run with either:

    python -m unittest test_phase5.py -v
    pytest test_phase5.py -v

Coverage:
    * TestProgressEngine       — the spec's Day 1 / Day 2 scenario exactly
                                  (150m HDD claim -> 30% physical, then NDT
                                  clearance -> 30% verified), plus edge
                                  cases: unmatched decisions, unknown
                                  activities, unit mismatches, cumulative
                                  claims across multiple DPRs, QA capping,
                                  and over-claim percentage capping.
    * TestAuditLogger           — SHA-256 hash generation/determinism,
                                  ledger append behavior, and the tamper
                                  detection scenario from the spec.
    * TestAuditLoggerConcurrency — an actual multi-threaded stress test
                                  proving the "thread-safe for ledger
                                  writes" requirement, not just asserting it
                                  in a docstring.
    * TestPhase5Integration     — the full Day1/Day2 scenario run through
                                  ProgressEngine AND AuditLogger together,
                                  matching the spec's three named test cases
                                  end-to-end.

All tests use isolated temp-directory ledger files (never touching the
real data/evidence_ledger.json), so this suite is fully repeatable and
leaves no side effects behind.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import unittest

from audit_logger import GENESIS_HASH, AuditLogger
from progress_engine import ProgressEngine
from schemas import MatchDecision, NormalizedObservation

logging.disable(logging.CRITICAL)


# --------------------------------------------------------------------------
# Shared test fixtures
# --------------------------------------------------------------------------
def make_hdd_activity(planned_quantity: float = 0.500) -> dict:
    """The exact activity the spec's Day 1/Day 2 scenario is built around."""
    return {
        "activity_id": "PIP-L5-024-003",
        "activity_name": "HDD River Crossing Execution at KP 24+600",
        "discipline": "Piping",
        "location_kp": "KP 24+600",
        "facility": "OCS-4",
        "planned_quantity": planned_quantity,
        "unit": "KM",
        "planned_start": "2026-01-05",
        "planned_finish": "2026-01-20",
        "baseline_duration_days": 15,
        "requires_qa_gate": True,
        "qa_gate_type": "NDT_RADIOGRAPHY",
    }


def make_observation(
    obs_id: str,
    raw_phrase: str,
    normalized_quantity: float,
    normalized_unit: str = "KM",
    is_qa_clearance: bool = False,
    report_id: str = "TEST-REPORT",
) -> NormalizedObservation:
    return NormalizedObservation(
        observation_id=obs_id,
        report_id=report_id,
        raw_phrase=raw_phrase,
        raw_quantity=normalized_quantity * 1000 if normalized_unit == "KM" else normalized_quantity,
        raw_unit="m" if normalized_unit == "KM" else normalized_unit.lower(),
        normalized_quantity=normalized_quantity,
        normalized_unit=normalized_unit,
        conversion_applied="Meters to KM (div 1000)" if normalized_unit == "KM" else "n/a",
        is_qa_clearance=is_qa_clearance,
    )


def make_decision(
    match_id: str,
    observation_id: str,
    activity_id: str | None,
    decision_type: str = "AUTO_ACCEPT",
) -> MatchDecision:
    return MatchDecision(
        match_id=match_id,
        observation_id=observation_id,
        selected_activity_id=activity_id,
        entity_score=1.0,
        semantic_score=1.0,
        quantity_score=1.0,
        final_confidence_score=1.0 if activity_id else 0.0,
        decision_type=decision_type,
        reasoning="test fixture",
        candidate_scores=[],
    )


class TestProgressEngine(unittest.TestCase):
    """Dual progress tracking — the spec's Day 1/Day 2 scenario and edge cases."""

    def setUp(self) -> None:
        self.engine = ProgressEngine([make_hdd_activity()])

    # -- Spec Day 1: physical claim -----------------------------------------
    def test_day1_physical_claim_updates_physical_progress_only(self):
        obs = make_observation("DAY1-OBS-001", "150m HDD drilling finished near KP 24+600", 0.150, "KM")
        decision = make_decision("MATCH-0001", "DAY1-OBS-001", "PIP-L5-024-003")
        state = self.engine.apply_match_decision(decision, obs)

        self.assertEqual(state.physical_progress_pct, 30.0)
        self.assertEqual(state.physical_claimed_quantity, 0.150)
        self.assertEqual(state.verified_progress_pct, 0.0)
        self.assertEqual(state.verified_earned_quantity, 0.0)
        self.assertEqual(state.qa_gate_status, "PENDING_QA")

    # -- Spec Day 2: QA clearance unlocks verified progress ------------------
    def test_day2_qa_clearance_unlocks_verified_progress(self):
        obs1 = make_observation("DAY1-OBS-001", "150m HDD drilling finished near KP 24+600", 0.150, "KM")
        d1 = make_decision("MATCH-0001", "DAY1-OBS-001", "PIP-L5-024-003")
        self.engine.apply_match_decision(d1, obs1)

        obs2 = make_observation(
            "DAY2-OBS-001", "NDT Radiography passed for 150m HDD section at KP 24+600",
            0.150, "KM", is_qa_clearance=True,
        )
        d2 = make_decision("MATCH-0002", "DAY2-OBS-001", "PIP-L5-024-003")
        state = self.engine.apply_match_decision(d2, obs2)

        self.assertEqual(state.verified_progress_pct, 30.0)
        self.assertEqual(state.verified_earned_quantity, 0.150)
        self.assertEqual(state.qa_gate_status, "VERIFIED_PASSED")
        # Physical progress is untouched by the QA-branch update.
        self.assertEqual(state.physical_progress_pct, 30.0)

    # -- Cumulative claims across multiple DPRs ------------------------------
    def test_multiple_physical_claims_accumulate(self):
        obs1 = make_observation("OBS-A", "text", 0.100, "KM")
        obs2 = make_observation("OBS-B", "text", 0.150, "KM")
        self.engine.apply_match_decision(make_decision("M-A", "OBS-A", "PIP-L5-024-003"), obs1)
        state = self.engine.apply_match_decision(make_decision("M-B", "OBS-B", "PIP-L5-024-003"), obs2)

        self.assertAlmostEqual(state.physical_claimed_quantity, 0.250, places=4)
        self.assertAlmostEqual(state.physical_progress_pct, 50.0, places=2)  # 0.250/0.500

    # -- QA verification is capped at physical claimed -----------------------
    def test_qa_verification_capped_at_physical_claimed(self):
        # Claim only 0.100 KM physically...
        obs1 = make_observation("OBS-A", "text", 0.100, "KM")
        self.engine.apply_match_decision(make_decision("M-A", "OBS-A", "PIP-L5-024-003"), obs1)
        # ...but a QA report tries to verify 0.150 KM (more than claimed).
        obs2 = make_observation("OBS-B", "NDT passed", 0.150, "KM", is_qa_clearance=True)
        state = self.engine.apply_match_decision(make_decision("M-B", "OBS-B", "PIP-L5-024-003"), obs2)

        # verified_earned_quantity must never exceed physical_claimed_quantity.
        self.assertEqual(state.verified_earned_quantity, 0.100)
        self.assertEqual(state.verified_progress_pct, 20.0)  # 0.100/0.500

    # -- Percentage capping on overclaim --------------------------------------
    def test_physical_progress_pct_caps_at_100_on_overclaim(self):
        obs = make_observation("OBS-OVER", "text", 0.750, "KM")  # exceeds planned 0.500
        state = self.engine.apply_match_decision(make_decision("M-OVER", "OBS-OVER", "PIP-L5-024-003"), obs)

        self.assertEqual(state.physical_progress_pct, 100.0)
        # The raw claimed quantity itself is NOT truncated — audit trail preserved.
        self.assertEqual(state.physical_claimed_quantity, 0.750)

    # -- No QA gate required activity ----------------------------------------
    def test_activity_without_qa_gate_starts_not_required(self):
        no_gate_activity = make_hdd_activity()
        no_gate_activity["activity_id"] = "CIV-L5-001-001"
        no_gate_activity["requires_qa_gate"] = False
        no_gate_activity["qa_gate_type"] = None
        engine = ProgressEngine([no_gate_activity])
        state = engine.get_progress_state("CIV-L5-001-001")

        self.assertEqual(state.qa_gate_status, "NOT_REQUIRED")
        self.assertEqual(state.qa_gate_type, "NONE")

    # -- Edge cases: unmatched / unknown activity / unit mismatch -----------
    def test_unmatched_decision_raises_value_error(self):
        obs = make_observation("OBS-X", "text", 0.1, "KM")
        decision = make_decision("M-X", "OBS-X", None, decision_type="UNMATCHED")
        with self.assertRaises(ValueError):
            self.engine.apply_match_decision(decision, obs)

    def test_unknown_activity_id_raises_value_error(self):
        obs = make_observation("OBS-Y", "text", 0.1, "KM")
        decision = make_decision("M-Y", "OBS-Y", "NONEXISTENT-ACTIVITY")
        with self.assertRaises(ValueError):
            self.engine.apply_match_decision(decision, obs)

    def test_unit_mismatch_skips_update_without_raising(self):
        obs = make_observation("OBS-Z", "text", 5.0, "JOINTS")  # activity is tracked in KM
        decision = make_decision("M-Z", "OBS-Z", "PIP-L5-024-003")
        state = self.engine.apply_match_decision(decision, obs)

        self.assertEqual(state.physical_claimed_quantity, 0.0)  # unchanged
        self.assertEqual(state.physical_progress_pct, 0.0)

    def test_get_all_progress_states(self):
        states = self.engine.get_all_progress_states()
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].activity_id, "PIP-L5-024-003")

    def test_get_progress_state_unknown_returns_none(self):
        self.assertIsNone(self.engine.get_progress_state("DOES-NOT-EXIST"))

    def test_invalid_planned_quantity_activity_is_skipped(self):
        bad_activity = make_hdd_activity(planned_quantity=0.0)
        bad_activity["activity_id"] = "BAD-ACT"
        engine = ProgressEngine([bad_activity, make_hdd_activity()])
        self.assertIsNone(engine.get_progress_state("BAD-ACT"))
        self.assertIsNotNone(engine.get_progress_state("PIP-L5-024-003"))


class TestAuditLogger(unittest.TestCase):
    """SHA-256 hashing, ledger append behavior, and tamper detection."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmpdir, "evidence_ledger.json")
        self.logger = AuditLogger(ledger_path=self.ledger_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- Hash generation ----------------------------------------------------
    def test_generate_hash_is_deterministic(self):
        data = {"a": 1, "b": "text", "c": 3.14}
        self.assertEqual(self.logger.generate_hash(data), self.logger.generate_hash(data))

    def test_generate_hash_ignores_key_order(self):
        h1 = self.logger.generate_hash({"a": 1, "b": 2})
        h2 = self.logger.generate_hash({"b": 2, "a": 1})
        self.assertEqual(h1, h2)

    def test_generate_hash_is_sensitive_to_value_changes(self):
        h1 = self.logger.generate_hash({"quantity": 0.150})
        h2 = self.logger.generate_hash({"quantity": 0.151})
        self.assertNotEqual(h1, h2)

    def test_generate_hash_returns_valid_sha256_hex(self):
        h = self.logger.generate_hash({"x": 1})
        self.assertEqual(len(h), 64)
        int(h, 16)  # raises ValueError if not valid hex — implicitly asserts format

    # -- Ledger append behavior -----------------------------------------------
    def test_log_evidence_creates_and_persists_entry(self):
        obs = make_observation("OBS-1", "text", 0.150, "KM")
        decision = make_decision("M-1", "OBS-1", "PIP-L5-024-003")
        evidence = self.logger.log_evidence(decision, obs, "PHYSICAL_CLAIM")

        self.assertTrue(evidence.evidence_id.startswith("EV-"))
        self.assertEqual(evidence.activity_id, "PIP-L5-024-003")
        self.assertEqual(len(evidence.evidence_hash), 64)
        self.assertTrue(os.path.exists(self.ledger_path))

        ledger = self.logger.get_ledger()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["evidence_id"], evidence.evidence_id)

    def test_first_entry_chains_from_genesis_hash(self):
        obs = make_observation("OBS-1", "text", 0.1, "KM")
        decision = make_decision("M-1", "OBS-1", "PIP-L5-024-003")
        self.logger.log_evidence(decision, obs, "PHYSICAL_CLAIM")
        ledger = self.logger.get_ledger()
        # We can't directly see previous_hash (not a stored field), but we
        # CAN confirm the chain validates from genesis via verify_ledger_integrity.
        self.assertTrue(self.logger.verify_ledger_integrity())
        self.assertNotEqual(ledger[0]["evidence_hash"], GENESIS_HASH)

    def test_multiple_entries_chain_correctly(self):
        obs1 = make_observation("OBS-1", "text", 0.1, "KM")
        obs2 = make_observation("OBS-2", "text", 0.1, "KM", is_qa_clearance=True)
        self.logger.log_evidence(make_decision("M-1", "OBS-1", "PIP-L5-024-003"), obs1, "PHYSICAL_CLAIM")
        self.logger.log_evidence(make_decision("M-2", "OBS-2", "PIP-L5-024-003"), obs2, "QA_VERIFIED")

        self.assertEqual(len(self.logger.get_ledger()), 2)
        self.assertTrue(self.logger.verify_ledger_integrity())

    def test_log_evidence_unmatched_decision_raises(self):
        obs = make_observation("OBS-X", "text", 0.1, "KM")
        decision = make_decision("M-X", "OBS-X", None, decision_type="UNMATCHED")
        with self.assertRaises(ValueError):
            self.logger.log_evidence(decision, obs, "PHYSICAL_CLAIM")

    def test_reviewer_id_recorded_when_supplied(self):
        obs = make_observation("OBS-R", "text", 0.1, "KM")
        decision = make_decision("M-R", "OBS-R", "PIP-L5-024-003", decision_type="HUMAN_REVIEW")
        evidence = self.logger.log_evidence(decision, obs, "PHYSICAL_CLAIM", reviewer_id="planner_priya")
        self.assertEqual(evidence.reviewer_id, "planner_priya")

    # -- Empty ledger integrity ------------------------------------------------
    def test_empty_ledger_is_trivially_intact(self):
        self.assertTrue(self.logger.verify_ledger_integrity())

    # -- The spec's tamper detection scenario ------------------------------------
    def test_tampering_with_a_record_is_detected(self):
        obs1 = make_observation("OBS-1", "150m HDD drilling finished", 0.150, "KM")
        obs2 = make_observation("OBS-2", "NDT passed", 0.150, "KM", is_qa_clearance=True)
        self.logger.log_evidence(make_decision("M-1", "OBS-1", "PIP-L5-024-003"), obs1, "PHYSICAL_CLAIM")
        self.logger.log_evidence(make_decision("M-2", "OBS-2", "PIP-L5-024-003"), obs2, "QA_VERIFIED")

        self.assertTrue(self.logger.verify_ledger_integrity())

        # Directly tamper with the ledger file, as an attacker with
        # filesystem access (but not the ability to recompute a valid
        # hash chain) might attempt.
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        ledger[0]["quantity_added"] = 999.0  # falsify the first entry's quantity
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

        self.assertFalse(self.logger.verify_ledger_integrity())

    def test_tampering_with_later_record_still_detected(self):
        """Confirm the SECOND (not just the first) entry is also protected —
        proving the hash chain, not just a single record's own hash."""
        obs1 = make_observation("OBS-1", "text", 0.1, "KM")
        obs2 = make_observation("OBS-2", "text", 0.1, "KM", is_qa_clearance=True)
        self.logger.log_evidence(make_decision("M-1", "OBS-1", "PIP-L5-024-003"), obs1, "PHYSICAL_CLAIM")
        self.logger.log_evidence(make_decision("M-2", "OBS-2", "PIP-L5-024-003"), obs2, "QA_VERIFIED")

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        ledger[1]["reviewer_id"] = "someone_who_never_reviewed_this"
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

        self.assertFalse(self.logger.verify_ledger_integrity())

    def test_deleting_a_record_breaks_the_chain(self):
        """Deletion (not just field tampering) must also be detected —
        this is exactly what hash-chaining catches that isolated per-record
        hashing would miss."""
        for i in range(3):
            obs = make_observation(f"OBS-{i}", "text", 0.05, "KM")
            self.logger.log_evidence(make_decision(f"M-{i}", f"OBS-{i}", "PIP-L5-024-003"), obs, "PHYSICAL_CLAIM")
        self.assertTrue(self.logger.verify_ledger_integrity())

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        del ledger[1]  # remove the middle entry
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

        self.assertFalse(self.logger.verify_ledger_integrity())

    def test_corrupted_ledger_file_treated_as_empty_without_raising(self):
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        # Should not raise — corrupted files degrade to an empty ledger read.
        ledger = self.logger.get_ledger()
        self.assertEqual(ledger, [])


class TestAuditLoggerConcurrency(unittest.TestCase):
    """Proves thread-safety with an actual concurrent write stress test,
    rather than just asserting it in a docstring."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmpdir, "evidence_ledger.json")
        self.logger = AuditLogger(ledger_path=self.ledger_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_writes_do_not_lose_entries_or_break_the_chain(self):
        NUM_THREADS = 25
        errors: list[Exception] = []

        def write_one(i: int) -> None:
            try:
                obs = make_observation(f"OBS-CONC-{i}", f"concurrent write {i}", 0.01, "KM")
                decision = make_decision(f"M-CONC-{i}", f"OBS-CONC-{i}", "PIP-L5-024-003")
                self.logger.log_evidence(decision, obs, "PHYSICAL_CLAIM")
            except Exception as exc:  # pragma: no cover — captured for the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=write_one, args=(i,)) for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent writes raised: {errors}")

        ledger = self.logger.get_ledger()
        self.assertEqual(len(ledger), NUM_THREADS, "Lost writes under concurrency — race condition present.")

        observation_ids = {entry["observation_id"] for entry in ledger}
        self.assertEqual(len(observation_ids), NUM_THREADS, "Duplicate/overwritten entries under concurrency.")

        self.assertTrue(
            self.logger.verify_ledger_integrity(),
            "Hash chain broken after concurrent writes — entries were interleaved incorrectly.",
        )


class TestPhase5Integration(unittest.TestCase):
    """The spec's full Day1 / Day2 / Audit Integrity scenario, run through
    ProgressEngine and AuditLogger together."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmpdir, "evidence_ledger.json")
        self.engine = ProgressEngine([make_hdd_activity()])
        self.logger = AuditLogger(ledger_path=self.ledger_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_day1_day2_and_audit_integrity_end_to_end(self):
        # --- Day 1: supervisor DPR ---
        obs1 = make_observation(
            "DAY1-OBS-001", "150m HDD drilling finished near KP 24+600", 0.150, "KM", report_id="DPR-DAY1"
        )
        d1 = make_decision("MATCH-0001", "DAY1-OBS-001", "PIP-L5-024-003")
        state1 = self.engine.apply_match_decision(d1, obs1)
        self.assertEqual(state1.physical_progress_pct, 30.0)
        self.assertEqual(state1.verified_progress_pct, 0.0)
        self.assertEqual(state1.qa_gate_status, "PENDING_QA")
        ev1 = self.logger.log_evidence(d1, obs1, "PHYSICAL_CLAIM")

        # --- Day 2: NDT inspection report ---
        obs2 = make_observation(
            "DAY2-OBS-001", "NDT Radiography passed for 150m HDD section at KP 24+600",
            0.150, "KM", is_qa_clearance=True, report_id="DPR-DAY2",
        )
        d2 = make_decision("MATCH-0002", "DAY2-OBS-001", "PIP-L5-024-003")
        state2 = self.engine.apply_match_decision(d2, obs2)
        self.assertEqual(state2.verified_progress_pct, 30.0)
        self.assertEqual(state2.qa_gate_status, "VERIFIED_PASSED")
        ev2 = self.logger.log_evidence(d2, obs2, "QA_VERIFIED")

        # --- Audit integrity ---
        self.assertTrue(self.logger.verify_ledger_integrity())
        self.assertEqual(len(self.logger.get_ledger()), 2)
        self.assertNotEqual(ev1.evidence_hash, ev2.evidence_hash)

        # Tamper and re-verify.
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        ledger[1]["quantity_added"] = 5.0
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
        self.assertFalse(self.logger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main(verbosity=2)
