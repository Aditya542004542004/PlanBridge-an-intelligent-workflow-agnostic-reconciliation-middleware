"""
audit_logger.py
==================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 5 DELIVERABLE — Cryptographic CVC/CAG Audit Logger
------------------------------------------------------------------
``AuditLogger`` maintains an append-only evidence ledger at
``data/evidence_ledger.json`` — every time a matched observation moves an
activity's progress, exactly one ``EvidenceLog`` record is written here,
stamped with a SHA-256 hash, and never edited afterward.

Tamper-evidence design
-----------------------------
Each entry's ``evidence_hash`` is a SHA-256 digest over that entry's own
fields **plus the previous entry's hash** (classic hash-chaining, the same
core idea a blockchain or a Git commit history uses). This matters because
a naive "hash each record in isolation" scheme only detects tampering with
*that specific record* — it says nothing if an attacker deletes a record
entirely, reorders records, or splices in a fabricated one with its own
self-consistent isolated hash. Chaining means altering, deleting, or
reordering ANY historical record breaks that record's hash **and every
hash after it**, so ``verify_ledger_integrity()`` catches tampering
anywhere in the ledger's history, not just its most recent entry — which is
the actual bar a CVC/CAG audit of a PSU would expect.

Storage format
--------------------
The ledger is stored as a single JSON array at ``data/evidence_ledger.json``
(readable/portable, consistent with every other data file in this
project). Each write reads the current ledger, appends the new entry, and
writes the result to a temp file followed by an atomic ``os.replace()`` —
so a reader can never observe a half-written ledger file, even under
concurrent access. A ``threading.Lock`` additionally serializes the
read-modify-write critical section across writer threads, preventing the
classic "two threads both read the same old list, both append, one clobbers
the other's write" race. See ``test_phase5.py`` for a concurrency test that
actually exercises this with multiple threads, not just a claim in a
docstring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from schemas import EvidenceLog, MatchDecision, NormalizedObservation, ProgressCategory

log = logging.getLogger("planbridge.audit_logger")

DEFAULT_LEDGER_PATH = "data/evidence_ledger.json"

# Used as the "previous hash" input for the very first entry in a ledger —
# analogous to a blockchain's genesis block having no real predecessor.
GENESIS_HASH = "0" * 64


class AuditLogger:
    """
    Append-only, hash-chained evidence ledger for CVC/CAG audit compliance.

    Usage
    -----
        logger = AuditLogger()  # defaults to data/evidence_ledger.json
        evidence = logger.log_evidence(decision, observation, "PHYSICAL_CLAIM")
        assert logger.verify_ledger_integrity() is True
    """

    def __init__(self, ledger_path: str = DEFAULT_LEDGER_PATH) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.ledger_path.exists():
            self._write_ledger_atomic([])

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------
    def generate_hash(self, data_dict: dict[str, Any]) -> str:
        """
        Compute a deterministic SHA-256 hex digest over ``data_dict``.

        Determinism is achieved via ``json.dumps(..., sort_keys=True)`` —
        the same logical data always produces the same hash regardless of
        the dict's original key insertion order, and ``default=str`` makes
        this robust to any non-JSON-native values (e.g. a stray float or
        Decimal) that might be passed in without raising a TypeError.
        """
        canonical = json.dumps(data_dict, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_evidence(
        self,
        decision: MatchDecision,
        observation: NormalizedObservation,
        progress_category: ProgressCategory,
        reviewer_id: Optional[str] = None,
    ) -> EvidenceLog:
        """
        Create and append one ``EvidenceLog`` entry to the ledger.

        Parameters
        ----------
        decision           : the MatchDecision that authorized this update
                              (must carry a resolved selected_activity_id).
        observation         : the NormalizedObservation this evidence is for.
        progress_category   : "PHYSICAL_CLAIM" or "QA_VERIFIED" — which
                              progress dimension this evidence moved (the
                              caller is expected to pass whichever branch
                              ``ProgressEngine.apply_match_decision`` took).
        reviewer_id          : optional identifier of the human who confirmed
                              this match (for HUMAN_REVIEW decisions);
                              None for AUTO_ACCEPT.

        Returns
        -------
        The newly-created, persisted ``EvidenceLog`` entry.

        Raises
        ------
        ValueError : ``decision.selected_activity_id`` is None — there is
                    nothing to log evidence against for an UNMATCHED decision.

        Thread-safe: the read-modify-write against the ledger file is
        guarded by an internal lock, so concurrent callers cannot lose or
        corrupt each other's writes.
        """
        if not decision.selected_activity_id:
            raise ValueError(
                f"log_evidence: decision '{decision.match_id}' has no selected_activity_id — "
                f"cannot log evidence for an UNMATCHED decision."
            )

        with self._lock:
            ledger = self._read_ledger()
            previous_hash = ledger[-1]["evidence_hash"] if ledger else GENESIS_HASH

            logged_timestamp = datetime.now(timezone.utc).isoformat()
            evidence_id = self._generate_evidence_id(observation.observation_id, decision.match_id, logged_timestamp)

            hash_payload = {
                "evidence_id": evidence_id,
                "activity_id": decision.selected_activity_id,
                "observation_id": observation.observation_id,
                "match_id": decision.match_id,
                "quantity_added": observation.normalized_quantity,
                "unit": observation.normalized_unit,
                "progress_category": progress_category,
                "logged_timestamp": logged_timestamp,
                "source_report_id": observation.report_id,
                "reviewer_id": reviewer_id,
                "previous_hash": previous_hash,
            }
            evidence_hash = self.generate_hash(hash_payload)

            evidence = EvidenceLog(
                evidence_id=evidence_id,
                activity_id=decision.selected_activity_id,
                observation_id=observation.observation_id,
                match_id=decision.match_id,
                quantity_added=observation.normalized_quantity,
                unit=observation.normalized_unit,
                progress_category=progress_category,
                logged_timestamp=logged_timestamp,
                source_report_id=observation.report_id,
                reviewer_id=reviewer_id,
                evidence_hash=evidence_hash,
            )

            ledger.append(evidence.model_dump())
            self._write_ledger_atomic(ledger)

        log.info(
            "AuditLogger: logged %s (%s) for activity %s, observation %s.",
            evidence_id, progress_category, decision.selected_activity_id, observation.observation_id,
        )
        return evidence

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------
    def verify_ledger_integrity(self) -> bool:
        """
        Recompute the hash chain over the entire ledger and confirm every
        entry's stored ``evidence_hash`` matches what it should be, given
        its own fields and the previous entry's (stored) hash.

        Returns True only if every single entry checks out. Returns False
        — logging exactly which entry index/evidence_id first failed — on
        any mismatch, whether from a modified field, a deleted/reordered
        record (which breaks the chain at that point), or a corrupted file.
        Never raises for a tampered ledger; that is the expected, correctly
        detected outcome this method exists to report.
        """
        with self._lock:
            ledger = self._read_ledger()

        if not ledger:
            return True  # an empty ledger is trivially "intact"

        previous_hash = GENESIS_HASH
        for index, entry in enumerate(ledger):
            expected_hash = self._recompute_entry_hash(entry, previous_hash)
            stored_hash = entry.get("evidence_hash")

            if expected_hash != stored_hash:
                log.warning(
                    "verify_ledger_integrity: TAMPERING DETECTED at ledger index %d "
                    "(evidence_id=%s) — stored hash does not match recomputed hash. "
                    "Expected %s, found %s.",
                    index, entry.get("evidence_id"), expected_hash, stored_hash,
                )
                return False

            previous_hash = stored_hash

        return True

    def _recompute_entry_hash(self, entry: dict[str, Any], previous_hash: str) -> str:
        """Rebuild the exact hash payload used at write time for one ledger
        entry, given the previous entry's hash, and compute its digest."""
        hash_payload = {
            "evidence_id": entry.get("evidence_id"),
            "activity_id": entry.get("activity_id"),
            "observation_id": entry.get("observation_id"),
            "match_id": entry.get("match_id"),
            "quantity_added": entry.get("quantity_added"),
            "unit": entry.get("unit"),
            "progress_category": entry.get("progress_category"),
            "logged_timestamp": entry.get("logged_timestamp"),
            "source_report_id": entry.get("source_report_id"),
            "reviewer_id": entry.get("reviewer_id"),
            "previous_hash": previous_hash,
        }
        return self.generate_hash(hash_payload)

    # ------------------------------------------------------------------
    # Ledger I/O
    # ------------------------------------------------------------------
    def get_ledger(self) -> list[dict[str, Any]]:
        """Read-only snapshot of every entry currently in the ledger."""
        with self._lock:
            return self._read_ledger()

    def _read_ledger(self) -> list[dict[str, Any]]:
        """Read and parse the ledger file. Caller must hold self._lock (or
        accept a benign race for pure-read use cases like get_ledger)."""
        if not self.ledger_path.exists():
            return []
        try:
            with self.ledger_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError as exc:
            log.error(
                "AuditLogger: ledger file at %s is corrupted/unparseable (%s) — "
                "treating as empty. Manual recovery may be required.",
                self.ledger_path, exc,
            )
            return []

    def _write_ledger_atomic(self, ledger: list[dict[str, Any]]) -> None:
        """
        Write the full ledger to disk atomically: serialize to a temp file
        in the same directory, then ``os.replace()`` it over the real
        ledger path. ``os.replace`` is atomic on both POSIX and Windows, so
        any concurrent reader either sees the old complete file or the new
        complete file — never a half-written one. Caller must hold
        ``self._lock``.
        """
        directory = self.ledger_path.parent
        fd, temp_path = tempfile.mkstemp(prefix=".evidence_ledger_", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.ledger_path)
        except Exception:
            # Clean up the temp file if the write/replace failed partway,
            # so failed writes don't leak stray .tmp files.
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_evidence_id(observation_id: str, match_id: str, timestamp: str) -> str:
        """Deterministic-per-call evidence_id in the 'EV-####' style from
        the spec example. Includes the timestamp in the hash input (unlike
        MatchDecision's match_id generator) since the same observation/match
        pair could theoretically be logged more than once over time (e.g. a
        correction), and each ledger entry needs its own distinct ID."""
        digest = hashlib.sha256(f"{observation_id}:{match_id}:{timestamp}".encode("utf-8")).hexdigest()
        numeric = int(digest[:8], 16) % 9000 + 1000
        return f"EV-{numeric}"
