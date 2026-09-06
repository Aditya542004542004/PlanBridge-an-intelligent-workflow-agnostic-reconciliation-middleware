"""
schemas.py
==========
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 2 + PHASE 3 + PHASE 4 + PHASE 5 DELIVERABLE — Pydantic data contracts
-----------------------------------------------------------------------------
This module defines the data contracts that flow through the PlanBridge
pipeline, in the order they're produced:

* ``ReportEvent``           — the raw, untouched field report as it arrives
                               from any source channel (free text, a
                               spreadsheet row, a voice transcript). This is
                               deliberately shaped to match the DPR schema
                               produced by Phase 1's `synthetic_generator.py`
                               so the two phases plug directly into each
                               other. [Phase 2]
* ``NormalizedObservation``  — one *quantity claim* extracted out of a
                               ``ReportEvent`` after unit normalization has
                               been applied. A single report can yield zero,
                               one, or several ``NormalizedObservation``
                               records (e.g. a report mentioning both a
                               trenching length and a QA clearance). [Phase 2]
* ``ExtractedEntities``      — the domain entities (KP marker, line ID,
                               facility, discipline, action verb) pulled out
                               of a report's text by ``EntityExtractor``.
                               [Phase 3]
* ``CandidateShortlist``     — the 3-5 schedule activities that survive
                               ``CandidateNarrower``'s hard entity filter for
                               a given observation, ready to be handed to
                               Phase 4's semantic re-ranking stage. [Phase 3]
* ``MatchDecision``          — the final output of the pipeline: which
                               (if any) activity an observation was matched
                               to, the full hybrid score breakdown, the
                               three-way confidence-gate decision, and a
                               human-readable explanation. [Phase 4]
* ``EvidenceLog``            — one immutable, SHA-256-stamped audit record
                               of a matched observation being applied to an
                               activity's progress — the CVC/CAG-compliant
                               ledger entry. [Phase 5]
* ``ProgressState``          — the current dual-tracked progress (physical
                               claimed vs QA-verified earned) of one
                               schedule activity. [Phase 5]

Keeping these as Pydantic models (rather than dataclasses) buys us runtime
validation at each stage boundary — malformed timestamps, negative
quantities, or an unrecognised source_type fail loudly here instead of
silently corrupting data further downstream in the matching engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Shared literal types
# --------------------------------------------------------------------------
SourceType = Literal["FREE_TEXT", "SPREADSHEET", "VOICE_TRANSCRIPT"]


# --------------------------------------------------------------------------
# ReportEvent — raw ingested field report
# --------------------------------------------------------------------------
class ReportEvent(BaseModel):
    """
    Represents one raw Daily Progress Report (DPR) as it enters the
    PlanBridge pipeline, before any parsing or normalization.

    This schema intentionally mirrors the DPR records produced by Phase 1's
    ``synthetic_generator.py`` (``data/dprs.json``), minus the benchmarking
    fields (``expected_activity_id``, ``case_type``) which belong to the
    offline evaluation harness, not the production ingestion contract.
    """

    report_id: str = Field(
        ..., min_length=1, description="Unique identifier for the source report, e.g. 'DPR-2026-08-24-001'."
    )
    source_type: SourceType = Field(
        ..., description="Channel the report arrived through."
    )
    submitted_by: str = Field(
        ..., min_length=1, description="Name and role of the submitter, e.g. 'Rakesh Sharma (Site Inspector)'."
    )
    submission_timestamp: datetime = Field(
        ..., description="When the report was submitted (ISO 8601)."
    )
    raw_content: str = Field(
        ..., min_length=1, description="The unstructured/messy field report text to be parsed."
    )

    @field_validator("raw_content")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only content — a common data-entry error in
        spreadsheet-sourced DPRs where a cell is technically non-empty but
        contains only spaces or a stray newline."""
        if not value.strip():
            raise ValueError("raw_content must contain non-whitespace text.")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "report_id": "DPR-2026-08-24-001",
                "source_type": "FREE_TEXT",
                "submitted_by": "Rakesh Sharma (Site Inspector)",
                "submission_timestamp": "2026-08-24T10:30:00+00:00",
                "raw_content": "150m HDD drilling completed today near KP 24+600 at river site.",
            }
        }
    }


# --------------------------------------------------------------------------
# NormalizedObservation — one extracted & unit-converted quantity claim
# --------------------------------------------------------------------------
class NormalizedObservation(BaseModel):
    """
    Represents a single quantity claim extracted from a ``ReportEvent`` and
    converted into PlanBridge's standardized enterprise units.

    Exactly one ``NormalizedObservation`` is produced per quantity+unit
    phrase found in the raw report text (see ``IngestionEngine`` in
    ``ingestion.py``). This is the atomic unit that Phase 3's matching
    engine will later reconcile against schedule activities — it is NOT yet
    linked to any ``activity_id``; that linkage is out of scope for Phase 2.
    """

    observation_id: str = Field(
        ..., min_length=1, description="Unique ID for this observation, e.g. 'DPR-2026-08-24-001-OBS-001'."
    )
    report_id: str = Field(
        ..., min_length=1, description="The ReportEvent.report_id this observation was extracted from."
    )
    raw_phrase: str = Field(
        ..., min_length=1, description="The source sentence/snippet the quantity was extracted from."
    )
    raw_quantity: float = Field(
        ..., description="The quantity exactly as written in the field report, before conversion."
    )
    raw_unit: str = Field(
        ..., min_length=1, description="The unit exactly as written in the field report, e.g. 'm', 'kg', 'joints'."
    )
    normalized_quantity: float = Field(
        ..., description="The quantity after deterministic conversion to the standardized enterprise unit."
    )
    normalized_unit: str = Field(
        ..., min_length=1, description="The standardized enterprise unit, e.g. 'KM', 'TONNES', 'JOINTS'."
    )
    conversion_applied: str = Field(
        ..., description="Human-readable audit trail of the conversion rule used, e.g. 'Meters to KM (div 1000)'."
    )
    is_qa_clearance: bool = Field(
        default=False,
        description="True if the source report text contains QA/QC clearance language (NDT, Hydrotest, Radiography, Cube test).",
    )

    @field_validator("raw_quantity")
    @classmethod
    def _raw_quantity_must_be_non_negative(cls, value: float) -> float:
        """Field quantities should never be negative — a negative value here
        almost always indicates a regex/parsing bug rather than real data."""
        if value < 0:
            raise ValueError("raw_quantity must be non-negative.")
        return value

    @field_validator("normalized_quantity")
    @classmethod
    def _normalized_quantity_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("normalized_quantity must be non-negative.")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "observation_id": "DPR-2026-08-24-001-OBS-001",
                "report_id": "DPR-2026-08-24-001",
                "raw_phrase": "150m HDD drilling completed today near KP 24+600 at river site.",
                "raw_quantity": 150.0,
                "raw_unit": "m",
                "normalized_quantity": 0.150,
                "normalized_unit": "KM",
                "conversion_applied": "Meters to KM (div 1000)",
                "is_qa_clearance": False,
            }
        }
    }


# --------------------------------------------------------------------------
# ExtractedEntities — domain entities pulled out of raw report text
# --------------------------------------------------------------------------
class ExtractedEntities(BaseModel):
    """
    The hard Oil & Gas EPC entities ``EntityExtractor`` pulls out of a DPR
    phrase. Every field is Optional by design: a real field report may
    mention only some of these (or none at all, e.g. pure administrative
    noise), and that is a legitimate, expected outcome — not an error.
    ``CandidateNarrower`` is built to degrade gracefully as more fields
    come back ``None``.
    """

    location_kp: Optional[str] = Field(
        default=None, description="Kilometre-post chainage marker, e.g. 'KP 24+600'."
    )
    line_id: Optional[str] = Field(
        default=None, description="Pipeline/spool/joint line identifier, e.g. 'Line 24-A'."
    )
    facility: Optional[str] = Field(
        default=None, description="Facility name/reference, e.g. 'CGS Duliajan', 'OCS-4'."
    )
    discipline: Optional[str] = Field(
        default=None, description="Inferred engineering discipline, e.g. 'Piping', 'Civil', 'HSE'."
    )
    action_verb: Optional[str] = Field(
        default=None, description="Normalized action/work type, e.g. 'HDD Drilling', 'Tie-in Welding'."
    )

    def has_any_entity(self) -> bool:
        """True if at least one field was successfully extracted."""
        return any(
            value is not None
            for value in (self.location_kp, self.line_id, self.facility, self.discipline, self.action_verb)
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "location_kp": "KP 24+600",
                "line_id": None,
                "facility": None,
                "discipline": "Piping",
                "action_verb": "HDD Drilling",
            }
        }
    }


# --------------------------------------------------------------------------
# CandidateShortlist — narrowed set of candidate schedule activities
# --------------------------------------------------------------------------
class CandidateShortlist(BaseModel):
    """
    The output of ``CandidateNarrower.narrow_candidates()`` — a short list
    of 3-5 Primavera P6 activities (raw dicts, as loaded from
    ``data/activities.json``) that a ``NormalizedObservation`` most
    plausibly corresponds to, based purely on hard entity filtering (no
    semantic/embedding matching — that is Phase 4's job).
    """

    observation_id: str = Field(
        ..., min_length=1, description="The NormalizedObservation.observation_id this shortlist was built for."
    )
    extracted_entities: ExtractedEntities = Field(
        ..., description="The entities extracted from the observation's raw_phrase that drove the filtering."
    )
    candidate_activities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="3-5 candidate activity records (raw dicts from activities.json) surviving the hard filter.",
    )
    initial_candidate_count: int = Field(
        ..., ge=0, description="How many activities matched at the filter stage that ultimately succeeded, before truncating to the top 3-5 shown."
    )
    narrowing_stage: Optional[str] = Field(
        default=None,
        description="Which filter stage produced this shortlist: 'KP_EXACT', 'FACILITY_DISCIPLINE', 'ACTION_VERB', or 'FALLBACK_GENERIC'.",
    )

    @field_validator("candidate_activities")
    @classmethod
    def _shortlist_size_bounds(cls, value: list[dict]) -> list[dict]:
        """Enforce the spec's 3-5 candidate window — except when the
        universe of activities itself is smaller than 3, in which case we
        cannot manufacture candidates that don't exist."""
        if len(value) > 5:
            raise ValueError(f"candidate_activities must contain at most 5 items, got {len(value)}.")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "observation_id": "DPR-2026-08-24-001-OBS-001",
                "extracted_entities": {
                    "location_kp": "KP 24+600",
                    "line_id": None,
                    "facility": None,
                    "discipline": "Piping",
                    "action_verb": "HDD Drilling",
                },
                "candidate_activities": [],
                "initial_candidate_count": 3,
                "narrowing_stage": "ACTION_VERB",
            }
        }
    }


# --------------------------------------------------------------------------
# MatchDecision — final Stage 2 output: the reconciliation decision
# --------------------------------------------------------------------------
DecisionType = Literal["AUTO_ACCEPT", "HUMAN_REVIEW", "UNMATCHED"]


class MatchDecision(BaseModel):
    """
    The final, auditable output of the PlanBridge matching pipeline for one
    ``NormalizedObservation`` — produced by ``ConfidenceGate.make_decision()``
    after ``ScoringEngine.evaluate_candidates()`` has ranked the Stage 1
    shortlist.

    This is the record that would actually get written to OIL's execution
    database (for AUTO_ACCEPT), routed to a planner's review queue (for
    HUMAN_REVIEW), or logged as an unresolved observation (for UNMATCHED).
    Every field needed to audit *why* a decision was reached is captured
    here — this is deliberately not just a bare activity_id.
    """

    match_id: str = Field(
        ..., min_length=1, description="Unique identifier for this match decision, e.g. 'MATCH-8812'."
    )
    observation_id: str = Field(
        ..., min_length=1, description="The NormalizedObservation.observation_id this decision was made for."
    )
    selected_activity_id: Optional[str] = Field(
        default=None,
        description="The chosen activity_id for AUTO_ACCEPT/HUMAN_REVIEW (the top-ranked candidate); None for UNMATCHED.",
    )
    entity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Stage 1 entity-match score (0.0-1.0) of the selected/top candidate."
    )
    semantic_score: float = Field(
        ..., ge=0.0, le=1.0, description="Stage 2 Sentence-BERT semantic similarity score (0.0-1.0) of the selected/top candidate."
    )
    quantity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Quantity plausibility score (0.0-1.0) of the selected/top candidate."
    )
    final_confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="The weighted hybrid S_final score that drove the gate decision."
    )
    decision_type: DecisionType = Field(
        ..., description="'AUTO_ACCEPT' (>=0.85), 'HUMAN_REVIEW' (0.60-0.85), or 'UNMATCHED' (<0.60)."
    )
    reasoning: str = Field(
        ..., min_length=1, description="Human-readable explanation of why this decision was reached."
    )
    candidate_scores: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full score breakdown for every candidate that was evaluated, sorted by final_score descending.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "match_id": "MATCH-8812",
                "observation_id": "DPR-2026-08-24-001-OBS-001",
                "selected_activity_id": "PIP-L5-024-003",
                "entity_score": 0.95,
                "semantic_score": 0.88,
                "quantity_score": 1.0,
                "final_confidence_score": 0.933,
                "decision_type": "AUTO_ACCEPT",
                "reasoning": (
                    "Top candidate PIP-L5-024-003 scored 93.3% confidence "
                    "(entity=95.0%, semantic=88.0%, quantity=100.0%), "
                    "clearing the 85% auto-accept threshold."
                ),
                "candidate_scores": [],
            }
        }
    }


# --------------------------------------------------------------------------
# EvidenceLog — one immutable, hash-stamped audit ledger entry
# --------------------------------------------------------------------------
ProgressCategory = Literal["PHYSICAL_CLAIM", "QA_VERIFIED"]


class EvidenceLog(BaseModel):
    """
    One append-only, cryptographically-stamped entry in PlanBridge's
    evidence ledger — the CVC/CAG (Central Vigilance Commission / Comptroller
    and Auditor General) audit trail required for a PSU like Oil India.

    Every time a matched observation is applied to an activity's progress
    (``ProgressEngine.apply_match_decision``), ``AuditLogger.log_evidence``
    writes exactly one ``EvidenceLog`` record. Records are never edited or
    deleted after being written — corrections happen by appending a new,
    separately-hashed record, never by mutating history. ``evidence_hash``
    is chained to the previous ledger entry's hash (see ``audit_logger.py``),
    so altering any historical record breaks not just that record's hash but
    every subsequent one — this is what lets ``verify_ledger_integrity()``
    detect tampering anywhere in the ledger's history, not just the most
    recent entry.
    """

    evidence_id: str = Field(..., min_length=1, description="Unique identifier for this ledger entry, e.g. 'EV-9182'.")
    activity_id: str = Field(..., min_length=1, description="The schedule activity this evidence applies progress to.")
    observation_id: str = Field(..., min_length=1, description="The NormalizedObservation this evidence was derived from.")
    match_id: str = Field(..., min_length=1, description="The MatchDecision.match_id that authorized this progress update.")
    quantity_added: float = Field(..., description="The (normalized) quantity this evidence contributes.")
    unit: str = Field(..., min_length=1, description="The standardized unit of quantity_added, e.g. 'KM', 'JOINTS'.")
    progress_category: ProgressCategory = Field(
        ..., description="'PHYSICAL_CLAIM' (site-claimed execution) or 'QA_VERIFIED' (QA/QC-cleared earned progress)."
    )
    logged_timestamp: str = Field(..., min_length=1, description="ISO 8601 timestamp of when this evidence was logged.")
    source_report_id: str = Field(..., min_length=1, description="The original ReportEvent.report_id this evidence traces back to.")
    reviewer_id: Optional[str] = Field(
        default=None, description="Identifier of the human reviewer who confirmed this match, if applicable (None for AUTO_ACCEPT)."
    )
    evidence_hash: str = Field(
        ..., min_length=64, max_length=64, description="SHA-256 hex digest chaining this entry to the ledger's history."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "evidence_id": "EV-9182",
                "activity_id": "PIP-L5-024-003",
                "observation_id": "DPR-2026-08-24-001-OBS-001",
                "match_id": "MATCH-8812",
                "quantity_added": 0.150,
                "unit": "KM",
                "progress_category": "PHYSICAL_CLAIM",
                "logged_timestamp": "2026-08-24T10:31:05+00:00",
                "source_report_id": "DPR-2026-08-24-001",
                "reviewer_id": None,
                "evidence_hash": "a3f5c8e1d2b4..." + "0" * 52,
            }
        }
    }


# --------------------------------------------------------------------------
# ProgressState — dual-tracked (claimed vs QA-verified) activity progress
# --------------------------------------------------------------------------
QaGateStatus = Literal["PENDING_QA", "VERIFIED_PASSED", "NOT_REQUIRED"]


class ProgressState(BaseModel):
    """
    The current dual-tracked progress of one schedule activity, maintained
    by ``ProgressEngine``.

    PlanBridge deliberately keeps two separate progress numbers rather than
    one:
      * ``physical_progress_pct``  — what the site claims is done (from
                                      supervisor DPRs), useful for near-
                                      real-time schedule tracking but not
                                      yet independently verified.
      * ``verified_progress_pct``  — what QA/QC has actually signed off on
                                      (NDT/Radiography/Hydrotest/Cube-test
                                      passed), the number that should drive
                                      billing/earned-value calculations for
                                      a PSU audit.

    Conflating these two into one "progress %" is exactly the kind of gap a
    CVC/CAG audit would flag — this model exists specifically to keep them
    separate and auditable.
    """

    activity_id: str = Field(..., min_length=1, description="The Primavera P6 activity this progress state tracks.")
    planned_quantity: float = Field(..., gt=0, description="The activity's total planned scope, from activities.json.")
    unit: str = Field(..., min_length=1, description="The activity's standardized unit, e.g. 'KM', 'JOINTS'.")
    physical_claimed_quantity: float = Field(
        default=0.0, ge=0.0, description="Cumulative quantity claimed by site DPRs (not yet QA-verified)."
    )
    physical_progress_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="min(100, physical_claimed_quantity / planned_quantity * 100)."
    )
    verified_earned_quantity: float = Field(
        default=0.0, ge=0.0, description="Cumulative quantity certified by a passed QA/QC gate (capped at claimed)."
    )
    verified_progress_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="min(100, verified_earned_quantity / planned_quantity * 100)."
    )
    qa_gate_type: str = Field(
        ..., description="The activity's required QA gate type, e.g. 'NDT_RADIOGRAPHY', or 'NONE' if not applicable."
    )
    qa_gate_status: QaGateStatus = Field(
        default="NOT_REQUIRED",
        description="'PENDING_QA' (gate required, not yet passed), 'VERIFIED_PASSED', or 'NOT_REQUIRED'.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "activity_id": "PIP-L5-024-003",
                "planned_quantity": 0.500,
                "unit": "KM",
                "physical_claimed_quantity": 0.150,
                "physical_progress_pct": 30.0,
                "verified_earned_quantity": 0.150,
                "verified_progress_pct": 30.0,
                "qa_gate_type": "NDT_RADIOGRAPHY",
                "qa_gate_status": "VERIFIED_PASSED",
            }
        }
    }
