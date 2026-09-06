#!/usr/bin/env python3
"""
synthetic_generator.py
=======================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 1 DELIVERABLE
--------------------
Self-contained synthetic benchmark data generator. Produces two paired JSON
datasets that emulate the two worlds PlanBridge must reconcile:

  1. data/activities.json  — Primavera P6 style L5/L6 schedule activities
                              (the "planning world" ground truth).
  2. data/dprs.json         — Messy field Daily Progress Reports (DPRs)
                              (the "execution world" raw evidence), each
                              pre-labelled with the activity it *should*
                              resolve to (or `null` if it legitimately has
                              no match), so downstream matching/NLP
                              components can be benchmarked against a known
                              answer key.

Design notes
------------
* No external dependencies — uses only the Python standard library, so this
  script can run before `pip install -r requirements.txt` has even
  completed, unblocking anyone who just wants the benchmark data.
* Deterministic by default (fixed RNG seed) so the generated dataset is
  reproducible across machines and CI runs; pass --seed to change it.
* Every "hard case" category called out in the PS26122 problem framing
  (unit mismatches, ambiguous free text, QA-gate language, pure noise) is
  deliberately over-represented relative to a real DPR stream, because the
  point of this dataset is to stress-test the matching engine, not to mimic
  raw production frequencies.

Usage
-----
    python synthetic_generator.py
    python synthetic_generator.py --num-activities 600 --num-dprs 150
    python synthetic_generator.py --seed 7 --out-dir data

See the bottom of this file / README instructions for verification steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("planbridge.synthetic_generator")


# --------------------------------------------------------------------------
# Domain reference data
# --------------------------------------------------------------------------
# These lists encode the OIL / Primavera P6 domain vocabulary the problem
# statement asks us to seed. Keeping them centralized makes it trivial to
# extend the generator to new disciplines or facilities later.

DISCIPLINES = ["Piping", "Civil", "Mechanical", "Electrical", "HSE"]

DISCIPLINE_CODE = {
    "Piping": "PIP",
    "Civil": "CIV",
    "Mechanical": "MEC",
    "Electrical": "ELE",
    "HSE": "HSE",
}

FACILITIES = [
    "CGS Duliajan",
    "CGS Moran",
    "OCS-4",
    "Trunkline ROW",
    "Booster Station Naharkatiya",
    "Terminal Duliajan",
    "Pump Station Barekuri",
]

FACILITY_SHORT = {
    "CGS Duliajan": "DULIAJAN",
    "CGS Moran": "MORAN",
    "OCS-4": "OCS4",
    "Trunkline ROW": "TL-ROW",
    "Booster Station Naharkatiya": "NAHARKATIYA",
    "Terminal Duliajan": "TERM-DULIAJAN",
    "Pump Station Barekuri": "BAREKURI",
}

QA_GATE_TYPES = ["NDT_RADIOGRAPHY", "HYDROTEST", "CIVIL_CUBE_TEST"]

# Each "activity template" defines: the discipline it belongs to, a name
# pattern (with {kp}/{section}/{facility} placeholders), the unit of
# measure, a realistic planned-quantity range, a typical duration range
# (days), and whether/what QA gate normally governs sign-off of that work
# type. This is the single source of truth that keeps every generated
# activity domain-plausible instead of randomly assembled nonsense.
ACTIVITY_TEMPLATES = [
    # -- Piping --------------------------------------------------------
    dict(discipline="Piping", name="HDD River Crossing Execution at {kp}",
         unit="M", qty_range=(80, 320), dur_range=(10, 25),
         qa_gate="NDT_RADIOGRAPHY", qa_prob=0.9, wbs_tag="PIP-HDD"),
    dict(discipline="Piping", name="Tie-in Welding at Chainage {kp}",
         unit="JOINTS", qty_range=(4, 24), dur_range=(2, 6),
         qa_gate="NDT_RADIOGRAPHY", qa_prob=0.95, wbs_tag="PIP-TIE"),
    dict(discipline="Piping", name="Trenching & Backfilling — Section {section}",
         unit="KM", qty_range=(0.3, 2.5), dur_range=(4, 14),
         qa_gate=None, qa_prob=0.05, wbs_tag="PIP-TRB"),
    dict(discipline="Piping", name="Pipe Stringing along ROW — Section {section}",
         unit="KM", qty_range=(0.5, 3.0), dur_range=(3, 10),
         qa_gate=None, qa_prob=0.0, wbs_tag="PIP-STR"),
    dict(discipline="Piping", name="Spool Fabrication for {facility} Tie-in Rack",
         unit="SPOOLS", qty_range=(5, 40), dur_range=(5, 18),
         qa_gate="NDT_RADIOGRAPHY", qa_prob=0.6, wbs_tag="PIP-SPL"),
    dict(discipline="Piping", name="Hydrotesting of Pipeline Section {section}",
         unit="KM", qty_range=(1.0, 4.0), dur_range=(3, 8),
         qa_gate="HYDROTEST", qa_prob=1.0, wbs_tag="PIP-HYD"),

    # -- Civil -----------------------------------------------------------
    dict(discipline="Civil", name="ROW Clearing — Section {section}",
         unit="KM", qty_range=(0.5, 3.0), dur_range=(3, 12),
         qa_gate=None, qa_prob=0.0, wbs_tag="CIV-ROW"),
    dict(discipline="Civil", name="Valve Pit Excavation at {kp}",
         unit="PIT", qty_range=(1, 3), dur_range=(2, 6),
         qa_gate="CIVIL_CUBE_TEST", qa_prob=0.5, wbs_tag="CIV-VLP"),
    dict(discipline="Civil", name="Access Road Construction near {facility}",
         unit="KM", qty_range=(0.2, 1.5), dur_range=(5, 15),
         qa_gate="CIVIL_CUBE_TEST", qa_prob=0.4, wbs_tag="CIV-ACR"),
    dict(discipline="Civil", name="RCC Foundation Casting at {facility}",
         unit="TONNES", qty_range=(10, 60), dur_range=(4, 10),
         qa_gate="CIVIL_CUBE_TEST", qa_prob=0.95, wbs_tag="CIV-FND"),

    # -- Mechanical -------------------------------------------------------
    dict(discipline="Mechanical", name="CGS Manifold Setup at {facility}",
         unit="TONNES", qty_range=(5, 30), dur_range=(6, 20),
         qa_gate=None, qa_prob=0.2, wbs_tag="MEC-MAN"),
    dict(discipline="Mechanical", name="Skid-Mounted Equipment Installation at {facility}",
         unit="TONNES", qty_range=(8, 45), dur_range=(7, 21),
         qa_gate=None, qa_prob=0.15, wbs_tag="MEC-SKD"),
    dict(discipline="Mechanical", name="Pig Launcher/Receiver Fabrication at {facility}",
         unit="TONNES", qty_range=(3, 12), dur_range=(5, 15),
         qa_gate="NDT_RADIOGRAPHY", qa_prob=0.7, wbs_tag="MEC-PLR"),

    # -- Electrical -----------------------------------------------------
    dict(discipline="Electrical", name="Cathodic Protection (CP) Test Station Installation at {kp}",
         unit="JOINTS", qty_range=(1, 6), dur_range=(1, 4),
         qa_gate=None, qa_prob=0.1, wbs_tag="ELE-CP"),
    dict(discipline="Electrical", name="SCADA Instrumentation Cable Laying — Section {section}",
         unit="KM", qty_range=(0.3, 2.0), dur_range=(3, 9),
         qa_gate=None, qa_prob=0.0, wbs_tag="ELE-SCD"),
    dict(discipline="Electrical", name="Transformer & MCC Panel Installation at {facility}",
         unit="TONNES", qty_range=(2, 10), dur_range=(4, 12),
         qa_gate=None, qa_prob=0.1, wbs_tag="ELE-TRF"),

    # -- HSE --------------------------------------------------------------
    dict(discipline="HSE", name="HSE Compliance Audit at {facility}",
         unit="PIT", qty_range=(1, 1), dur_range=(1, 2),
         qa_gate=None, qa_prob=0.0, wbs_tag="HSE-AUD"),
    dict(discipline="HSE", name="Mock Emergency Drill near {kp}",
         unit="PIT", qty_range=(1, 1), dur_range=(1, 1),
         qa_gate=None, qa_prob=0.0, wbs_tag="HSE-DRL"),
]

# Report submitters — a small realistic pool of site roles/names so the DPR
# dataset reads like it came from an actual field team.
SUBMITTERS = [
    "Rakesh Sharma (Site Inspector)",
    "Biju Gogoi (Piping Supervisor)",
    "Anita Deka (QA/QC Engineer)",
    "Manoj Tiwari (Civil Foreman)",
    "Suresh Yadav (HSE Officer)",
    "Pranjal Bora (Junior Engineer)",
    "Kavita Rani (Project Coordinator)",
    "Deepak Saikia (Welding Inspector)",
    "Imran Ali (Mechanical Supervisor)",
    "Ritu Baruah (Site Engineer)",
    "Field Voice Log (unattended transcription)",
]

SOURCE_TYPES = ["FREE_TEXT", "SPREADSHEET", "VOICE_TRANSCRIPT"]

# Filler content for UNMATCHED_NOISE reports — genuine field chatter that
# should NOT resolve to any schedule activity. This is what should force
# the matching engine to correctly abstain rather than force a false match.
NOISE_SENTENCES = [
    "Constructed temporary mud pump pit near yard due to heavy rain.",
    "Site office generator refuelled; no construction activity today due to local bandh.",
    "Toolbox talk conducted for all contractor staff on monsoon safety precautions.",
    "Material delivery truck delayed at check post; unloading rescheduled for tomorrow.",
    "Visitor group from district administration inspected the ROW corridor.",
    "Temporary drainage channel dug around labour camp to prevent waterlogging.",
    "Weekly housekeeping and waste segregation drive carried out at laydown yard.",
    "Local villagers raised access dispute near chainage; work paused pending resolution.",
    "First-aid training session conducted for new joinees at site clinic.",
    "Fuel bowser arrived late; no productive hours recorded for the shift.",
    "Security fencing repaired after cattle intrusion near the yard boundary.",
    "Contractor submitted updated insurance documents to site admin office.",
]

AMBIGUOUS_PHRASES = [
    "Piping work progressing near {facility} inlet valve pit.",
    "Trenching activity ongoing somewhere along {section} corridor.",
    "Civil team working near {facility}; exact extent to be confirmed tomorrow.",
    "Welding crew active in the {facility} tie-in area, quantity not recorded.",
    "Excavation continuing close to {kp}; surveyor yet to confirm chainage.",
    "Cable laying reported near {facility} substation, section not specified.",
]


# --------------------------------------------------------------------------
# Data classes (schema definitions)
# --------------------------------------------------------------------------
@dataclass
class Activity:
    activity_id: str
    wbs_code: str
    activity_name: str
    discipline: str
    location_kp: str
    facility: str
    planned_quantity: float
    unit: str
    planned_start: str
    planned_finish: str
    baseline_duration_days: int
    requires_qa_gate: bool
    qa_gate_type: Optional[str]


@dataclass
class DPR:
    report_id: str
    source_type: str
    submitted_by: str
    submission_timestamp: str
    raw_content: str
    expected_activity_id: Optional[str]
    case_type: str


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------
def fmt_kp(km_value: float) -> str:
    """Format a chainage value as an OIL-style KP string, e.g. 'KP 24+600'."""
    whole_km = int(km_value)
    meters = int(round((km_value - whole_km) * 1000))
    return f"KP {whole_km}+{meters:03d}"


def random_date_in_range(rng: random.Random, start: date, end: date) -> date:
    """Pick a uniformly random calendar date between start and end inclusive."""
    span_days = (end - start).days
    if span_days <= 0:
        return start
    return start + timedelta(days=rng.randint(0, span_days))


def build_wbs_code(rng: random.Random, facility: str, wbs_tag: str) -> str:
    """
    Build a Primavera-style hierarchical WBS code, e.g.
    'OIL.DULIAJAN.ROW4.PIP-HDD'. The ROW segment number is randomized to
    simulate different right-of-way work packages within the same facility.
    """
    facility_short = FACILITY_SHORT[facility]
    row_segment = f"ROW{rng.randint(1, 9)}"
    return f"OIL.{facility_short}.{row_segment}.{wbs_tag}"


def round_qty(rng: random.Random, low: float, high: float, unit: str) -> float:
    """Generate a planned_quantity appropriate to the unit's usual precision."""
    if unit in ("KM",):
        return round(rng.uniform(low, high), 3)
    if unit in ("M",):
        return round(rng.uniform(low, high), 1)
    if unit in ("TONNES",):
        return round(rng.uniform(low, high), 2)
    # JOINTS, SPOOLS, PIT are naturally integer counts
    return float(rng.randint(int(low), int(high)))


# --------------------------------------------------------------------------
# Activity generation
# --------------------------------------------------------------------------
def generate_activities(rng: random.Random, count: int) -> list[Activity]:
    """
    Generate `count` synthetic Primavera P6 L5/L6 activities by repeatedly
    sampling from ACTIVITY_TEMPLATES and randomizing location/date/quantity
    fields. activity_id is guaranteed unique via a per-discipline running
    counter, mirroring how real P6 activity codes are sequential within a
    WBS branch (e.g. PIP-L5-024-003).
    """
    activities: list[Activity] = []
    discipline_counters = {d: 0 for d in DISCIPLINES}

    # Project-level schedule horizon for planned_start/planned_finish
    project_start = date(2026, 1, 5)
    project_end = date(2027, 6, 30)

    for _ in range(count):
        template = rng.choice(ACTIVITY_TEMPLATES)
        discipline = template["discipline"]
        facility = rng.choice(FACILITIES)

        # Chainage / section context used to fill the name template
        kp_value = round(rng.uniform(0.0, 48.0), 3)
        kp_str = fmt_kp(kp_value)
        section_str = f"{rng.randint(1, 9)}{rng.choice('ABCD')}"

        activity_name = template["name"].format(
            kp=kp_str, section=section_str, facility=facility
        )

        unit = template["unit"]
        low, high = template["qty_range"]
        planned_quantity = round_qty(rng, low, high, unit)

        # Sequential-ish activity_id: DISC-L5-<week-of-year>-<counter>
        discipline_counters[discipline] += 1
        seq = discipline_counters[discipline]
        week_tag = rng.randint(1, 52)
        activity_id = (
            f"{DISCIPLINE_CODE[discipline]}-L5-{week_tag:03d}-{seq:03d}"
        )

        wbs_code = build_wbs_code(rng, facility, template["wbs_tag"])

        planned_start = random_date_in_range(
            rng, project_start, project_end - timedelta(days=30)
        )
        duration_low, duration_high = template["dur_range"]
        baseline_duration_days = rng.randint(duration_low, duration_high)
        planned_finish = planned_start + timedelta(days=baseline_duration_days)

        requires_qa_gate = bool(rng.random() < template["qa_prob"]) and template["qa_gate"] is not None
        qa_gate_type = template["qa_gate"] if requires_qa_gate else None

        activities.append(
            Activity(
                activity_id=activity_id,
                wbs_code=wbs_code,
                activity_name=activity_name,
                discipline=discipline,
                location_kp=kp_str,
                facility=facility,
                planned_quantity=planned_quantity,
                unit=unit,
                planned_start=planned_start.isoformat(),
                planned_finish=planned_finish.isoformat(),
                baseline_duration_days=baseline_duration_days,
                requires_qa_gate=requires_qa_gate,
                qa_gate_type=qa_gate_type,
            )
        )

    # De-duplicate activity_id collisions (rare, since week_tag is random)
    # by appending a disambiguating suffix — keeps IDs guaranteed-unique
    # without biasing the overall distribution.
    seen: dict[str, int] = {}
    for act in activities:
        if act.activity_id in seen:
            seen[act.activity_id] += 1
            act.activity_id = f"{act.activity_id}-{seen[act.activity_id]}"
        else:
            seen[act.activity_id] = 0

    return activities


# --------------------------------------------------------------------------
# DPR generation
# --------------------------------------------------------------------------
def _easy_match_sentence(rng: random.Random, act: Activity) -> str:
    """A clean, high-signal report that should resolve unambiguously."""
    verbs = {
        "Piping": ["completed", "executed", "carried out"],
        "Civil": ["completed", "carried out", "finished"],
        "Mechanical": ["completed", "installed", "carried out"],
        "Electrical": ["completed", "installed", "carried out"],
        "HSE": ["completed", "conducted"],
    }
    verb = rng.choice(verbs.get(act.discipline, ["completed"]))
    if act.unit in ("JOINTS", "SPOOLS", "PIT"):
        qty_frac: float | int = max(1, int(round(act.planned_quantity * rng.uniform(0.3, 1.0))))
    else:
        qty_frac = round(act.planned_quantity * rng.uniform(0.3, 1.0), 2)
    templates = [
        f"{qty_frac} {act.unit.lower()} of {act.activity_name.lower()} {verb} today near {act.location_kp}.",
        f"{act.activity_name} progress: {qty_frac} {act.unit} {verb} at {act.facility}, {act.location_kp}.",
        f"Today's update — {act.activity_name.lower()} {verb}, {qty_frac} {act.unit} logged at {act.location_kp}.",
    ]
    return rng.choice(templates)


def _unit_mismatch_sentence(rng: random.Random, act: Activity) -> str:
    """
    A report expressed in a different (but convertible) unit than the
    schedule's planned unit — e.g. schedule in KM, DPR in meters. Exercises
    the middleware's unit-normalization logic before matching.
    """
    if act.unit == "KM":
        meters = round(act.planned_quantity * rng.uniform(0.2, 0.8) * 1000)
        return (
            f"{meters} meters trenching completed on Section near {act.facility}."
            if "Trenching" in act.activity_name
            else f"{meters} meters of work completed near {act.location_kp} on {act.activity_name.split(' — ')[0].lower()}."
        )
    if act.unit == "M":
        km_val = round((act.planned_quantity * rng.uniform(0.2, 0.8)) / 1000, 4)
        return f"{km_val} km of HDD drilling progressed today at {act.location_kp}."
    if act.unit == "TONNES":
        kg_val = int(act.planned_quantity * rng.uniform(0.2, 0.8) * 1000)
        return f"Approx {kg_val} kg of structural steel erected at {act.facility} today."
    # Fallback for count-based units (JOINTS/SPOOLS/PIT) — express as "pairs"
    # or a loosely-worded quantity to still create a normalization challenge.
    approx = max(1, int(act.planned_quantity * rng.uniform(0.3, 0.9)))
    return f"About a dozen tie-in joints welded near {act.location_kp}, roughly {approx} nos completed."


def _qa_clearance_sentence(rng: random.Random, act: Activity) -> str:
    """QA/QC sign-off language referencing the activity's gate type."""
    if act.unit in ("JOINTS", "SPOOLS", "PIT"):
        qty_frac: float | int = max(1, int(round(act.planned_quantity * rng.uniform(0.5, 1.0))))
    else:
        qty_frac = round(act.planned_quantity * rng.uniform(0.5, 1.0), 1)
    gate_text = {
        "NDT_RADIOGRAPHY": f"NDT Radiography passed for {qty_frac} {act.unit.lower()} {act.discipline.lower()} section at {act.location_kp}.",
        "HYDROTEST": f"Hydrotest cleared for {qty_frac} {act.unit} pipeline section near {act.facility}, {act.location_kp}.",
        "CIVIL_CUBE_TEST": f"Concrete cube test results received — 28-day strength achieved for {act.facility} foundation near {act.location_kp}.",
    }
    return gate_text.get(
        act.qa_gate_type,
        f"QA clearance obtained for {act.activity_name.lower()} at {act.location_kp}.",
    )


def _ambiguous_sentence(rng: random.Random, act: Activity) -> str:
    phrase = rng.choice(AMBIGUOUS_PHRASES)
    return phrase.format(facility=act.facility, section=f"{rng.randint(1,9)}{rng.choice('ABCD')}", kp=act.location_kp)


def _timestamp_for(rng: random.Random, act: Activity) -> str:
    """Pick a submission timestamp within the activity's planned window."""
    start = date.fromisoformat(act.planned_start)
    finish = date.fromisoformat(act.planned_finish)
    report_date = random_date_in_range(rng, start, max(finish, start))
    hour = rng.randint(7, 19)
    minute = rng.choice([0, 15, 30, 45])
    dt = datetime(
        report_date.year, report_date.month, report_date.day, hour, minute,
        tzinfo=timezone.utc,
    )
    return dt.isoformat()


def generate_dprs(rng: random.Random, activities: list[Activity], count: int) -> list[DPR]:
    """
    Generate `count` synthetic DPRs distributed across the five benchmark
    case types called out in the problem framing. Distribution is weighted
    toward EASY_MATCH (since that's the volume case in production) while
    still guaranteeing a healthy population of every hard-case category for
    benchmarking.
    """
    case_weights = {
        "EASY_MATCH": 0.35,
        "AMBIGUOUS_MEDIUM": 0.20,
        "UNIT_MISMATCH": 0.20,
        "QA_CLEARANCE": 0.15,
        "UNMATCHED_NOISE": 0.10,
    }
    case_types = list(case_weights.keys())
    weights = list(case_weights.values())

    qa_eligible_activities = [a for a in activities if a.requires_qa_gate]
    if not qa_eligible_activities:
        log.warning(
            "No QA-gated activities were generated; QA_CLEARANCE reports "
            "will fall back to sampling from all activities."
        )

    dprs: list[DPR] = []
    report_date_counters: dict[str, int] = {}

    for i in range(count):
        case_type = rng.choices(case_types, weights=weights, k=1)[0]
        source_type = rng.choices(
            SOURCE_TYPES, weights=[0.55, 0.25, 0.20], k=1
        )[0]
        submitted_by = rng.choice(SUBMITTERS)

        if case_type == "EASY_MATCH":
            act = rng.choice(activities)
            raw_content = _easy_match_sentence(rng, act)
            expected_id = act.activity_id
            timestamp_source_act = act

        elif case_type == "UNIT_MISMATCH":
            act = rng.choice(activities)
            raw_content = _unit_mismatch_sentence(rng, act)
            expected_id = act.activity_id
            timestamp_source_act = act

        elif case_type == "QA_CLEARANCE":
            pool = qa_eligible_activities or activities
            act = rng.choice(pool)
            raw_content = _qa_clearance_sentence(rng, act)
            expected_id = act.activity_id
            timestamp_source_act = act

        elif case_type == "AMBIGUOUS_MEDIUM":
            act = rng.choice(activities)
            raw_content = _ambiguous_sentence(rng, act)
            # Ground-truth label still points at the best-fit activity so
            # the benchmark can score "did the model correctly flag this as
            # low-confidence / route to review" vs "did it guess right" —
            # but the sentence itself deliberately under-specifies chainage
            # or quantity so an over-eager exact-match engine will struggle.
            expected_id = act.activity_id
            timestamp_source_act = act

        else:  # UNMATCHED_NOISE
            raw_content = rng.choice(NOISE_SENTENCES)
            expected_id = None
            timestamp_source_act = rng.choice(activities)  # just for a plausible date

        timestamp = _timestamp_for(rng, timestamp_source_act)
        report_date_key = timestamp[:10]
        report_date_counters[report_date_key] = report_date_counters.get(report_date_key, 0) + 1
        seq = report_date_counters[report_date_key]
        report_id = f"DPR-{report_date_key}-{seq:03d}"

        dprs.append(
            DPR(
                report_id=report_id,
                source_type=source_type,
                submitted_by=submitted_by,
                submission_timestamp=timestamp,
                raw_content=raw_content,
                expected_activity_id=expected_id,
                case_type=case_type,
            )
        )

    return dprs


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_activities(activities: list[Activity]) -> None:
    ids = [a.activity_id for a in activities]
    assert len(ids) == len(set(ids)), "Duplicate activity_id detected after de-duplication pass."
    for a in activities:
        assert a.planned_quantity > 0, f"Non-positive planned_quantity on {a.activity_id}"
        assert date.fromisoformat(a.planned_start) < date.fromisoformat(a.planned_finish), (
            f"planned_start not before planned_finish on {a.activity_id}"
        )
        if a.requires_qa_gate:
            assert a.qa_gate_type in QA_GATE_TYPES, f"Missing/invalid qa_gate_type on {a.activity_id}"
        else:
            assert a.qa_gate_type is None, f"qa_gate_type set despite requires_qa_gate=False on {a.activity_id}"


def validate_dprs(dprs: list[DPR], activity_ids: set[str]) -> None:
    ids = [d.report_id for d in dprs]
    assert len(ids) == len(set(ids)), "Duplicate report_id detected."
    for d in dprs:
        if d.expected_activity_id is not None:
            assert d.expected_activity_id in activity_ids, (
                f"{d.report_id} references unknown activity_id {d.expected_activity_id}"
            )
        if d.case_type == "UNMATCHED_NOISE":
            assert d.expected_activity_id is None, f"{d.report_id} is UNMATCHED_NOISE but has an expected match."
        assert d.raw_content.strip() != "", f"{d.report_id} has empty raw_content."


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate PlanBridge synthetic benchmark data (activities.json + dprs.json)."
    )
    parser.add_argument("--num-activities", type=int, default=520,
                         help="Number of P6 activities to generate (default: 520, spec minimum 500).")
    parser.add_argument("--num-dprs", type=int, default=120,
                         help="Number of DPR reports to generate (default: 120, spec minimum 100).")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed for reproducible output (default: 42).")
    parser.add_argument("--out-dir", type=str, default="data",
                         help="Output directory for the generated JSON files (default: ./data).")
    args = parser.parse_args()

    if args.num_activities < 500:
        log.warning("--num-activities=%d is below the PS26122 spec minimum of 500.", args.num_activities)
    if args.num_dprs < 100:
        log.warning("--num-dprs=%d is below the PS26122 spec minimum of 100.", args.num_dprs)

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Could not create output directory '%s': %s", out_dir, exc)
        return 1

    log.info("Generating %d synthetic activities (seed=%d)...", args.num_activities, args.seed)
    activities = generate_activities(rng, args.num_activities)

    log.info("Validating activities dataset...")
    try:
        validate_activities(activities)
    except AssertionError as exc:
        log.error("Activity validation failed: %s", exc)
        return 1

    log.info("Generating %d synthetic DPRs (seed=%d)...", args.num_dprs, args.seed)
    dprs = generate_dprs(rng, activities, args.num_dprs)

    log.info("Validating DPR dataset...")
    try:
        validate_dprs(dprs, {a.activity_id for a in activities})
    except AssertionError as exc:
        log.error("DPR validation failed: %s", exc)
        return 1

    activities_path = out_dir / "activities.json"
    dprs_path = out_dir / "dprs.json"

    try:
        with activities_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in activities], f, indent=2, ensure_ascii=False)
        with dprs_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in dprs], f, indent=2, ensure_ascii=False)
    except OSError as exc:
        log.error("Failed to write output JSON files: %s", exc)
        return 1

    # -- Summary report ----------------------------------------------------
    disc_counts: dict[str, int] = {}
    for a in activities:
        disc_counts[a.discipline] = disc_counts.get(a.discipline, 0) + 1

    case_counts: dict[str, int] = {}
    for d in dprs:
        case_counts[d.case_type] = case_counts.get(d.case_type, 0) + 1

    log.info("Wrote %d activities -> %s", len(activities), activities_path.resolve())
    for disc, n in sorted(disc_counts.items()):
        log.info("    %-12s %d", disc, n)

    log.info("Wrote %d DPRs -> %s", len(dprs), dprs_path.resolve())
    for case, n in sorted(case_counts.items()):
        log.info("    %-18s %d", case, n)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
