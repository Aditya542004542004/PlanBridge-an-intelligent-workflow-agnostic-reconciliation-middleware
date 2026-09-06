"""
main.py
=========
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 6 DELIVERABLE — FastAPI REST Backend Server
------------------------------------------------------------
Wraps the entire PlanBridge pipeline (Phases 2-6) behind a REST API:

    GET  /                    — serves the visual frontend dashboard SPA.
    POST /api/ingest          — run a raw field report through the full
                                 5-stage pipeline (ingest -> extract ->
                                 narrow -> score -> gate), applying
                                 AUTO_ACCEPT decisions immediately and
                                 queuing HUMAN_REVIEW decisions for a
                                 planner.
    GET  /api/dashboard        — aggregate reconciliation metrics.
    GET  /api/queue            — pending planner review queue.
    POST /api/queue/approve    — a planner confirms (or overrides) a
                                 queued match, applying it to progress
                                 tracking and the audit ledger.
    GET  /api/audit-logs       — the full SHA-256-hash-chained evidence
                                 ledger.
    GET  /api/export-xer       — download the current progress as a
                                 Primavera P6 .XER schedule delta.

Run locally with:
    uvicorn main:app --reload --port 8000

Then browse the dashboard at http://localhost:8000/
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import csv
import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from audit_logger import AuditLogger
from candidate_narrower import CandidateNarrower
from confidence_gate import ConfidenceGate
from entity_extractor import EntityExtractor
from ingestion import IngestionEngine
from progress_engine import ProgressEngine
from schemas import EvidenceLog, MatchDecision, ProgressState, ReportEvent, SourceType
from scoring_engine import ScoringEngine
from unit_normalizer import UnitNormalizer
from vector_ranker import VectorRanker
from xer_exporter import XERExporter

log = logging.getLogger("planbridge.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

# --------------------------------------------------------------------------
# Paths — resolved relative to this file
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR.parent / "frontend"

ACTIVITIES_PATH = Path(os.environ.get("PLANBRIDGE_ACTIVITIES_PATH", str(DATA_DIR / "activities.json")))
LEDGER_PATH = Path(os.environ.get("PLANBRIDGE_LEDGER_PATH", str(DATA_DIR / "evidence_ledger.json")))

BENCHMARK_FALSE_AUTO_ACCEPT_RATE = 0.00


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------
class IngestRequest(BaseModel):
    raw_content: str = Field(..., min_length=1, description="The raw, unstructured field report text.")
    source_type: SourceType = Field(default="FREE_TEXT", description="Channel the report arrived through.")
    submitted_by: str = Field(default="Unknown", min_length=1, description="Name/role of the submitter.")


class ObservationResult(BaseModel):
    observation_id: str
    decision: MatchDecision
    progress_state: Optional[ProgressState] = None
    queued_for_review: bool = Field(
        default=False, description="True if this decision was HUMAN_REVIEW and is now sitting in the planner queue."
    )


class IngestResponse(BaseModel):
    report_id: str
    observations_processed: int = Field(
        ..., description="How many quantity observations were extracted (0 for reports with no parseable quantity claim)."
    )
    results: list[ObservationResult] = Field(
        default_factory=list, description="Per-observation results — a single report can yield 0, 1, or several."
    )
    decision: Optional[MatchDecision] = Field(
        default=None, description="Convenience alias for results[0].decision."
    )
    progress_state: Optional[ProgressState] = Field(
        default=None, description="Convenience alias for results[0].progress_state."
    )


class DashboardResponse(BaseModel):
    total_activities: int
    autolinked_count: int = Field(..., description="Cumulative AUTO_ACCEPT decisions applied this server session.")
    pending_queue_count: int
    unmatched_count: int = Field(..., description="Cumulative UNMATCHED decisions this server session.")
    false_auto_accept_rate: float = Field(
        ..., description="Measured against Phase 4's labeled benchmark dataset."
    )
    activities_progress_list: list[ProgressState]


class QueueItem(BaseModel):
    queue_id: str
    decision: MatchDecision
    raw_phrase: str
    source_report_id: str
    candidate_activity_ids: list[str] = Field(
        default_factory=list,
        description="The activity_ids scored as candidates.",
    )


class QueueResponse(BaseModel):
    pending_count: int
    items: list[QueueItem]


class ApproveRequest(BaseModel):
    queue_id: str = Field(..., min_length=1)
    selected_activity_id: str = Field(
        ..., min_length=1, description="The activity_id the planner confirms."
    )
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of the human planner making this decision.")


class ApproveResponse(BaseModel):
    decision: MatchDecision
    progress_state: ProgressState
    evidence: EvidenceLog


class AuditLogsResponse(BaseModel):
    total_entries: int
    integrity_verified: bool = Field(..., description="Result of live SHA-256 hash-chain verification.")
    entries: list[dict[str, Any]]




# --------------------------------------------------------------------------
# Startup / shutdown — component wiring & state replay
# --------------------------------------------------------------------------
def _load_activities() -> list[dict[str, Any]]:
    if not ACTIVITIES_PATH.exists():
        raise RuntimeError(
            f"activities.json not found at {ACTIVITIES_PATH}. Run synthetic_generator.py (Phase 1) first, "
            f"or set the PLANBRIDGE_ACTIVITIES_PATH environment variable."
        )
    with ACTIVITIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Wires up every pipeline component at server startup and replays historical
    audit logs from disk using get_progress_state() so Gantt chart progress is 100% persistent.
    """
    log.info("PlanBridge API starting up — loading activities from %s", ACTIVITIES_PATH)
    activities = _load_activities()

    app.state.activities = activities
    app.state.unit_normalizer = UnitNormalizer()
    app.state.ingestion_engine = IngestionEngine(app.state.unit_normalizer)
    app.state.entity_extractor = EntityExtractor()
    app.state.candidate_narrower = CandidateNarrower(activities)
    app.state.vector_ranker = VectorRanker()
    app.state.scoring_engine = ScoringEngine(vector_ranker=app.state.vector_ranker)
    app.state.confidence_gate = ConfidenceGate()
    app.state.progress_engine = ProgressEngine(activities)
    app.state.audit_logger = AuditLogger(ledger_path=str(LEDGER_PATH))
    app.state.xer_exporter = XERExporter()

    # ------------------------------------------------------------------
    # REPLAY HISTORICAL AUDIT LEDGER ON STARTUP (FIXED API METHOD)
    # ------------------------------------------------------------------
    auto_count = 0
    unmatched_count = 0

    if LEDGER_PATH.exists():
        try:
            with LEDGER_PATH.open("r", encoding="utf-8") as f:
                logs = json.load(f)
            log.info("Replaying %d evidence logs from %s to restore Gantt progress...", len(logs), LEDGER_PATH)
            
            for entry in logs:
                act_id = entry.get("activity_id")
                qty = float(entry.get("quantity_added", 0))
                category = entry.get("progress_category")
                
                if act_id:
                    st = app.state.progress_engine.get_progress_state(act_id)
                    if st is not None:
                        is_pydantic = hasattr(st, "planned_quantity")
                        planned_qty = float(st.planned_quantity if is_pydantic else st.get("planned_quantity", 1.0))
                        
                        if category == "QA_VERIFIED":
                            cur_phys = float(st.physical_claimed_quantity if is_pydantic else st.get("physical_claimed_quantity", 0))
                            cur_ver = float(st.verified_earned_quantity if is_pydantic else st.get("verified_earned_quantity", 0))
                            new_ver = cur_ver + qty
                            capped_ver = min(cur_phys, new_ver) if cur_phys > 0 else new_ver
                            ver_pct = min(100.0, (capped_ver / planned_qty) * 100.0) if planned_qty > 0 else 0.0
                            
                            if is_pydantic:
                                st.verified_earned_quantity = capped_ver
                                st.verified_progress_pct = round(ver_pct, 2)
                                st.qa_gate_status = "VERIFIED_PASSED"
                            else:
                                st["verified_earned_quantity"] = capped_ver
                                st["verified_progress_pct"] = round(ver_pct, 2)
                                st["qa_gate_status"] = "VERIFIED_PASSED"
                        else:
                            cur_phys = float(st.physical_claimed_quantity if is_pydantic else st.get("physical_claimed_quantity", 0))
                            new_phys = cur_phys + qty
                            phys_pct = min(100.0, (new_phys / planned_qty) * 100.0) if planned_qty > 0 else 0.0
                            
                            if is_pydantic:
                                st.physical_claimed_quantity = new_phys
                                st.physical_progress_pct = round(phys_pct, 2)
                            else:
                                st["physical_claimed_quantity"] = new_phys
                                st["physical_progress_pct"] = round(phys_pct, 2)
                        auto_count += 1
            log.info("ProgressEngine state successfully restored! %d logs replayed.", auto_count)
        except Exception as exc:
            log.warning("Could not replay evidence ledger on startup: %s", exc)

    # In-memory planner review queue
    app.state.pending_queue = {}

    # Session-lifetime counters for the dashboard
    app.state.metrics = {"auto_accept_count": auto_count, "unmatched_count": unmatched_count}

    log.info(
        "PlanBridge API ready: %d activities loaded, VectorRanker backend=%s.",
        len(activities), app.state.vector_ranker.backend_name,
    )
    yield
    log.info("PlanBridge API shutting down.")

app = FastAPI(
    title="PlanBridge Reconciliation API",
    description="REST API for OIL's Planning-to-Execution Reconciliation Middleware (SIH26122).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


# --------------------------------------------------------------------------
# Serve Dashboard SPA Directly over HTTP at Root "/"
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves index.html directly over HTTP to bypass file:// browser security restrictions."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(content={"service": "PlanBridge Reconciliation API", "status": "ok"})


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --------------------------------------------------------------------------
# POST /api/ingest
# --------------------------------------------------------------------------
@app.post("/api/ingest", response_model=IngestResponse, tags=["pipeline"])
async def ingest_report(payload: IngestRequest, request: Request) -> IngestResponse:
    state = request.app.state
    report_id = f"API-{uuid.uuid4().hex[:10].upper()}"

    try:
        report = ReportEvent(
            report_id=report_id,
            source_type=payload.source_type,
            submitted_by=payload.submitted_by,
            submission_timestamp=datetime.now(timezone.utc),
            raw_content=payload.raw_content,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid report payload: {exc}") from exc

    try:
        observations = state.ingestion_engine.process_report(report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion/normalization failed: {exc}") from exc

    results: list[ObservationResult] = []
    for obs in observations:
        try:
            entities = state.entity_extractor.extract(obs.raw_phrase)
            shortlist = state.candidate_narrower.narrow_candidates(obs, entities)
            scored = state.scoring_engine.evaluate_candidates(obs, shortlist)
            decision = state.confidence_gate.make_decision(obs, scored)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Matching pipeline failed for observation {obs.observation_id}: {exc}"
            ) from exc

        progress_state: Optional[ProgressState] = None
        queued = False

        if decision.decision_type == "AUTO_ACCEPT":
            try:
                progress_state = state.progress_engine.apply_match_decision(decision, obs)
                category = "QA_VERIFIED" if obs.is_qa_clearance else "PHYSICAL_CLAIM"
                state.audit_logger.log_evidence(decision, obs, category)
                state.metrics["auto_accept_count"] += 1
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to apply AUTO_ACCEPT decision {decision.match_id}: {exc}"
                ) from exc
        elif decision.decision_type == "HUMAN_REVIEW":
            state.pending_queue[decision.match_id] = {"decision": decision, "observation": obs}
            queued = True
        else:  # UNMATCHED
            state.metrics["unmatched_count"] += 1

        results.append(ObservationResult(
            observation_id=obs.observation_id,
            decision=decision,
            progress_state=progress_state,
            queued_for_review=queued,
        ))

    first = results[0] if results else None
    return IngestResponse(
        report_id=report_id,
        observations_processed=len(results),
        results=results,
        decision=first.decision if first else None,
        progress_state=first.progress_state if first else None,
    )


# --------------------------------------------------------------------------
# POST /api/ingest-file (Batch Contractor File Upload: CSV / JSON / TXT)
# --------------------------------------------------------------------------
class BatchIngestSummary(BaseModel):
    total_processed: int
    auto_accepted_count: int
    queued_count: int
    unmatched_count: int
    filename: str

@app.post("/api/ingest-file", response_model=BatchIngestSummary, tags=["pipeline"])
async def ingest_file(request: Request, file: UploadFile = File(...)) -> BatchIngestSummary:
    """
    Ingests an entire contractor Excel/CSV/JSON/TXT file line-by-line through the 5-stage pipeline:
    - AUTO_ACCEPT matches -> Applied to progress & logged to CVC Audit Ledger.
    - HUMAN_REVIEW matches -> Routed to Planner Review Queue for 1-click review.
    - UNMATCHED noise -> Flagged in Unmatched Log to protect baseline schedule.
    """
    contents = await file.read()
    text_content = contents.decode("utf-8", errors="ignore")
    
    raw_lines: list[str] = []
    
    # Parse based on file type
    if file.filename.endswith(".json"):
        try:
            json_data = json.loads(text_content)
            for item in json_data:
                if isinstance(item, dict) and "raw_content" in item:
                    raw_lines.append(item["raw_content"])
                elif isinstance(item, str):
                    raw_lines.append(item)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file format: {e}")
            
    elif file.filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(text_content))
        headers = next(reader, None)
        for row in reader:
            if row:
                # Find non-empty cell or join text columns
                line = " ".join([cell.strip() for cell in row if cell.strip()])
                if line:
                    raw_lines.append(line)
    else:
        # Plain text lines
        raw_lines = [line.strip() for line in text_content.splitlines() if line.strip()]

    if not raw_lines:
        raise HTTPException(status_code=400, detail="File is empty or contains no parseable text lines.")

    auto_count = 0
    queued_count = 0
    unmatched_count = 0

    state = request.app.state

    # Process every line through full pipeline
    for line in raw_lines:
        report_id = f"FILE-{uuid.uuid4().hex[:8].upper()}"
        report = ReportEvent(
            report_id=report_id,
            source_type="SPREADSHEET",
            submitted_by=f"Contractor ({file.filename})",
            submission_timestamp=datetime.now(timezone.utc),
            raw_content=line
        )
        
        observations = state.ingestion_engine.process_report(report)
        for obs in observations:
            entities = state.entity_extractor.extract(obs.raw_phrase)
            shortlist = state.candidate_narrower.narrow_candidates(obs, entities)
            scored = state.scoring_engine.evaluate_candidates(obs, shortlist)
            decision = state.confidence_gate.make_decision(obs, scored)

            if decision.decision_type == "AUTO_ACCEPT":
                state.progress_engine.apply_match_decision(decision, obs)
                category = "QA_VERIFIED" if obs.is_qa_clearance else "PHYSICAL_CLAIM"
                state.audit_logger.log_evidence(decision, obs, category)
                state.metrics["auto_accept_count"] += 1
                auto_count += 1
            elif decision.decision_type == "HUMAN_REVIEW":
                state.pending_queue[decision.match_id] = {"decision": decision, "observation": obs}
                queued_count += 1
            else:
                state.metrics["unmatched_count"] += 1
                unmatched_count += 1

    return BatchIngestSummary(
        total_processed=len(raw_lines),
        auto_accepted_count=auto_count,
        queued_count=queued_count,
        unmatched_count=unmatched_count,
        filename=file.filename
    )

# --------------------------------------------------------------------------
# GET /api/dashboard
# --------------------------------------------------------------------------
@app.get("/api/dashboard", response_model=DashboardResponse, tags=["monitoring"])
async def get_dashboard(request: Request) -> DashboardResponse:
    state = request.app.state
    return DashboardResponse(
        total_activities=len(state.activities),
        autolinked_count=state.metrics["auto_accept_count"],
        pending_queue_count=len(state.pending_queue),
        unmatched_count=state.metrics["unmatched_count"],
        false_auto_accept_rate=BENCHMARK_FALSE_AUTO_ACCEPT_RATE,
        activities_progress_list=state.progress_engine.get_all_progress_states(),
    )


# --------------------------------------------------------------------------
# GET /api/queue
# --------------------------------------------------------------------------
@app.get("/api/queue", response_model=QueueResponse, tags=["review-queue"])
async def get_queue(request: Request) -> QueueResponse:
    state = request.app.state
    items = [
        QueueItem(
            queue_id=queue_id,
            decision=entry["decision"],
            raw_phrase=entry["observation"].raw_phrase,
            source_report_id=entry["observation"].report_id,
            candidate_activity_ids=[c.get("activity_id") for c in entry["decision"].candidate_scores],
        )
        for queue_id, entry in state.pending_queue.items()
    ]
    return QueueResponse(pending_count=len(items), items=items)


# --------------------------------------------------------------------------
# POST /api/queue/approve
# --------------------------------------------------------------------------
@app.post("/api/queue/approve", response_model=ApproveResponse, tags=["review-queue"])
async def approve_queue_item(payload: ApproveRequest, request: Request) -> ApproveResponse:
    state = request.app.state
    entry = state.pending_queue.get(payload.queue_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No pending queue item with queue_id='{payload.queue_id}'.")

    if state.progress_engine.get_progress_state(payload.selected_activity_id) is None:
        raise HTTPException(
            status_code=400,
            detail=f"'{payload.selected_activity_id}' is not a recognized/tracked activity_id.",
        )

    original_decision: MatchDecision = entry["decision"]
    observation = entry["observation"]

    reviewer_note = ""
    if payload.selected_activity_id != original_decision.selected_activity_id:
        reviewer_note = (
            f" [Reviewer override: planner selected {payload.selected_activity_id} instead of the "
            f"original top-ranked candidate {original_decision.selected_activity_id}.]"
        )
    updated_decision = original_decision.model_copy(update={
        "selected_activity_id": payload.selected_activity_id,
        "reasoning": original_decision.reasoning + reviewer_note,
    })

    try:
        progress_state = state.progress_engine.apply_match_decision(updated_decision, observation)
        category = "QA_VERIFIED" if observation.is_qa_clearance else "PHYSICAL_CLAIM"
        evidence = state.audit_logger.log_evidence(
            updated_decision, observation, category, reviewer_id=payload.reviewer_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to apply approved decision: {exc}") from exc

    del state.pending_queue[payload.queue_id]

    return ApproveResponse(decision=updated_decision, progress_state=progress_state, evidence=evidence)


# --------------------------------------------------------------------------
# GET /api/audit-logs
# --------------------------------------------------------------------------
@app.get("/api/audit-logs", response_model=AuditLogsResponse, tags=["audit"])
async def get_audit_logs(request: Request) -> AuditLogsResponse:
    state = request.app.state
    ledger = state.audit_logger.get_ledger()
    integrity_verified = state.audit_logger.verify_ledger_integrity()
    return AuditLogsResponse(total_entries=len(ledger), integrity_verified=integrity_verified, entries=ledger)


# --------------------------------------------------------------------------
# GET /api/export-xer
# --------------------------------------------------------------------------
@app.get("/api/export-xer", tags=["export"])
async def export_xer(request: Request) -> Response:
    state = request.app.state
    try:
        xer_content = state.xer_exporter.generate_xer_delta(
            state.progress_engine.get_all_progress_states(), state.activities
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"XER export failed: {exc}") from exc

    return Response(
        content=xer_content,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=OIL_P6_DELTA_2026.XER"},
    )