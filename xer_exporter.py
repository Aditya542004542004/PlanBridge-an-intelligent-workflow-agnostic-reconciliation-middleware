"""
xer_exporter.py
==================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 6 DELIVERABLE — Primavera P6 .XER Schedule Delta Exporter
------------------------------------------------------------------------
``XERExporter`` turns PlanBridge's tracked ``ProgressState`` records back
into a Primavera P6-importable ``.XER`` text payload — the "write path"
that closes the loop: DPRs go in, a verified schedule delta comes out,
ready for a planner to import back into the enterprise P6 database.

Real ``.XER`` files are TAB-delimited text with a small, strict grammar:
    ERMHDR	<export metadata...>
    %T	<table name>
    %F	<tab-separated field names>
    %R	<tab-separated row values>       (one %R per row)
    %E
This module emits exactly that grammar for the ``TASK`` table (the table
the spec's field list describes), using real P6 field names and enum
values (e.g. ``status_code`` uses the actual P6 codes ``TK_NotStart`` /
``TK_Active`` / ``TK_Complete``) so the output is recognizable to anyone
who has worked with a real ``.XER`` file, even though this module — by
design and scope — only emits the ``TASK`` table, not a full multi-table
project export (``PROJECT``, ``PROJWBS``, ``CALENDAR``, etc.). A
production system feeding this back into a live P6 database would need
those supporting tables too; this exporter's job is the progress *delta*,
which is what Phase 6 asks for.

Design decision worth flagging explicitly: which number becomes P6's
``act_qty`` / ``phys_complete_pct``?
--------------------------------------------------------------------------------
PlanBridge tracks TWO progress numbers per activity (Phase 5): what the
site *claims* (``physical_claimed_quantity``) and what QA/QC has
independently *verified* (``verified_earned_quantity``). Exporting the
unverified site claim back into the official enterprise schedule would
quietly defeat the entire point of building a dual-tracking, audit-grade
system in the first place — a CVC/CAG-compliant delta should reflect
audited progress, not raw unverified field claims. **By default, this
exporter uses the QA-verified figures.** A caller who explicitly wants a
"draft" preview export showing unverified site claims instead can pass
``use_verified_progress=False`` — but that is an opt-in override, not the
default.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas import ProgressState

log = logging.getLogger("planbridge.xer_exporter")

# Real Primavera P6 activity status codes.
STATUS_NOT_STARTED = "TK_NotStart"
STATUS_ACTIVE = "TK_Active"
STATUS_COMPLETE = "TK_Complete"

TASK_TABLE_FIELDS = (
    "task_id", "proj_id", "wbs_id", "task_code", "task_name",
    "status_code", "target_qty", "act_qty", "phys_complete_pct",
)


class XERExporter:
    """
    Generates a Primavera P6 ``.XER``-format TASK table delta payload from
    PlanBridge's tracked progress states.

    Usage
    -----
        exporter = XERExporter(project_id="20286")
        xer_text = exporter.generate_xer_delta(progress_states, activities)
        exporter.export_to_file("data/OIL_P6_DELTA_2026.XER", xer_text)
    """

    def __init__(self, project_id: str = "20286", exported_by: str = "PLANBRIDGE") -> None:
        """
        Parameters
        ----------
        project_id  : the P6 ``proj_id`` this delta applies to. Real P6
                      ``proj_id`` values are internal numeric database keys
                      looked up from the enterprise PROJECT table — since
                      PlanBridge's synthetic dataset represents a single
                      project, this is a fixed, configurable constant here
                      rather than something looked up per-activity.
        exported_by : recorded in the ERMHDR line as the export's origin
                      system/user, for traceability in whatever P6 import
                      log picks this file up.
        """
        self.project_id = project_id
        self.exported_by = exported_by

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_xer_delta(
        self,
        progress_states: list[ProgressState],
        activities: list[dict[str, Any]],
        use_verified_progress: bool = True,
        include_zero_progress: bool = False,
    ) -> str:
        """
        Build the full ``.XER`` text payload for the given progress states.

        Parameters
        ----------
        progress_states        : the ProgressState records to export.
        activities              : the source activities.json records, used
                                  to look up wbs_code and activity_name
                                  (ProgressState itself doesn't carry those).
        use_verified_progress   : if True (default), ``act_qty`` and
                                  ``phys_complete_pct`` come from the
                                  QA-verified figures. If False, from the
                                  unverified physical site claims instead —
                                  an explicit opt-in for a "draft preview"
                                  export. See the module docstring for why
                                  this defaults to verified progress.
        include_zero_progress   : if False (default), activities with no
                                  progress at all (0% both physical and
                                  verified) are omitted — this is a DELTA
                                  export, only activities that actually
                                  changed since baseline belong in it. Set
                                  True for a full-schedule dump instead.

        Returns
        -------
        The complete ``.XER`` text payload as a string (TAB-delimited,
        ready to write to a ``.XER`` file as-is).

        Never raises solely because ``progress_states`` is empty — an
        empty-but-well-formed ``.XER`` file (header + empty TASK table) is
        returned, since "no progress to export yet" is a legitimate state
        for a freshly-started project, not an error.
        """
        activities_by_id = {act.get("activity_id"): act for act in activities}

        rows: list[str] = []
        skipped_missing_activity = 0
        for state in progress_states:
            has_progress = state.physical_progress_pct > 0 or state.verified_progress_pct > 0
            if not include_zero_progress and not has_progress:
                continue

            activity = activities_by_id.get(state.activity_id)
            if activity is None:
                log.warning(
                    "generate_xer_delta: no source activity record found for activity_id=%s "
                    "(wbs_code/task_name will be blank); skipping row.",
                    state.activity_id,
                )
                skipped_missing_activity += 1
                continue

            rows.append(self._build_task_row(state, activity, use_verified_progress))

        if skipped_missing_activity:
            log.warning(
                "generate_xer_delta: skipped %d progress state(s) with no matching activity record.",
                skipped_missing_activity,
            )

        return self._assemble_xer_document(rows)

    def export_to_file(self, filepath: str, xer_content: str) -> str:
        """
        Write ``xer_content`` to ``filepath``, creating parent directories
        as needed and writing atomically (temp file + ``os.replace``) so a
        concurrent reader never observes a partially-written ``.XER`` file.

        Returns the resolved filepath on success.

        Raises
        ------
        OSError : if the write genuinely fails (e.g. disk full, permission
                 denied) — re-raised with added context rather than
                 swallowed, since a silently-failed schedule export is far
                 worse than a loud one.
        """
        path = Path(filepath)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"export_to_file: could not create directory '{path.parent}': {exc}") from exc

        fd, temp_path = tempfile.mkstemp(prefix=".xer_export_", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(xer_content)
            os.replace(temp_path, path)
        except OSError as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise OSError(f"export_to_file: failed to write '{path}': {exc}") from exc

        log.info("XERExporter: wrote %d bytes to %s.", len(xer_content.encode("utf-8")), path)
        return str(path)

    # ------------------------------------------------------------------
    # Internal row/document assembly
    # ------------------------------------------------------------------
    def _build_task_row(
        self, state: ProgressState, activity: dict[str, Any], use_verified_progress: bool
    ) -> str:
        task_id = self._synthetic_numeric_id(state.activity_id)
        wbs_id = activity.get("wbs_code", "")
        task_code = state.activity_id
        task_name = activity.get("activity_name", "")

        if use_verified_progress:
            act_qty = state.verified_earned_quantity
            complete_pct = state.verified_progress_pct
        else:
            act_qty = state.physical_claimed_quantity
            complete_pct = state.physical_progress_pct

        status_code = self._status_code_for(complete_pct)

        values = [
            task_id,
            self.project_id,
            str(wbs_id),
            task_code,
            task_name,
            status_code,
            f"{state.planned_quantity:.4f}",
            f"{act_qty:.4f}",
            f"{complete_pct:.2f}",
        ]
        # Tab-delimited, with %R row marker — real XER row syntax.
        return "%R\t" + "\t".join(self._sanitize_field(v) for v in values)

    def _assemble_xer_document(self, rows: list[str]) -> str:
        lines = [
            self._build_ermhdr_line(),
            "%T\tTASK",
            "%F\t" + "\t".join(TASK_TABLE_FIELDS),
            *rows,
            "%E",
        ]
        return "\n".join(lines) + "\n"

    def _build_ermhdr_line(self) -> str:
        """
        A minimal but structurally real ERMHDR line — the mandatory first
        line of any .XER file, identifying export version/date/origin. A
        full production export would also emit PROJECT/PROJWBS/CALENDAR
        tables that a P6 import expects alongside ERMHDR; this exporter is
        scoped to the TASK delta specifically, per Phase 6's brief.
        """
        export_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Real ERMHDR fields (subset): version, date, project-flag(P),
        # exported-by, exported-by-full-name, db-flag, currency, language.
        fields = ["ERMHDR", "20.12", export_date, "P", self.exported_by, "PlanBridge Reconciliation Engine", "Y", "INR", "en"]
        return "\t".join(fields)

    @staticmethod
    def _status_code_for(complete_pct: float) -> str:
        if complete_pct <= 0.0:
            return STATUS_NOT_STARTED
        if complete_pct >= 100.0:
            return STATUS_COMPLETE
        return STATUS_ACTIVE

    @staticmethod
    def _synthetic_numeric_id(activity_id: str) -> str:
        """
        Real P6 task_id values are internal numeric database keys, not the
        human-readable activity code. Since PlanBridge doesn't have (and
        for a delta-only export doesn't need) a live P6 database
        connection to look up the true internal ID, this generates a
        stable, deterministic numeric ID from the activity_id — the same
        activity always maps to the same synthetic task_id across exports,
        which matters for a P6 import tool trying to match rows to
        existing tasks by ID consistency.
        """
        digest = hashlib.sha256(activity_id.encode("utf-8")).hexdigest()
        return str(int(digest[:8], 16) % 900000 + 100000)

    @staticmethod
    def _sanitize_field(value: Any) -> str:
        """Strip tabs/newlines out of a field value before writing it into
        a tab-delimited row — a task_name containing a stray tab or
        newline would otherwise silently corrupt the file's column
        alignment."""
        text = str(value)
        return text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
