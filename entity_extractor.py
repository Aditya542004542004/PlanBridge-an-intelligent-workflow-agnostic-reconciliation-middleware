"""
entity_extractor.py
=====================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 3 DELIVERABLE — Stage 1: Entity Extraction
------------------------------------------------------
``EntityExtractor`` pulls the hard, structurally-regular Oil & Gas EPC
entities out of free-form DPR text:

    * Activity IDs  ("PIP-L5-044-036") — via regex (strict P6 Activity ID syntax)
    * KP markers    ("KP 24+600")     — via regex (strict chainage syntax)
    * Line IDs      ("Line 24-A")     — via regex (strict alphanumeric syntax)
    * Facility names ("CGS Duliajan")  — via regex (known facility prefixes)
    * Discipline / Action verb         — via a keyword dictionary, matched
                                          two ways:
                                            1. spaCy lemma matching (when available)
                                            2. A pure regex/keyword fallback
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from schemas import ExtractedEntities

log = logging.getLogger("planbridge.entity_extractor")

try:
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover — exercised only in spaCy-less environments
    spacy = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ActionRule:
    """One row of the discipline/action-verb keyword dictionary."""
    keywords: tuple[str, ...]
    lemmas: tuple[str, ...]
    action: str
    discipline: str


class EntityExtractor:
    """
    Extracts Activity IDs, KP markers, Line IDs, Facility names, and
    Discipline/Action-verb pairs from raw DPR text.

    Usage
    -----
        extractor = EntityExtractor()
        entities = extractor.extract("150m HDD drilling finished near KP 24+600 for PIP-L5-044-036")
        # entities.activity_id == "PIP-L5-044-036"
        # entities.location_kp == "KP 24+600"
        # entities.action_verb == "HDD Drilling"
        # entities.discipline  == "Piping"
    """

    # ----------------------------------------------------------------------
    # Regex matchers — as specified for this project. Compiled once.
    # ----------------------------------------------------------------------
    ACTIVITY_ID_PATTERN = re.compile(
        r"\b[A-Z]{3}-L[56]-\d{3}-\d{3}\b", re.IGNORECASE
    )

    KP_PATTERN = re.compile(r"KP\s*\d+(?:\+\d+)?", re.IGNORECASE)

    LINE_ID_PATTERN = re.compile(
        r"(?:Line|Pipeline|Spool|Joint)\s*[A-Z0-9\-]+", re.IGNORECASE
    )

    FACILITY_PATTERN = re.compile(
        r"(?:CGS|OCS|GCP|Valve\s+Pit|Yard|Station)\s*[A-Za-z0-9\-]*|Duliajan|Numaligarh",
        re.IGNORECASE,
    )

    # ----------------------------------------------------------------------
    # Keyword dictionary for Disciplines & Action Verbs, in priority order
    # ----------------------------------------------------------------------
    ACTION_RULES: tuple[ActionRule, ...] = (
        ActionRule(
            keywords=("ndt", "radiography", "hydrotest", "hydro test", "cube test"),
            lemmas=("radiography", "hydrotest"),
            action="QA Inspection",
            discipline="HSE",
        ),
        ActionRule(
            keywords=("hdd", "drilling", "drill", "boring", "bore"),
            lemmas=("drill", "bore"),
            action="HDD Drilling",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("welded", "welding", "weld", "tie-in", "tie in", "tiein"),
            lemmas=("weld",),
            action="Tie-in Welding",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("excavated", "excavation", "trenching", "trench", "pit", "backfilling", "backfill", "road"),
            lemmas=("excavate", "trench", "backfill", "backfille"),
            action="Excavation",
            discipline="Civil",
        ),
    )

    _SPACY_MODEL_NAME = "en_core_web_sm"

    def __init__(self, use_spacy: bool = True) -> None:
        self._nlp: "Language | None" = None
        if use_spacy and spacy is not None:
            try:
                self._nlp = spacy.load(self._SPACY_MODEL_NAME)
                log.info("EntityExtractor: loaded spaCy model '%s'.", self._SPACY_MODEL_NAME)
            except OSError:
                log.warning(
                    "EntityExtractor: spaCy model '%s' not installed — falling back to regex-only matching.",
                    self._SPACY_MODEL_NAME,
                )
        elif use_spacy and spacy is None:
            log.warning("EntityExtractor: spaCy is not installed — falling back to regex-only matching.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, text: str) -> ExtractedEntities:
        if not text or not text.strip():
            return ExtractedEntities()

        activity_id = self._extract_activity_id(text)
        location_kp = self._extract_kp(text)
        line_id = self._extract_line_id(text)
        facility = self._extract_facility(text)
        action_verb, discipline = self._extract_action_and_discipline(text)

        # Construct ExtractedEntities safely
        extracted_data = {
            "location_kp": location_kp,
            "line_id": line_id,
            "facility": facility,
            "discipline": discipline,
            "action_verb": action_verb,
        }
        
        # Check if ExtractedEntities schema accepts activity_id
        if hasattr(ExtractedEntities, "__fields__") and "activity_id" in ExtractedEntities.__fields__:
            extracted_data["activity_id"] = activity_id

        entities = ExtractedEntities(**extracted_data)
        
        # Attach activity_id dynamically if model doesn't have it explicitly
        if activity_id and not hasattr(entities, "activity_id"):
            object.__setattr__(entities, "activity_id", activity_id)

        return entities

    # ------------------------------------------------------------------
    # Regex-based entity extractors
    # ------------------------------------------------------------------
    def _extract_activity_id(self, text: str) -> str | None:
        """Extract explicit Primavera Activity IDs like PIP-L5-044-036, CIV-L5-010-001."""
        match = self.ACTIVITY_ID_PATTERN.search(text)
        if not match:
            return None
        return match.group(0).upper()

    def _extract_kp(self, text: str) -> str | None:
        match = self.KP_PATTERN.search(text)
        if not match:
            return None
        return self._normalize_kp(match.group(0))

    @staticmethod
    def _normalize_kp(raw: str) -> str:
        digits = re.search(r"\d+(?:\+\d+)?", raw)
        return f"KP {digits.group(0)}" if digits else raw.strip().upper()

    def _extract_line_id(self, text: str) -> str | None:
        match = self.LINE_ID_PATTERN.search(text)
        if not match:
            return None
        return self._normalize_whitespace(match.group(0))

    def _extract_facility(self, text: str) -> str | None:
        match = self.FACILITY_PATTERN.search(text)
        if not match:
            return None
        return self._normalize_whitespace(match.group(0))

    @staticmethod
    def _normalize_whitespace(raw: str) -> str:
        return re.sub(r"\s+", " ", raw).strip()

    # ------------------------------------------------------------------
    # Discipline / action-verb extraction
    # ------------------------------------------------------------------
    def _extract_action_and_discipline(self, text: str) -> tuple[str | None, str | None]:
        text_lower = text.lower()
        lemmas: set[str] = set()
        if self._nlp is not None:
            try:
                doc = self._nlp(text)
                lemmas = {token.lemma_.lower() for token in doc}
            except Exception as exc:
                log.warning("EntityExtractor: spaCy processing failed (%s); continuing with regex-only.", exc)
                lemmas = set()

        for rule in self.ACTION_RULES:
            keyword_hit = any(
                re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in rule.keywords
            )
            lemma_hit = bool(lemmas & set(rule.lemmas))
            if keyword_hit or lemma_hit:
                return rule.action, rule.discipline

        return None, None