"""
entity_extractor.py
PlanBridge — Stage 1: Entity Extraction
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
except ImportError:
    spacy = None
    Language = None

@dataclass(frozen=True)
class ActionRule:
    keywords: tuple[str, ...]
    lemmas: tuple[str, ...]
    action: str
    discipline: str

class EntityExtractor:
    """Extracts Activity IDs, KP markers, Line IDs, Facility names, and Actions."""

    ACTIVITY_ID_PATTERN = re.compile(
        r"\b[A-Z]{3}-L[56]-\d{3}-\d{3}\b", re.IGNORECASE
    )
    KP_PATTERN = re.compile(r"KP\s*\d+(?:\+\d+)?", re.IGNORECASE)
    LINE_ID_PATTERN = re.compile(
        r"(?:Line|Pipeline|Spool|Joint|Section)\s*[A-Z0-9\-]+", re.IGNORECASE
    )
    FACILITY_PATTERN = re.compile(
        r"Booster\s+Station\s+Naharkatiya|Pump\s+Station\s+Barekuri|Terminal\s+Duliajan|"
        r"CGS\s+Duliajan|CGS\s+Moran|Trunkline\s+ROW|OCS-4|"
        r"Naharkatiya|Barekuri|Duliajan|Moran",
        re.IGNORECASE,
    )

    ACTION_RULES: tuple[ActionRule, ...] = (
        ActionRule(
            keywords=("ndt", "radiography", "hydrotest", "hydro test", "cube test", "qa clearance"),
            lemmas=("radiography", "hydrotest", "test"),
            action="QA Inspection",
            discipline="HSE",
        ),
        ActionRule(
            keywords=("mock drill", "emergency drill", "safety drill", "compliance audit", "hse audit"),
            lemmas=("audit",),
            action="HSE Monitoring",
            discipline="HSE",
        ),
        ActionRule(
            keywords=("hdd", "drilling", "boring", "river crossing"),
            lemmas=("drill", "bore", "cross"),
            action="HDD Drilling",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("welded", "welding", "weld", "tie-in", "tie in", "tiein"),
            lemmas=("weld", "tie"),
            action="Tie-in Welding",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("spool", "spool fabrication", "rack fabrication"),
            lemmas=("fabricate", "spool"),
            action="Spool Fabrication",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("pipe stringing", "stringing", "strung", "pipe laying"),
            lemmas=("string", "lay"),
            action="Pipe Stringing",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("hydrotesting", "hydro test", "pressure test"),
            lemmas=("hydrotest", "test"),
            action="Hydrotesting",
            discipline="Piping",
        ),
        ActionRule(
            keywords=("row clearing", "clearing", "bush clearing", "ground preparation"),
            lemmas=("clear", "clean"),
            action="ROW Clearing",
            discipline="Civil",
        ),
        ActionRule(
            keywords=("access road", "road construction", "paving"),
            lemmas=("road", "construct"),
            action="Access Road Construction",
            discipline="Civil",
        ),
        ActionRule(
            keywords=("valve pit", "pit excavation", "pit"),
            lemmas=("excavate", "pit"),
            action="Valve Pit Excavation",
            discipline="Civil",
        ),
        ActionRule(
            keywords=("rcc foundation", "foundation casting", "foundation", "casting", "concrete"),
            lemmas=("cast", "found"),
            action="Foundation Casting",
            discipline="Civil",
        ),
        ActionRule(
            keywords=("trenching", "trench", "backfilling", "backfill", "excavated", "excavation"),
            lemmas=("excavate", "trench", "backfill"),
            action="Excavation",
            discipline="Civil",
        ),
        ActionRule(
            keywords=("manifold", "manifold setup", "cgs manifold"),
            lemmas=("manifold", "setup"),
            action="CGS Manifold Setup",
            discipline="Mechanical",
        ),
        ActionRule(
            keywords=("pig launcher", "pig receiver", "launcher fabrication", "receiver fabrication"),
            lemmas=("launch", "receive"),
            action="Pig Launcher Fabrication",
            discipline="Mechanical",
        ),
        ActionRule(
            keywords=("skid", "skid-mounted", "equipment installation", "skid installation"),
            lemmas=("skid", "install", "mount"),
            action="Skid Installation",
            discipline="Mechanical",
        ),
        ActionRule(
            keywords=("scada", "instrumentation", "cable laying", "control cable"),
            lemmas=("cable", "wire"),
            action="SCADA Cable Laying",
            discipline="Electrical",
        ),
        ActionRule(
            keywords=("transformer", "mcc panel", "mcc", "substation panel"),
            lemmas=("transform", "panel"),
            action="Transformer Installation",
            discipline="Electrical",
        ),
        ActionRule(
            keywords=("cathodic", "cathodic protection", "cp test", "test station"),
            lemmas=("protect", "station"),
            action="CP Test Station",
            discipline="Electrical",
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
                log.warning("EntityExtractor: spaCy model not installed — using regex.")
        elif use_spacy and spacy is None:
            log.warning("EntityExtractor: spaCy not installed — using regex.")

    def extract(self, text: str) -> ExtractedEntities:
        if not text or not text.strip():
            return ExtractedEntities()

        activity_id = self._extract_activity_id(text)
        location_kp = self._extract_kp(text)
        line_id = self._extract_line_id(text)
        facility = self._extract_facility(text)
        action_verb, discipline = self._extract_action_and_discipline(text)

        extracted_data = {
            "location_kp": location_kp,
            "line_id": line_id,
            "facility": facility,
            "discipline": discipline,
            "action_verb": action_verb,
        }

        if hasattr(ExtractedEntities, "__fields__") and "activity_id" in ExtractedEntities.__fields__:
            extracted_data["activity_id"] = activity_id

        entities = ExtractedEntities(**extracted_data)
        if activity_id and not hasattr(entities, "activity_id"):
            object.__setattr__(entities, "activity_id", activity_id)

        return entities

    def _extract_activity_id(self, text: str) -> str | None:
        match = self.ACTIVITY_ID_PATTERN.search(text)
        return match.group(0).upper() if match else None

    def _extract_kp(self, text: str) -> str | None:
        match = self.KP_PATTERN.search(text)
        return self._normalize_kp(match.group(0)) if match else None

    @staticmethod
    def _normalize_kp(raw: str) -> str:
        digits = re.search(r"\d+(?:\+\d+)?", raw)
        return f"KP {digits.group(0)}" if digits else raw.strip().upper()

    def _extract_line_id(self, text: str) -> str | None:
        match = self.LINE_ID_PATTERN.search(text)
        return self._normalize_whitespace(match.group(0)) if match else None

    def _extract_facility(self, text: str) -> str | None:
        match = self.FACILITY_PATTERN.search(text)
        if not match:
            return None
        found = self._normalize_whitespace(match.group(0))
        found_lower = found.lower()
        if "terminal" in found_lower or found_lower == "duliajan":
            if "cgs" in found_lower:
                return "CGS Duliajan"
            return "Terminal Duliajan"
        if "moran" in found_lower:
            return "CGS Moran"
        if "barekuri" in found_lower:
            return "Pump Station Barekuri"
        if "naharkatiya" in found_lower:
            return "Booster Station Naharkatiya"
        if "ocs" in found_lower:
            return "OCS-4"
        if "trunkline" in found_lower or "row" in found_lower:
            return "Trunkline ROW"
        return found

    @staticmethod
    def _normalize_whitespace(raw: str) -> str:
        return re.sub(r"\s+", " ", raw).strip()

    def _extract_action_and_discipline(self, text: str) -> tuple[str | None, str | None]:
        text_lower = text.lower()
        lemmas: set[str] = set()
        if self._nlp is not None:
            try:
                doc = self._nlp(text)
                lemmas = {token.lemma_.lower() for token in doc}
            except Exception:
                lemmas = set()

        for rule in self.ACTION_RULES:
            keyword_hit = any(
                re.search(rf"\b{re.escape(kw)}", text_lower) for kw in rule.keywords
            )
            lemma_hit = bool(lemmas & set(rule.lemmas))
            if keyword_hit or lemma_hit:
                return rule.action, rule.discipline

        return None, None
