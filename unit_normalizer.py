"""
unit_normalizer.py
===================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 2 DELIVERABLE — Deterministic unit normalization engine
-----------------------------------------------------------------
This module is intentionally NOT machine-learned. Unit conversion is a
closed, exact-mathematics problem — 150 meters is *always* 0.150 km — so
using an ML model here would trade determinism and auditability for no
benefit. The problem statement is explicit that this conversion must happen
"BEFORE any matching occurs", so `UnitNormalizer` has zero dependency on
the (probabilistic) matching/embedding components built in later phases.

Two responsibilities live here:

1. ``UnitNormalizer.parse(text)`` — a regex-based extractor that finds every
   "<number> <unit>" (or "<unit> <number>") phrase in a chunk of free text,
   e.g. pulling ``(150.0, "m")`` out of "150m HDD drilling finished".

2. ``UnitNormalizer.normalize(raw_quantity, raw_unit, target_unit=None)`` —
   applies a fixed conversion-rules table to turn a raw field measurement
   into OIL's standardized enterprise unit (KM, M, TONNES, JOINTS, SPOOLS,
   PIT), returning a full audit trail string alongside the converted value.

Every conversion factor is a named constant at the top of the file so the
rule table can be reviewed/audited in one glance without hunting through
class internals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------
# Conversion constants (exact, not approximated beyond standard precision)
# --------------------------------------------------------------------------
METERS_PER_KM = 1000.0
FEET_TO_METERS_FACTOR = 0.3048          # international foot, exact by definition
KG_PER_TONNE = 1000.0

# Length units that are physically comparable via simple conversion, even
# though PlanBridge's field-report normalization always targets KM for
# meter-denominated claims (per the fixed conversion table above). Some
# schedule activities are natively planned in meters (e.g. short HDD river
# crossings), so a KM-normalized field claim compared against an M-planned
# activity's target quantity is still a physically meaningful comparison —
# it just needs a unit conversion first, which is what
# ``convert_for_comparison`` below provides. This is a distinct concern
# from DPR-text normalization: it exists purely so ScoringEngine and
# ProgressEngine can compare two quantities that are expressed in
# different but compatible units.
_LENGTH_UNIT_KM_FACTOR = {"M": 0.001, "KM": 1.0}


def convert_for_comparison(quantity: float, from_unit: str, to_unit: str) -> Optional[float]:
    """
    Convert ``quantity`` from ``from_unit`` to ``to_unit`` purely for the
    purpose of comparing two quantities expressed in different (but
    physically compatible) units — e.g. a KM-normalized field claim
    against an M-denominated planned activity quantity.

    Returns ``None`` if the two units are not known to be comparable
    (different physical dimension, e.g. KM vs JOINTS, or an unrecognized
    unit) — callers should treat that as "this comparison is not valid",
    not attempt a fallback conversion.
    """
    if from_unit == to_unit:
        return quantity
    if from_unit in _LENGTH_UNIT_KM_FACTOR and to_unit in _LENGTH_UNIT_KM_FACTOR:
        km_equivalent = quantity * _LENGTH_UNIT_KM_FACTOR[from_unit]
        return km_equivalent / _LENGTH_UNIT_KM_FACTOR[to_unit]
    return None


class UnitConversionError(ValueError):
    """Raised when a raw unit cannot be resolved to a known conversion rule
    and no usable fallback (target_unit override) was supplied."""


class QuantityParseError(ValueError):
    """Raised when a phrase claims to carry a numeric quantity but the
    number cannot actually be parsed (e.g. malformed numerals)."""


@dataclass(frozen=True)
class ConversionRule:
    """
    One row of the conversion-rules table.

    target_unit : the standardized enterprise unit this raw unit converts to
    factor      : the numeric factor used in the conversion
    operation   : "divide" or "multiply" — raw_quantity <op> factor = normalized_quantity
    description : human-readable audit-trail text, e.g. "Meters to KM (div 1000)"
    """
    target_unit: str
    factor: float
    operation: str  # "divide" | "multiply"
    description: str


@dataclass(frozen=True)
class ParsedQuantity:
    """One quantity+unit phrase found in a block of free text."""
    raw_quantity: float
    raw_unit: str          # exactly as matched in the source text (lowercased)
    raw_phrase: str        # a short contextual snippet containing the match
    start: int              # character offset of the match in the source text
    end: int                # character offset (exclusive) of the match end


class UnitNormalizer:
    """
    Deterministic quantity extraction + unit conversion engine.

    Usage
    -----
        normalizer = UnitNormalizer()
        matches = normalizer.parse("150m HDD drilling finished")
        qty, unit, audit = normalizer.normalize(matches[0].raw_quantity, matches[0].raw_unit)
        # qty == 0.150, unit == "KM", audit == "Meters to KM (div 1000)"
    """

    # ----------------------------------------------------------------------
    # Unit alias table: maps every spelling variant seen in field reports to
    # a single canonical internal unit code. This is the "messy input" side.
    # ----------------------------------------------------------------------
    UNIT_ALIASES: dict[str, str] = {
        # metres
        "m": "M", "meter": "M", "meters": "M", "metre": "M", "metres": "M",
        # kilometres
        "km": "KM", "km.": "KM", "kilometer": "KM", "kilometers": "KM",
        "kilometre": "KM", "kilometres": "KM",
        # feet
        "ft": "FT", "feet": "FT", "foot": "FT",
        # kilograms
        "kg": "KG", "kgs": "KG", "kilogram": "KG", "kilograms": "KG",
        # tonnes
        "tonne": "TONNES", "tonnes": "TONNES", "mt": "TONNES", "ton": "TONNES", "tons": "TONNES",
        # count-based units — preserved as-is, no numeric conversion
        "joint": "JOINTS", "joints": "JOINTS",
        "spool": "SPOOLS", "spools": "SPOOLS",
        "pit": "PIT", "pits": "PIT",
    }

    # ----------------------------------------------------------------------
    # Conversion-rules table: canonical unit -> (target unit, factor, op).
    # This is the "standardized output" side, and the single source of
    # truth for every arithmetic conversion PlanBridge performs.
    # ----------------------------------------------------------------------
    CONVERSION_RULES: dict[str, ConversionRule] = {
        "M": ConversionRule(
            target_unit="KM", factor=METERS_PER_KM, operation="divide",
            description=f"Meters to KM (div {METERS_PER_KM:g})",
        ),
        "KM": ConversionRule(
            target_unit="KM", factor=1.0, operation="multiply",
            description="Kilometers to KM (x1, already standard)",
        ),
        "FT": ConversionRule(
            target_unit="M", factor=FEET_TO_METERS_FACTOR, operation="multiply",
            description=f"Feet to M (x{FEET_TO_METERS_FACTOR})",
        ),
        "KG": ConversionRule(
            target_unit="TONNES", factor=KG_PER_TONNE, operation="divide",
            description=f"Kilograms to TONNES (div {KG_PER_TONNE:g})",
        ),
        "TONNES": ConversionRule(
            target_unit="TONNES", factor=1.0, operation="multiply",
            description="Tonnes to TONNES (x1, already standard)",
        ),
        "JOINTS": ConversionRule(
            target_unit="JOINTS", factor=1.0, operation="multiply",
            description="Joints preserved as-is (count-based unit, no conversion)",
        ),
        "SPOOLS": ConversionRule(
            target_unit="SPOOLS", factor=1.0, operation="multiply",
            description="Spools preserved as-is (count-based unit, no conversion)",
        ),
        "PIT": ConversionRule(
            target_unit="PIT", factor=1.0, operation="multiply",
            description="Pits preserved as-is (count-based unit, no conversion)",
        ),
    }

    # Regex: a number (optionally decimal) followed by whitespace-optional
    # unit token, unit token drawn from the alias table (longest tokens
    # first so e.g. "kilometers" isn't cut short by a shorter alternative).
    # A negative lookahead prevents matching mid-word (e.g. "5mm" won't
    # register as "5m" + stray "m").
    _NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"

    def __init__(self) -> None:
        # Build the unit-alternation regex once at construction time,
        # longest aliases first so greedy alternation doesn't truncate
        # e.g. "kilometers" down to matching just "km"-like substrings.
        aliases_sorted = sorted(self.UNIT_ALIASES.keys(), key=len, reverse=True)
        unit_alternation = "|".join(re.escape(u) for u in aliases_sorted)

        # Trailing form: "150m", "0.15 km", "500 kg" — number then unit.
        self._trailing_pattern = re.compile(
            rf"{self._NUMBER_PATTERN}\s*({unit_alternation})(?![a-zA-Z])",
            flags=re.IGNORECASE,
        )
        # Leading form: "km 2.5" — unit then number (rarer in field text,
        # but supported for robustness against spreadsheet-style columns
        # like "Unit: KM, Qty: 2.5").
        self._leading_pattern = re.compile(
            rf"\b({unit_alternation})\b[\s:]*{self._NUMBER_PATTERN}",
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse(self, text: str) -> list[ParsedQuantity]:
        """
        Scan free text and extract every quantity+unit phrase found.

        Returns a list of ``ParsedQuantity`` in the order they appear in
        the text. Overlapping trailing/leading matches at the same
        character offset are de-duplicated (trailing form takes
        precedence, since it is the far more common phrasing in DPRs).

        Edge cases handled:
        * No quantity present at all -> returns an empty list (caller
          decides how to treat a report with zero extractable numbers).
        * A number with no recognizable unit attached (e.g. "completed
          on Section 4B") -> that number is silently skipped, since a
          bare number with no unit cannot be safely normalized.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        matches: list[ParsedQuantity] = []
        claimed_spans: list[tuple[int, int]] = []

        for m in self._trailing_pattern.finditer(text):
            raw_quantity = self._safe_float(m.group(1))
            if raw_quantity is None:
                continue
            raw_unit = m.group(2).lower()
            start, end = m.span()
            claimed_spans.append((start, end))
            matches.append(
                ParsedQuantity(
                    raw_quantity=raw_quantity,
                    raw_unit=raw_unit,
                    raw_phrase=self._context_sentence(sentences, start),
                    start=start,
                    end=end,
                )
            )

        for m in self._leading_pattern.finditer(text):
            start, end = m.span()
            if self._overlaps(start, end, claimed_spans):
                continue  # already captured by the trailing-form pass
            raw_quantity = self._safe_float(m.group(2))
            if raw_quantity is None:
                continue
            raw_unit = m.group(1).lower()
            claimed_spans.append((start, end))
            matches.append(
                ParsedQuantity(
                    raw_quantity=raw_quantity,
                    raw_unit=raw_unit,
                    raw_phrase=self._context_sentence(sentences, start),
                    start=start,
                    end=end,
                )
            )

        matches.sort(key=lambda pq: pq.start)
        return matches

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def normalize(
        self,
        raw_quantity: float,
        raw_unit: str,
        target_unit: Optional[str] = None,
    ) -> tuple[float, str, str]:
        """
        Convert a raw field quantity into PlanBridge's standardized unit.

        Parameters
        ----------
        raw_quantity : the quantity as written in the field report.
        raw_unit     : the unit as written in the field report (any
                       recognized alias, case-insensitive), e.g. "m",
                       "Kgs", "TONNES".
        target_unit  : optional override. If provided and it matches the
                       unit's natural conversion target, behaves exactly
                       as the default rule. If provided and it does NOT
                       match (e.g. asking to force "m" into "TONNES"),
                       raises ``UnitConversionError`` — PlanBridge never
                       silently performs a physically meaningless
                       conversion.

        Returns
        -------
        (normalized_quantity, normalized_unit, conversion_applied)

        Raises
        ------
        UnitConversionError : raw_unit is not recognized, or target_unit
            conflicts with the unit's defined conversion target.
        ValueError : raw_quantity is not a finite, non-negative number.
        """
        if raw_quantity is None or raw_quantity < 0:
            raise ValueError(f"raw_quantity must be a non-negative number, got: {raw_quantity!r}")

        canonical_unit = self.UNIT_ALIASES.get(raw_unit.strip().lower())
        if canonical_unit is None:
            raise UnitConversionError(
                f"Unrecognized unit '{raw_unit}'. Known units: {sorted(set(self.UNIT_ALIASES.values()))}"
            )

        rule = self.CONVERSION_RULES[canonical_unit]

        if target_unit is not None and target_unit.upper() != rule.target_unit:
            raise UnitConversionError(
                f"Cannot convert '{raw_unit}' to requested target '{target_unit}'. "
                f"'{raw_unit}' only converts to '{rule.target_unit}'."
            )

        if rule.operation == "divide":
            normalized_quantity = raw_quantity / rule.factor
        elif rule.operation == "multiply":
            normalized_quantity = raw_quantity * rule.factor
        else:  # pragma: no cover — defensive guard against future rule bugs
            raise UnitConversionError(f"Unknown conversion operation '{rule.operation}' for unit '{raw_unit}'.")

        # Round to 3 decimal places — matches Phase 1's planned_quantity
        # precision convention for KM/M and avoids float noise (e.g.
        # 0.15000000000000002) leaking into downstream comparisons.
        normalized_quantity = round(normalized_quantity, 3)

        return normalized_quantity, rule.target_unit, rule.description

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(token: str) -> Optional[float]:
        """Parse a numeric token, returning None (never raising) on
        malformed input so a single bad match doesn't abort a whole
        report's extraction."""
        try:
            return float(token)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        return any(not (end <= s or start >= e) for s, e in spans)

    @staticmethod
    def _split_sentences(text: str) -> list[tuple[int, int, str]]:
        """Split text into (start_offset, end_offset, sentence) tuples so
        a match's containing sentence can be looked up as readable
        context for ``raw_phrase``."""
        sentences: list[tuple[int, int, str]] = []
        cursor = 0
        for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
            if not piece:
                continue
            start = text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            sentences.append((start, end, piece))
            cursor = end
        if not sentences:
            sentences.append((0, len(text), text.strip()))
        return sentences

    @staticmethod
    def _context_sentence(sentences: list[tuple[int, int, str]], offset: int) -> str:
        """Return the sentence containing character offset `offset`,
        falling back to the nearest sentence if offsets don't line up
        exactly (defensive against edge-case whitespace handling)."""
        for start, end, sentence in sentences:
            if start <= offset < end:
                return sentence
        return sentences[-1][2] if sentences else ""
