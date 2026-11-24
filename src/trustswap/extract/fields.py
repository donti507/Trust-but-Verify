"""Field paths and the JSON schema the extraction models must return (Step 10).

The model copies each value verbatim from the document and names the line it
is on; character offsets are computed by the parser, not by the model.
"""
from __future__ import annotations

from trustswap.common.schema import FixedLeg, FloatingLeg

TOP_LEVEL = ("trade_date", "effective_date", "termination_date")
LEG_CLASSES = (("fixed_leg", FixedLeg), ("floating_leg", FloatingLeg), ("floating_leg_2", FloatingLeg))
PATHS = (TOP_LEVEL
         + tuple(f"{leg}.{name}" for leg, cls in LEG_CLASSES for name in cls.model_fields)
         + ("notice_address",))
STRUCTURES = ("fixed_float", "float_float")
STRUCTURE_LEGS = {"fixed_float": ("fixed_leg", "floating_leg"), "float_float": ("floating_leg", "floating_leg_2")}

_FIELD = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value", "line", "confidence"],
    "properties": {
        "value": {"type": ["string", "null"]},
        "line": {"type": ["integer", "null"]},
        "confidence": {"type": "number"},
    },
}


def output_schema() -> dict:
    """JSON schema for one extraction (strict-mode compatible: every key required, no extras)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["structure", "fields"],
        "properties": {
            "structure": {"type": "string", "enum": list(STRUCTURES)},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "required": list(PATHS),
                "properties": {path: _FIELD for path in PATHS},
            },
        },
    }
