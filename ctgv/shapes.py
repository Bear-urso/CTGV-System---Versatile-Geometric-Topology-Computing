"""
CTGV System - Shapes and Geometric Rules
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict

# =========================
# PROTO-GEBIT PICTOGRAMS
# =========================

class Shape(Enum):
    ORIGIN      = "●"
    FLOW        = "─"
    DECISOR     = "▲"
    MEMORY      = "■"
    RESONATOR   = "○"
    AMPLIFIER   = "◆"
    INHIBITOR   = "✖"
    TRANSFORMER = "⧉"
    LOOP        = "∞"
    SENSOR      = "◇"

# =========================
# FIELD PARAMETERS
# =========================

DECAY = {
    Shape.ORIGIN:      1.00,
    Shape.FLOW:        0.98,
    Shape.DECISOR:     0.90,
    Shape.MEMORY:      0.99,
    Shape.RESONATOR:   1.05,
    Shape.AMPLIFIER:   1.25,
    Shape.INHIBITOR:   0.40,
    Shape.TRANSFORMER: 0.95,
    Shape.LOOP:        0.98,
    Shape.SENSOR:      1.00,
}

# =========================
# GEOMETRIC CONSTRAINTS
# =========================

@dataclass
class GeometricConstraint:
    """Intrinsic topological constraints"""
    max_connections: int = float('inf')
    min_connections: int = 0
    allow_cycles: bool = True
    symmetric_required: bool = False

GEOMETRIC_RULES = {
    Shape.FLOW: GeometricConstraint(max_connections=2, allow_cycles=False),
    Shape.DECISOR: GeometricConstraint(max_connections=3, min_connections=2, symmetric_required=True),
    Shape.RESONATOR: GeometricConstraint(max_connections=4, allow_cycles=True, symmetric_required=True),
    Shape.LOOP: GeometricConstraint(max_connections=2, min_connections=2, allow_cycles=True, symmetric_required=True),
    Shape.ORIGIN: GeometricConstraint(max_connections=float('inf'), allow_cycles=True),
    Shape.MEMORY: GeometricConstraint(max_connections=float('inf'), allow_cycles=True),
    Shape.AMPLIFIER: GeometricConstraint(max_connections=float('inf'), allow_cycles=True),
    Shape.INHIBITOR: GeometricConstraint(max_connections=1, allow_cycles=False),
    Shape.TRANSFORMER: GeometricConstraint(max_connections=3, allow_cycles=True),
    Shape.SENSOR: GeometricConstraint(max_connections=1, allow_cycles=False),
}