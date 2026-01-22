"""
CTGV System - Versatile Geometric Topology Computing

A Proto-Gebit cognitive engine that models thinking through topological fields.
"""

from .shapes import Shape, DECAY, GEOMETRIC_RULES, GeometricConstraint
from .vector_field import VectorField
from .gebit import Gebit
from .engine import CTGVEngine
from .modeler import GeometricDataModeler
from .arbiter import TemporalBindingArbiter
from .clarification import ClarificationEngine
from .clarification_mechanism import ClarificationMechanism, EnhancedTemporalBindingArbiter
from .distributed_ctgv import (
    DistributedCTGVEngine, StreamProcessingPipeline, GPUAccelerator,
    HyperscaleCTGVSYSTEM, AutoScalingManager, SystemMonitor
)
from .utils import visualize_ctgv_processing

__version__ = "1.0.0"
__all__ = [
    'Shape', 'DECAY', 'GEOMETRIC_RULES', 'GeometricConstraint',
    'VectorField', 'Gebit', 'CTGVEngine', 'GeometricDataModeler',
    'TemporalBindingArbiter', 'ClarificationEngine', 'ClarificationMechanism',
    'EnhancedTemporalBindingArbiter', 'DistributedCTGVEngine',
    'StreamProcessingPipeline', 'GPUAccelerator', 'HyperscaleCTGVSYSTEM',
    'AutoScalingManager', 'SystemMonitor', 'visualize_ctgv_processing'
]