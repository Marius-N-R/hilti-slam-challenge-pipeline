"""
Pipeline stages for the Hilti-Trimble SLAM Challenge.

Each stage is a function that takes a container and input directory,
and returns an output directory. Stages can be chained together.
"""

from .align import AlignStage
from .all import AllStage
from .base import Stage, StageRegistry
from .floorplan_align import FloorplanAlignStage
from .floorplan_edges import FloorplanEdgesStage
from .floorplan_overlay import FloorplanOverlayStage
from .line_extractor import LineExtractorStage
from .pca_align import PcaAlignStage
from .rays import RaysStage
from .slam import SlamStage

registry = StageRegistry()
registry.register(AllStage())
registry.register(SlamStage())
registry.register(AlignStage())
registry.register(PcaAlignStage())
registry.register(LineExtractorStage())
registry.register(FloorplanEdgesStage())
registry.register(RaysStage())
registry.register(FloorplanAlignStage())
registry.register(FloorplanOverlayStage())

__all__ = [
    "Stage",
    "StageRegistry",
    "registry",
    "AllStage",
    "SlamStage",
    "AlignStage",
    "PcaAlignStage",
    "LineExtractorStage",
    "FloorplanEdgesStage",
    "RaysStage",
    "FloorplanAlignStage",
    "FloorplanOverlayStage",
]
