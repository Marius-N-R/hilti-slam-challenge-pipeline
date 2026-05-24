"""
Pipeline stages for the Hilti-Trimble SLAM Challenge.

Each stage is a function that takes a container and input directory,
and returns an output directory. Stages can be chained together.
"""

from .base import Stage, StageRegistry
from .clean import CleanStage
from .convert import ConvertStage
from .floorplan_overlay import FloorplanOverlayStage
from .pca_align import PcaAlignStage
from .plot_path import PlotPathStage
from .slam import SlamStage
from .stitch import StitchStage
from .windows_dino import WindowsDinoStage
from .windows_rectify import WindowsRectifyStage
from .windows_sam import WindowsSamStage
from .windows import WindowsStage

# Register all built-in stages
registry = StageRegistry()
registry.register(StitchStage())
registry.register(ConvertStage())
registry.register(SlamStage())
registry.register(PcaAlignStage())
registry.register(PlotPathStage())
registry.register(FloorplanOverlayStage())
registry.register(CleanStage())
registry.register(WindowsDinoStage())
registry.register(WindowsSamStage())
registry.register(WindowsRectifyStage())
registry.register(WindowsStage())

__all__ = [
    "Stage",
    "StageRegistry",
    "registry",
    "StitchStage",
    "ConvertStage",
    "SlamStage",
    "PcaAlignStage",
    "PlotPathStage",
    "FloorplanOverlayStage",
    "CleanStage",
    "WindowsDinoStage",
    "WindowsSamStage",
    "WindowsRectifyStage",
    "WindowsStage",
]
