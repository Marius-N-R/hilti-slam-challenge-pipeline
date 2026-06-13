"""
Pipeline stages for the Hilti-Trimble SLAM Challenge.

Each stage is a function that takes a container and input directory,
and returns an output directory. Stages can be chained together.
"""

from .align import AlignStage
from .all import AllStage
from .base import Stage, StageRegistry
from .combined_align import CombinedAlignStage
from .combined_overlay import CombinedOverlayStage
from .final_eval import FinalEvalStage
from .final_output import FinalOutputStage
from .floorplan_align import FloorplanAlignStage
from .floorplan_edges import FloorplanEdgesStage
from .floorplan_overlay import FloorplanOverlayStage
from .image_selector import ImageSelectorStage
from .line_extractor import LineExtractorStage
from .pca_align import PcaAlignStage
from .rays import RaysStage
from .slam import SlamStage
from .window_align import WindowAlignStage
from .window_dino import WindowDinoStage
from .window_overlay import WindowOverlayStage
from .window_pose import WindowPoseStage
from .window_rectify import WindowRectifyStage
from .window_sam import WindowSamStage

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
registry.register(FinalEvalStage())
registry.register(ImageSelectorStage())
registry.register(WindowDinoStage())
registry.register(WindowSamStage())
registry.register(WindowRectifyStage())
registry.register(WindowPoseStage())
registry.register(WindowAlignStage())
registry.register(WindowOverlayStage())
registry.register(CombinedAlignStage())
registry.register(CombinedOverlayStage())
registry.register(FinalOutputStage())

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
    "FinalEvalStage",
    "ImageSelectorStage",
    "WindowDinoStage",
    "WindowSamStage",
    "WindowRectifyStage",
    "WindowPoseStage",
    "WindowAlignStage",
    "WindowOverlayStage",
    "CombinedAlignStage",
    "CombinedOverlayStage",
    "FinalOutputStage",
]
