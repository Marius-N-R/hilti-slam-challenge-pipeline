"""Base classes for pipeline stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def stage_output_path(config: "StageConfig", stage_name: str) -> Path:
    """Resolve a sibling stage's canonical output directory.

    Output layout convention (set by pipeline.py): <output_root>/<stage>/<input_folder_name>/.
    `config.input_root` holds the output root (the field name is historical),
    and `config.extra['current_input_name']` holds the folder name.
    """
    folder = config.extra.get("current_input_name", "")
    if not folder or not config.input_root:
        raise RuntimeError(
            "Cannot resolve stage output path: input_root or current_input_name missing"
        )
    return Path(config.input_root) / stage_name / folder


@dataclass
class StageConfig:
    """Configuration passed to stages."""
    verbose: bool = False
    input_root: str = ""

    # SLAM options
    slam_rate: float = 0.5  # OpenVINS needs <= 0.5x replay on this dataset for stable init
    slam_timeout: int = 0  # seconds (0 disables timeout)

    # Custom options (for extensibility)
    extra: Dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Abstract base class for pipeline stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this stage."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this stage does."""
        pass

    @property
    def requires_ros_runtime(self) -> bool:
        """Whether this stage needs a running ROS2 system (ros2 launch, etc)."""
        return False

    @property
    def requires_container(self) -> bool:
        """Whether this stage needs a container build/run. Host-only stages return False."""
        return True

    @property
    def container_profile(self) -> str:
        """Container profile required by this stage."""
        return "ros"

    @property
    def input_type(self) -> str:
        """Type of input this stage expects: 'rosbag', 'euroc', 'directory'."""
        return "rosbag"

    @property
    def output_type(self) -> str:
        """Type of output this stage produces."""
        return "rosbag"

    @abstractmethod
    def run(
        self,
        runner: Any,
        input_dir: Path,
        config: StageConfig,
    ) -> Path:
        """
        Execute this stage.

        Args:
            runner: The selected container runtime backend.
            input_dir: Input directory mounted from host.
            config: Stage configuration.

        Returns:
            Output directory with results.
        """
        pass

    def get_ros_source_cmd(self) -> str:
        """Return the bash command to source ROS2 and workspace."""
        return (
            "set +u && "
            "source /opt/ros/jazzy/setup.bash && "
            "source /root/ros2_ws/install/setup.bash && "
            "set -u"
        )


class StageRegistry:
    """Registry for available pipeline stages."""

    def __init__(self):
        self._stages: Dict[str, Stage] = {}

    def register(self, stage: Stage) -> None:
        """Register a stage."""
        self._stages[stage.name] = stage

    def get(self, name: str) -> Optional[Stage]:
        """Get a stage by name."""
        return self._stages.get(name)

    def list_stages(self) -> List[Stage]:
        """List all registered stages."""
        return list(self._stages.values())

    def get_names(self) -> List[str]:
        """Get all stage names."""
        return list(self._stages.keys())

    def print_stages(self) -> None:
        """Print all available stages."""
        print("\nAvailable pipeline stages:")
        print("-" * 60)
        for stage in self._stages.values():
            ros_marker = " [ROS]" if stage.requires_ros_runtime else ""
            print(f"  {stage.name:12s} - {stage.description}{ros_marker}")
        print()
