"""Container runtime backend for Docker and Apptainer execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CACHE_DIR = ROOT / ".cache"
RUNTIME_DIR = CACHE_DIR / "runtime"


@dataclass(frozen=True)
class ProfileImage:
    profile: str
    docker_image: str
    dockerfile: Path
    apptainer_def: Path
    docker_archive: Path
    apptainer_image: Path


PROFILE_IMAGES = {
    "ros": ProfileImage(
        profile="ros",
        docker_image="slam-workspace:latest",
        dockerfile=ROOT / "Dockerfile.workspace",
        apptainer_def=ROOT / "container_defs" / "workspace.def",
        docker_archive=CACHE_DIR / "slam-workspace.tar",
        apptainer_image=CACHE_DIR / "slam-workspace.sif",
    ),
    "windows_cpu": ProfileImage(
        profile="windows_cpu",
        docker_image="windows-pipeline-cpu:latest",
        dockerfile=ROOT / "Dockerfile.windows.cpu",
        apptainer_def=ROOT / "container_defs" / "windows_cpu.def",
        docker_archive=CACHE_DIR / "windows-pipeline-cpu.tar",
        apptainer_image=CACHE_DIR / "windows-pipeline-cpu.sif",
    ),
    "windows_gpu": ProfileImage(
        profile="windows_gpu",
        docker_image="windows-pipeline-gpu:latest",
        dockerfile=ROOT / "Dockerfile.windows.gpu",
        apptainer_def=ROOT / "container_defs" / "windows_gpu.def",
        docker_archive=CACHE_DIR / "windows-pipeline-gpu.tar",
        apptainer_image=CACHE_DIR / "windows-pipeline-gpu.sif",
    ),
}

APPTAINER_IMAGE_OVERRIDE_ENV = {
    "ros": "PIPELINE_APPTAINER_ROS_IMAGE",
    "windows_cpu": "PIPELINE_APPTAINER_WINDOWS_CPU_IMAGE",
    "windows_gpu": "PIPELINE_APPTAINER_WINDOWS_GPU_IMAGE",
}


class StageExecutionError(RuntimeError):
    """Execution failure with access to stage output."""

    def __init__(self, stage_name: str, returncode: int, output_dir: Path):
        super().__init__(f"{stage_name} exited with status {returncode}")
        self.stage_name = stage_name
        self.returncode = returncode
        self.output_dir = output_dir


@dataclass(frozen=True)
class BindMount:
    source: Path
    target: str
    read_only: bool = False


@dataclass
class ExecutionSpec:
    stage_name: str
    command: list[str]
    files: dict[str, str]
    env: dict[str, str] = field(default_factory=dict)
    extra_mounts: list[BindMount] = field(default_factory=list)
    workdir: str = "/workspace"
    use_gpu: bool = False


class ContainerBackend:
    """Execute stages via Docker or Apptainer/Singularity."""

    def __init__(self, runtime: str, scripts_dir: Path, cache_dir: Path | None = None):
        self.runtime = runtime
        self.scripts_dir = scripts_dir.resolve()
        self.cache_dir = (cache_dir or CACHE_DIR).resolve()
        self.runtime_dir = self.cache_dir / "stage-runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.apptainer_bin = self._resolve_apptainer_binary() if runtime == "apptainer" else None

    def host_has_nvidia_gpu(self) -> bool:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    def profile_key(self, container_profile: str, windows_device: str) -> str:
        if container_profile == "ros":
            return "ros"
        if container_profile != "windows":
            raise ValueError(f"Unsupported container profile: {container_profile}")

        use_gpu = windows_device == "cuda" or (
            windows_device == "auto" and self.host_has_nvidia_gpu()
        )
        return "windows_gpu" if use_gpu else "windows_cpu"

    def ensure_profile_image(self, container_profile: str, windows_device: str) -> Path | str:
        profile_key = self.profile_key(container_profile, windows_device)
        profile = PROFILE_IMAGES[profile_key]
        if self.runtime == "docker":
            self._ensure_docker_image(profile)
            return profile.docker_image
        if self.runtime == "apptainer":
            override = self._apptainer_image_override(profile)
            if override is not None:
                return override
            return self._ensure_apptainer_image(profile)
        raise ValueError(f"Unsupported container runtime: {self.runtime}")

    def run_stage(
        self,
        *,
        container_profile: str,
        input_dir: Path,
        config,
        spec: ExecutionSpec,
    ) -> Path:
        image_ref = self.ensure_profile_image(container_profile, config.windows_device)

        stage_root = Path(
            tempfile.mkdtemp(prefix=f"{spec.stage_name}-", dir=self.runtime_dir)
        ).resolve()
        bundle_dir = stage_root / "bundle"
        output_dir = stage_root / "output"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, contents in spec.files.items():
            target = bundle_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
            target.chmod(0o755)

        mounts = [
            BindMount(input_dir.resolve(), "/input", read_only=True),
            BindMount(output_dir, "/output", read_only=False),
            BindMount(bundle_dir, "/stage_runtime", read_only=True),
            BindMount(self.scripts_dir, "/opt/pipeline_scripts", read_only=True),
        ]
        mounts.extend(spec.extra_mounts)

        env = dict(spec.env)
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            env.setdefault("HF_TOKEN", hf_token)
        if config.sam3_checkpoint:
            mounts.append(
                BindMount(
                    Path(config.sam3_checkpoint).resolve(),
                    "/opt/windows_pipeline/checkpoints/sam3.pt",
                    read_only=True,
                )
            )

        command = self._build_run_command(
            image_ref=image_ref,
            mounts=mounts,
            env=env,
            workdir=spec.workdir,
            command=spec.command,
            use_gpu=spec.use_gpu,
        )
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise StageExecutionError(spec.stage_name, result.returncode, output_dir)
        return output_dir

    def _ensure_docker_image(self, profile: ProfileImage) -> None:
        self._require_binary("docker")
        result = subprocess.run(
            ["docker", "images", "-q", profile.docker_image],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            return

        print(f"[build] Building Docker image: {profile.docker_image}")
        if not profile.dockerfile.exists():
            raise FileNotFoundError(f"Dockerfile not found: {profile.dockerfile}")
        result = subprocess.run(
            ["docker", "build", "-t", profile.docker_image, "-f", str(profile.dockerfile), "."],
            cwd=profile.dockerfile.parent,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build Docker image: {profile.docker_image}")
        print(f"[build] Docker image built successfully: {profile.docker_image}")

    def _ensure_docker_archive(self, profile: ProfileImage) -> Path:
        self._ensure_docker_image(profile)
        profile.docker_archive.parent.mkdir(parents=True, exist_ok=True)
        if profile.docker_archive.exists():
            return profile.docker_archive

        print(f"[build] Exporting Docker image tarball: {profile.docker_image}")
        result = subprocess.run(
            ["docker", "save", profile.docker_image, "-o", str(profile.docker_archive)],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to export Docker image: {profile.docker_image}")
        print(f"[build] Docker image tarball created: {profile.docker_archive}")
        return profile.docker_archive

    def _ensure_apptainer_image(self, profile: ProfileImage) -> Path:
        if self.apptainer_bin is None:
            raise RuntimeError("Apptainer/Singularity runtime requested but no binary was found")
        profile.apptainer_image.parent.mkdir(parents=True, exist_ok=True)
        if profile.apptainer_image.exists():
            return profile.apptainer_image

        print(f"[build] Building Apptainer image: {profile.apptainer_image.name}")
        docker_bin = shutil.which("docker")
        if docker_bin:
            archive_path = self._ensure_docker_archive(profile)
            command = [
                self.apptainer_bin,
                "build",
                "--force",
                str(profile.apptainer_image),
                f"docker-archive://{archive_path}",
            ]
            cwd = ROOT
        else:
            if not profile.apptainer_def.exists():
                raise FileNotFoundError(
                    f"Apptainer definition file not found: {profile.apptainer_def}"
                )
            print(
                "[build] Docker not found; building Apptainer image from native definition file"
            )
            command = [
                self.apptainer_bin,
                "build",
                "--force",
                "--ignore-fakeroot-command",
                str(profile.apptainer_image),
                str(profile.apptainer_def),
            ]
            cwd = ROOT

        result = subprocess.run(command, cwd=cwd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build Apptainer image: {profile.apptainer_image}")
        print(f"[build] Apptainer image built successfully: {profile.apptainer_image}")
        return profile.apptainer_image

    def _build_run_command(
        self,
        *,
        image_ref: Path | str,
        mounts: list[BindMount],
        env: dict[str, str],
        workdir: str,
        command: list[str],
        use_gpu: bool,
    ) -> list[str]:
        if self.runtime == "docker":
            cmd = ["docker", "run", "--rm", "--security-opt", "label=disable"]
            if use_gpu:
                cmd.extend(["--gpus", "all"])
            cmd.extend(["-w", workdir])
            for mount in mounts:
                spec = f"{mount.source}:{mount.target}"
                if mount.read_only:
                    spec += ":ro"
                cmd.extend(["-v", spec])
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
            cmd.append(str(image_ref))
            cmd.extend(command)
            return cmd

        if self.runtime == "apptainer":
            if self.apptainer_bin is None:
                raise RuntimeError("Apptainer/Singularity runtime requested but no binary was found")
            cmd = [self.apptainer_bin, "exec", "--cleanenv", "--pwd", workdir]
            if use_gpu:
                cmd.append("--nv")
            for mount in mounts:
                spec = f"{mount.source}:{mount.target}"
                if mount.read_only:
                    spec += ":ro"
                cmd.extend(["--bind", spec])
            for key, value in env.items():
                cmd.extend(["--env", f"{key}={value}"])
            cmd.append(str(image_ref))
            cmd.extend(command)
            return cmd

        raise ValueError(f"Unsupported runtime: {self.runtime}")

    @staticmethod
    def _require_binary(name: str) -> None:
        if shutil.which(name):
            return
        raise RuntimeError(f"Required binary not found in PATH: {name}")

    @staticmethod
    def _resolve_apptainer_binary() -> str | None:
        for candidate in ("apptainer", "singularity"):
            if shutil.which(candidate):
                return candidate
        return None

    def _apptainer_image_override(self, profile: ProfileImage) -> Path | None:
        env_name = APPTAINER_IMAGE_OVERRIDE_ENV.get(profile.profile)
        if not env_name:
            return None

        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            return None

        override_path = Path(raw_value).expanduser().resolve()
        if not override_path.exists():
            raise FileNotFoundError(
                f"{env_name} points to a missing Apptainer image: {override_path}"
            )

        print(f"[build] Using prebuilt Apptainer image from {env_name}: {override_path}")
        return override_path
