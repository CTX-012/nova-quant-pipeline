"""
config.py — Layer 1: single source of truth for the pipeline.

Every other layer imports from here. Nothing else imports from other layers
at config time. This isolation means you can change a run by editing one file
and re-running — no hunting through engine.py or benchmark.py.

RunPod deployment:
  - HF_TOKEN must be set as environment variable before running
  - volume_mount defaults to /workspace (RunPod network volume)
  - All paths derive from volume_mount
  - No Modal dependencies in this file
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuantMethod(str, Enum):
    """
    W4A16 = 4-bit weights, 16-bit activations (weight-only, fastest)
    W8A8  = 8-bit weights + 8-bit activations (near-lossless quality)
    W4A8  = 4-bit weights + 8-bit activations (most aggressive)
    NONE  = no quantization — baseline measurement run
    """
    W4A16 = "W4A16"
    W8A8  = "W8A8"
    W4A8  = "W4A8"
    NONE  = "NONE"


class CalibrationDataset(str, Enum):
    """
    GENERIC    = ultrachat_200k text samples
    NOVA_FLOOR = NOVA robot floor frames (domain-specific)
    MIXED      = 50% generic + 50% NOVA frames
    """
    GENERIC    = "generic"
    NOVA_FLOOR = "nova_floor"
    MIXED      = "mixed"


# ---------------------------------------------------------------------------
# Core config dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    """
    One instance fully describes one pipeline run.
    All paths derive from volume_mount — change that one field
    to move the entire pipeline to a different storage location.
    """

    # --- Identity ---
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    run_name: str = "qwen25vl3b_w4a16_generic"

    # --- Quantization ---
    quant_method: QuantMethod = QuantMethod.W4A16
    group_size: int = 128
    calibration_dataset: CalibrationDataset = CalibrationDataset.GENERIC
    calibration_samples: int = 128

    # --- Safety: what NOT to quantize ---
    extra_ignore_patterns: list[str] = field(default_factory=list)

    # --- Quality gate ---
    # None = disabled (recommended for RunPod parallel runs)
    # Float = abort if PPL exceeds baseline_ppl * threshold
    ppl_abort_threshold: Optional[float] = None

    # --- Storage (RunPod) ---
    # volume_mount: RunPod network volume mount point
    # All derived paths use this as root
    volume_mount: str = "/workspace"

    # --- Benchmark backend ---
    # use_vllm=True: quantized models benchmarked with vLLM (Marlin kernels)
    # use_vllm=False: all models benchmarked with transformers (debugging)
    use_vllm: bool = True

    # venv paths on RunPod network volume
    venv_quant: str = "/workspace/venv_quant"
    venv_vllm:  str = "/workspace/venv_vllm"

    # --- Output ---
    hf_org: str = "Mohaaxa"
    push_to_hub: bool = True
    generate_report: bool = True

    # --- Benchmark images ---
    # Empty = use synthetic gray image for VQA spot-check
    benchmark_image_paths: list[str] = field(default_factory=list)
    benchmark_prompts: list[str] = field(default_factory=lambda: [
        "Describe what you see in this image.",
        "List any hazards visible on the floor.",
        "What objects are on the floor that a robot should avoid?",
    ])

    # ---------------------------------------------------------------------------
    # Derived properties
    # ---------------------------------------------------------------------------

    @property
    def base_model_path(self) -> str:
        """Downloaded base model weights — shared across all runs."""
        return f"{self.volume_mount}/models/qwen25vl3b_base"

    @property
    def artifact_path(self) -> str:
        """Where quantized weights are saved."""
        return f"{self.volume_mount}/artifacts/{self.run_name}_{self.quant_method.value}"

    @property
    def report_path(self) -> str:
        """Shared append-only JSONL report."""
        return f"{self.volume_mount}/reports/pipeline_results.jsonl"

    @property
    def nova_frames_path(self) -> str:
        """NOVA calibration frames directory."""
        return f"{self.volume_mount}/calibration/nova_frames"

    @property
    def hf_repo_id(self) -> str:
        """HuggingFace repo for this run's artifact."""
        model_short = self.model_id.split("/")[-1]
        return f"{self.hf_org}/{model_short}-{self.quant_method.value}-{self.calibration_dataset.value}"

    def summary(self) -> str:
        return (
            f"{self.model_id} | {self.quant_method.value} | "
            f"cal={self.calibration_dataset.value}"
        )


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

def baseline_config() -> QuantConfig:
    """BF16 baseline — no quantization, just benchmarks."""
    return QuantConfig(
        run_name="qwen25vl3b_bf16_baseline",
        quant_method=QuantMethod.NONE,
        push_to_hub=False,
        ppl_abort_threshold=None,
        use_vllm=False,   # baseline uses transformers — reference implementation
    )


def w4a16_generic_config() -> QuantConfig:
    """W4A16 with generic text calibration."""
    return QuantConfig(
        run_name="qwen25vl3b_w4a16_generic",
        quant_method=QuantMethod.W4A16,
        calibration_dataset=CalibrationDataset.GENERIC,
        ppl_abort_threshold=None,
    )


def w4a16_nova_config() -> QuantConfig:
    """W4A16 with NOVA floor frame calibration — research contribution."""
    return QuantConfig(
        run_name="qwen25vl3b_w4a16_nova",
        quant_method=QuantMethod.W4A16,
        calibration_dataset=CalibrationDataset.NOVA_FLOOR,
        ppl_abort_threshold=None,
    )


def w8a8_config() -> QuantConfig:
    """W8A8 — near-lossless compression, 3x speedup."""
    return QuantConfig(
        run_name="qwen25vl3b_w8a8_generic",
        quant_method=QuantMethod.W8A8,
        ppl_abort_threshold=None,
    )