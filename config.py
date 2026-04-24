"""
config.py — Layer 1: single source of truth for the pipeline.

Every other layer imports from here. Nothing else imports from other layers
at config time. This isolation means you can change a run by editing one file
and re-running — no hunting through engine.py or benchmark.py.

⚠️ Assumption baked in: Modal secrets must contain HF_TOKEN.
   If that secret is missing, Layer 6 (pipeline.py) will fail at auth,
   not here — error surfaces late. Add a validator if that becomes painful.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums — constrained vocabularies prevent typo-driven bugs
# ---------------------------------------------------------------------------

class QuantMethod(str, Enum):
    """
    Supported quantization schemes.

    WHY an enum rather than a plain string:
    If you mistype "gptq" as "gpqt" in a dict, you get a silent wrong
    behaviour. An enum raises immediately at assignment time.

    W4A16  = 4-bit weights, 16-bit activations (weight-only, fastest path)
    W8A8   = 8-bit weights + 8-bit activations (requires static calibration)
    W4A8   = 4-bit weights + 8-bit activations (MQuant-style, most aggressive)
    NONE   = no quantization — used for the baseline measurement run
    """
    W4A16 = "W4A16"
    W8A8  = "W8A8"
    W4A8  = "W4A8"
    NONE  = "NONE"


class CalibrationDataset(str, Enum):
    """
    Calibration dataset source.

    GENERIC    = ShareGPT text samples (our existing AWQ calibration approach)
    NOVA_FLOOR = NOVA robot frames — domain-specific, stored in Modal volume
    MIXED      = 50% ShareGPT + 50% NOVA frames

    WHY this matters: AWQ paper shows instruction-following calibration
    outperforms random Wikipedia for instruction-tuned models. Domain-specific
    calibration (NOVA frames) is the research contribution — it may or may not
    beat generic calibration, which is precisely what the benchmark measures.
    """
    GENERIC    = "generic"
    NOVA_FLOOR = "nova_floor"
    MIXED      = "mixed"


class GPUTier(str, Enum):
    """
    Modal GPU target. Determines cost and availability.

    A10G  = ~$1.10/hr — sufficient for 3B quantization calibration
    A100  = ~$3.00/hr — required for 7B+ or W4A8 calibration (more VRAM)

    ⚠️ Verify Modal pricing — rates change. These are approximate as of Apr 2026.
    """
    A10G  = "A10G"
    A100  = "A100-40GB"


# ---------------------------------------------------------------------------
# Core config dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    """
    One instance of this class fully describes one pipeline run.

    Design decision: flat dataclass, not nested YAML.
    Reason: nested configs look clean but make programmatic override
    (e.g. sweep across methods) messy. A flat dataclass is trivial to
    copy-and-modify in a loop.

    What breaks if assumptions are wrong:
    - model_id must be a valid HuggingFace repo the HF_TOKEN can access.
      If private and token lacks read access → AuthenticationError at download.
    - calibration_samples=128 follows AWQ convention. Lower = faster but
      noisier scale estimates. Higher = slower, diminishing returns past 512.
    - ppl_abort_threshold: if None, pipeline never aborts on quality grounds.
      Set a value in production. Leaving it None during initial exploration
      is acceptable — you're learning the baseline first.
    """

    # --- Identity ---
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    run_name: str = "qwen25vl3b_w4a16_baseline"

    # --- Quantization ---
    quant_method: QuantMethod = QuantMethod.W4A16
    group_size: int = 128           # AWQ/GPTQ standard; smaller = better quality, larger model
    calibration_dataset: CalibrationDataset = CalibrationDataset.GENERIC
    calibration_samples: int = 128

    # --- Safety: what NOT to quantize ---
    # Registry (Layer 2) will populate this automatically based on model_id.
    # You can override here for experimentation, but the registry default
    # is always the safe choice.
    extra_ignore_patterns: list[str] = field(default_factory=list)

    # --- Quality gate ---
    # If perplexity after quantization exceeds this multiple of BF16 baseline,
    # the pipeline aborts before pushing any weights.
    # Example: 1.10 means "abort if PPL is more than 10% worse than BF16".
    # None = no gate (exploration mode).
    ppl_abort_threshold: Optional[float] = 1.10

    # --- Infrastructure ---
    gpu: GPUTier = GPUTier.A10G
    modal_volume_name: str = "nova-quant-pipeline-v1"
    modal_secret_name: str = "huggingface-secret"
    timeout_seconds: int = 3600     # 1hr; increase to 7200 for 7B W4A8

    # Volume mount point — set by pipeline.py to match Modal's VOLUME_MOUNT.
    # WHY a field not a constant: makes paths testable locally without Modal
    # by pointing volume_mount at a temp directory.
    volume_mount: str = "/cache"

    # --- Benchmark backend ---
    # WHY use_vllm flag:
    # Baseline (BF16) uses transformers — reference implementation.
    # Quantized artifacts use vLLM — dispatches to Marlin INT4 kernels.
    # The flag lets you disable vLLM for debugging without changing backend code.
    use_vllm: bool = True

    # --- Output ---
    hf_org: str = "Mohaaxa"
    push_to_hub: bool = True
    push_to_volume: bool = True
    generate_report: bool = True

    # --- Benchmark ---
    # Images used during benchmarking. Paths relative to Modal volume root.
    # If empty, benchmark uses synthetic text-only prompts (no VQA).
    benchmark_image_paths: list[str] = field(default_factory=list)
    benchmark_prompts: list[str] = field(default_factory=lambda: [
        "Describe what you see in this image.",
        "List any hazards visible on the floor.",
        "What objects are on the floor that a robot should avoid?",
    ])

    # ---------------------------------------------------------------------------
    # Derived properties — computed, not stored
    # ---------------------------------------------------------------------------

    @property
    def hf_repo_id(self) -> str:
        """
        Canonical HF repo name for this run's output.
        Example: Mohaaxa/Qwen2.5-VL-3B-Instruct-W4A16-nova
        """
        model_short = self.model_id.split("/")[-1]
        method      = self.quant_method.value
        cal_tag     = self.calibration_dataset.value
        return f"{self.hf_org}/{model_short}-{method}-{cal_tag}"

    @property
    def volume_model_path(self) -> str:
        """Path inside the Modal volume where weights land."""
        return f"{self.volume_mount}/models/{self.run_name}"

    @property
    def volume_report_path(self) -> str:
        """Path inside the Modal volume where the JSON report lands."""
        return f"{self.volume_mount}/reports/{self.run_name}.json"

    def summary(self) -> str:
        """Human-readable one-liner for logs."""
        return (
            f"{self.model_id} | {self.quant_method.value} | "
            f"cal={self.calibration_dataset.value} | gpu={self.gpu.value}"
        )


# ---------------------------------------------------------------------------
# Preset configs — copy-paste starting points, not magic globals
# ---------------------------------------------------------------------------

def baseline_config() -> QuantConfig:
    """BF16 baseline — no quantization, just benchmarks. Run this first."""
    return QuantConfig(
        run_name="qwen25vl3b_bf16_baseline",
        quant_method=QuantMethod.NONE,
        push_to_hub=False,          # don't push BF16 — it's already on HF
        ppl_abort_threshold=None,   # no gate: this IS the baseline
    )


def w4a16_generic_config() -> QuantConfig:
    """Standard W4A16, generic calibration. The safe first quantization run."""
    return QuantConfig(
        run_name="qwen25vl3b_w4a16_generic",
        quant_method=QuantMethod.W4A16,
        calibration_dataset=CalibrationDataset.GENERIC,
    )


def w4a16_nova_config() -> QuantConfig:
    """W4A16 with domain-specific NOVA calibration. Research contribution."""
    return QuantConfig(
        run_name="qwen25vl3b_w4a16_nova",
        quant_method=QuantMethod.W4A16,
        calibration_dataset=CalibrationDataset.NOVA_FLOOR,
        benchmark_image_paths=[
            "/calibration/nova_frames/frame_001.jpg",
            "/calibration/nova_frames/frame_002.jpg",
        ],
    )


def w8a8_config() -> QuantConfig:
    """W8A8 — higher quality, lower speed gain. Requires A100 for calibration."""
    return QuantConfig(
        run_name="qwen25vl3b_w8a8_generic",
        quant_method=QuantMethod.W8A8,
        gpu=GPUTier.A100,
        timeout_seconds=5400,
        ppl_abort_threshold=1.15,
    )