"""
registry.py — Layer 2: model family detection and component safety map.

Single responsibility: given a model_id string, return a ModelProfile
that tells Layer 3 exactly what it is allowed to quantize and at what ceiling.

Nothing in this file touches disk, network, or Modal. It is pure logic.
That isolation means it can be unit-tested without GPU access.

⚠️ Assumption baked in: model_id follows HuggingFace convention
   `org/model-name`. The family is detected from the model-name portion
   (after the slash). If someone passes a local path like
   `/home/user/my-model`, detection falls through to UNKNOWN and raises.
   Add a local-path branch if that use case materialises.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from config import QuantMethod


# ---------------------------------------------------------------------------
# ModelProfile — what Layer 3 reads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProfile:
    """
    Immutable description of a model's quantization constraints.

    frozen=True: Layer 3 cannot accidentally mutate the registry's answer.
    If it tries, Python raises a FrozenInstanceError immediately.

    Fields:
        family          human-readable family name for logs and report headers
        is_multimodal   True → registry populates ignore_patterns with visual.*
        ignore_patterns passed verbatim to llm-compressor's ignore= argument
        max_safe_method the hardest quantization this family is validated for
                        — registry will refuse a config that exceeds this
        vit_params_b    approximate ViT parameter count in billions (for report)
        llm_hidden_dim  LLM backbone hidden dimension (for Bonsai compat check)
        notes           free-text warnings surfaced in the report
    """
    family:          str
    is_multimodal:   bool
    ignore_patterns: list[str]
    max_safe_method: QuantMethod
    vit_params_b:    Optional[float] = None
    llm_hidden_dim:  Optional[int]   = None
    notes:           list[str]       = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry rules — ordered, first match wins
# ---------------------------------------------------------------------------
#
# WHY ordered list of (pattern, profile) rather than a dict?
# A dict keyed by regex string would require iterating and compiling anyway.
# An ordered list makes priority explicit: more specific patterns come first.
# Example: "qwen3.*vl" must appear before "qwen.*vl" or Qwen3-VL would
# match the generic Qwen-VL rule and inherit the wrong constraints.
#
# Each entry: (compiled_regex, ModelProfile)
# The regex is matched against the lowercased model name (after the org slash).

_REGISTRY: list[tuple[re.Pattern, ModelProfile]] = [

    # ── Qwen3-VL family ─────────────────────────────────────────────────────
    # Must appear before generic Qwen-VL rule.
    # Same visual.* ignore pattern — architecture is compatible with Qwen2.5-VL.
    # Max safe method raised to W4A8 based on Qwen3 empirical study (arXiv 2505.02214)
    # which shows W4A8 <1% drop on MMLU for 8B+ models. 3B may differ — treat as ceiling.
    (
        re.compile(r"qwen3.*vl", re.IGNORECASE),
        ModelProfile(
            family          = "Qwen3-VL",
            is_multimodal   = True,
            ignore_patterns = ["re:.*lm_head", "re:.*visual.*"],
            max_safe_method = QuantMethod.W4A8,
            vit_params_b    = 0.675,
            llm_hidden_dim  = 2048,
            notes           = [
                "Qwen3-VL uses same ViT architecture as Qwen2.5-VL.",
                "W4A8 ceiling from Qwen3 quantization study (arXiv 2505.02214).",
                "Verify transformers>=4.51.0 for Qwen3-VL support.",
            ],
        ),
    ),

    # ── Qwen2.5-VL family ───────────────────────────────────────────────────
    # Primary target. Ignore pattern derived from production recipes:
    # Red Hat llm-compressor, official Qwen AWQ, community GPTQModel builds.
    # All converge on ignore=["re:.*lm_head", "re:.*visual.*"].
    #
    # "re:.*visual.*" matches:
    #   model.visual.patch_embed.*
    #   model.visual.blocks.*          (all 32 ViT transformer blocks)
    #   model.visual.merger.*          (the 2-layer MLP projector)
    # This is intentional — the merger is NOT separately configurable here
    # because no production evidence shows safely quantizing it independently.
    #
    # max_safe_method = W8A8 — raised after empirical validation of W4A16
    # on A10G (Ampere). W8A8 quantizes activations in addition to weights;
    # requires A100 for calibration (more VRAM for activation statistics).
    # W4A8 ceiling not yet validated for this architecture on Ampere GPUs.
    (
        re.compile(r"qwen2\.5.*vl", re.IGNORECASE),
        ModelProfile(
            family          = "Qwen2.5-VL",
            is_multimodal   = True,
            ignore_patterns = ["re:.*lm_head", "re:.*visual.*"],
            max_safe_method = QuantMethod.W8A8,
            vit_params_b    = 0.675,
            llm_hidden_dim  = 2048,
            notes           = [
                "Use llm-compressor==0.5.1 + transformers==4.51.3 for quantization.",
                "Use vLLM>=0.7.2 for inference — Marlin kernels active on Ampere.",
                "Projector (model.visual.merger) kept at FP32 — matched by visual.* regex.",
                "OCR and bbox grounding regress 5x faster than MMMU under aggressive quant.",
                "Keep merger at FP32, not BF16, for best bbox coordinate precision.",
                "W8A8 requires A100 for calibration — activation statistics need >24GB VRAM.",
            ],
        ),
    ),

    # ── Qwen2-VL family ─────────────────────────────────────────────────────
    # Predecessor to Qwen2.5-VL. Same architecture, same ignore rules.
    # Lower max_safe_method — less empirical validation at W4A16.
    (
        re.compile(r"qwen2.*vl", re.IGNORECASE),
        ModelProfile(
            family          = "Qwen2-VL",
            is_multimodal   = True,
            ignore_patterns = ["re:.*lm_head", "re:.*visual.*"],
            max_safe_method = QuantMethod.W4A16,
            vit_params_b    = 0.675,
            llm_hidden_dim  = 2048,
            notes           = [
                "Qwen2-VL ViT uses 2D-RoPE — same quantization sensitivity as Qwen2.5-VL.",
                "MBQ paper (CVPR 2025) validated W4A8 ViT on Qwen2-VL-7B — minor impact.",
            ],
        ),
    ),

    # ── Qwen2.5 text-only family ────────────────────────────────────────────
    # MUST appear after all VL rules — "qwen2.5" matches VL too without the
    # negative lookahead. The lookahead `(?!.*vl)` ensures this rule only fires
    # when "vl" is absent from the model name.
    #
    # WHY the lookahead matters:
    # Without it, "Qwen2.5-VL-3B" would match this rule if the VL rules above
    # somehow failed. That would give a text-only profile to a VLM → Layer 3
    # would quantize visual.* → projector collapses. The lookahead is a
    # defence-in-depth measure against future registry editing errors.
    (
        re.compile(r"qwen2\.5(?!.*vl)", re.IGNORECASE),
        ModelProfile(
            family          = "Qwen2.5-text",
            is_multimodal   = False,
            ignore_patterns = ["re:.*lm_head"],
            max_safe_method = QuantMethod.W4A8,
            llm_hidden_dim  = 2048,   # for 3B; 7B = 3584, 14B = 5120 — same family
            notes           = [
                "Qwen2.5 is the most quantization-tolerant family tested (arXiv benchmark).",
                "W4A8 ceiling validated; W4A4 loses ~10% MMLU at 3B scale.",
            ],
        ),
    ),

    # ── Qwen3 text-only family ───────────────────────────────────────────────
    (
        re.compile(r"qwen3(?!.*vl)", re.IGNORECASE),
        ModelProfile(
            family          = "Qwen3-text",
            is_multimodal   = False,
            ignore_patterns = ["re:.*lm_head"],
            max_safe_method = QuantMethod.W4A8,
            notes           = [
                "Qwen3 shows more PPL sensitivity at 3-bit than LLaMA3 (arXiv 2505.02214).",
                "Avoid AWQ w3a16 — PPL spikes from 10.4 to 23.8 on Qwen3-8B-Base.",
            ],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API — this is all Layer 3 calls
# ---------------------------------------------------------------------------

def get_profile(model_id: str) -> ModelProfile:
    """
    Resolve a ModelProfile from a HuggingFace model_id.

    Raises:
        ValueError   if the model_id matches no known family.
                     Fails fast — do not let unknown architectures reach Layer 3.

    Example:
        profile = get_profile("Qwen/Qwen2.5-VL-3B-Instruct")
        # → ModelProfile(family="Qwen2.5-VL", is_multimodal=True, ...)
    """
    # Extract model name — everything after the last slash, or the whole string
    model_name = model_id.split("/")[-1]

    for pattern, profile in _REGISTRY:
        if pattern.search(model_name):
            return profile

    # No match — fail loud, not silent
    known = [p.pattern for p, _ in _REGISTRY]
    raise ValueError(
        f"Unknown model family: '{model_id}'.\n"
        f"Registry patterns: {known}\n"
        f"Add a new entry to registry.py before running this model."
    )


def validate_config(model_id: str, requested_method: QuantMethod) -> None:
    """
    Cross-check a QuantConfig against the registry's safety ceiling.

    Layer 3 calls this before touching any weights.
    If the requested method exceeds the registry's validated ceiling,
    this raises — not warns — because pushing over-aggressively quantized
    weights to HuggingFace is worse than aborting the run.

    WHY raise rather than clamp silently?
    Silent clamping hides the fact that you asked for W4A8 and got W4A16.
    Your benchmark table would report W4A8 numbers that are actually W4A16.
    That's a research integrity problem, not just a software bug.
    """
    if requested_method == QuantMethod.NONE:
        return   # baseline run — nothing to validate

    profile = get_profile(model_id)

    # Ordinal comparison via explicit aggressiveness ranking.
    # WHY a dict rather than list.index(): if a new QuantMethod variant is
    # added to the enum but omitted here, the KeyError below fires with a
    # clear message — not a cryptic ValueError from list.index() mid-run.
    #
    # Aggressiveness: NONE < W4A16 < W8A8 < W4A8
    # W8A8 > W4A16 because it quantizes activations too, not just weights.
    # W4A8 is most aggressive: 4-bit weights AND 8-bit activations.
    aggressiveness = {
        QuantMethod.NONE:  0,
        QuantMethod.W4A16: 1,
        QuantMethod.W8A8:  2,
        QuantMethod.W4A8:  3,
    }

    for m in (requested_method, profile.max_safe_method):
        if m not in aggressiveness:
            raise KeyError(
                f"QuantMethod.{m.value} has no aggressiveness rank. "
                f"Add it to the aggressiveness dict in validate_config()."
            )

    requested_idx = aggressiveness[requested_method]
    ceiling_idx   = aggressiveness[profile.max_safe_method]

    if requested_idx > ceiling_idx:
        raise ValueError(
            f"Requested method {requested_method.value} exceeds the registry's "
            f"validated ceiling of {profile.max_safe_method.value} "
            f"for {profile.family}.\n"
            f"To proceed anyway, raise max_safe_method in registry.py "
            f"and document your empirical validation."
        )


def resolve_ignore_patterns(model_id: str, extra: list[str]) -> list[str]:
    """
    Merge registry defaults with any extra patterns from the config.

    Layer 3 calls this to get the final ignore list for llm-compressor.
    Extra patterns from QuantConfig.extra_ignore_patterns are appended
    after the registry defaults — they can only add restrictions, not remove.

    WHY append-only?
    Allowing subtraction (e.g. removing "re:.*visual.*") would let a user
    accidentally override the registry's safety knowledge. If you genuinely
    need to quantize the ViT, that's a deliberate research decision that
    belongs in the registry as a named profile, not a config override.
    """
    profile = get_profile(model_id)
    # Deduplicate while preserving order
    seen    = set()
    merged  = []
    for p in profile.ignore_patterns + extra:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    return merged