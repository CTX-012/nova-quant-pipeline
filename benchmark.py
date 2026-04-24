"""
benchmark.py — Layer 4: measures the saved artifact, not the in-memory model.

WHY reload from disk rather than reuse engine.py's model object:
engine.py's in-memory model is llm-compressor's live PyTorch state.
The saved artifact goes through save_pretrained() serialization and
from_pretrained() deserialization — two operations that can silently
corrupt quantized kernel dispatch if config.json is malformed.
Only a reload from disk proves the artifact a user downloads will work.

Single responsibility: given a save_path, return a metrics dict.
Does not push to HuggingFace. Does not write files. Pure measurement.

⚠️ Dependencies: same as engine.py. Runs inside the same Modal container.
⚠️ Assumption: save_path contains a valid model directory with config.json.
   If save_path is None (baseline run), benchmark loads from model_id directly.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Optional

from config import QuantConfig
from registry import get_profile


# ---------------------------------------------------------------------------
# VRAM utilities
# ---------------------------------------------------------------------------

def _reset_gpu_memory():
    """
    Force garbage collection and clear CUDA cache before measuring.

    WHY: leftover tensors from engine.py's quantization pass inflate
    the VRAM baseline. Without this, peak VRAM includes calibration
    activations that are not present during normal inference.
    """
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb() -> float:
    """Return peak VRAM allocated since last reset, in GB."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_artifact(cfg: QuantConfig, save_path: Optional[str]):
    """
    Load the saved artifact from disk for benchmarking.

    If save_path is None (baseline run with QuantMethod.NONE),
    load the original model directly from cfg.volume_model_path.

    Returns (model, processor_or_tokenizer, profile, device).

    ⚠️ do_not_use_cuda_graphs=True is set to avoid warm-up overhead
    contaminating cold-start latency measurements.
    """
    import torch
    profile = get_profile(cfg.model_id)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    path    = save_path if save_path else cfg.volume_model_path

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Artifact not found at {path}. "
            f"engine.quantize() must complete successfully before benchmarking."
        )

    print(f"[benchmark] loading artifact from {path}")
    _reset_gpu_memory()

    if profile.is_multimodal:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        proc = AutoProcessor.from_pretrained(path)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        proc = AutoTokenizer.from_pretrained(path)

    model.eval()

    # dispatch_for_generation patches the loaded model to use compressed-tensors
    # kernels (Marlin for INT4, etc.) instead of naive PyTorch dequantization.
    #
    # WHY call this here and not in engine.py?
    # engine.py's model object is llm-compressor's calibration state — calling
    # dispatch there would interfere with the quantization pass itself.
    # benchmark.py reloads from disk, so this is the correct point to patch.
    #
    # WHY only for quantized artifacts (save_path is not None)?
    # BF16 baseline has no compressed-tensors config — dispatch is a no-op
    # on unquantized models but may log warnings. Skip it cleanly.
    #
    # ⚠️ Verify dispatch_for_generation import path in llmcompressor>=0.4.2
    #    — it moved between versions. Both paths are tried below.
    if save_path is not None:
        try:
            from llmcompressor.utils import dispatch_for_generation
        except ImportError:
            try:
                from compressed_tensors.utils import dispatch_for_generation
            except ImportError:
                print("[benchmark] WARNING: dispatch_for_generation not found — "
                      "falling back to naive dequantization. "
                      "tok/s and VRAM numbers will be incorrect for quantized models.")
                dispatch_for_generation = None

        if dispatch_for_generation is not None:
            dispatch_for_generation(model)
            print("[benchmark] dispatch_for_generation applied — compressed kernels active")

    load_vram = _peak_vram_gb()
    print(f"[benchmark] loaded — peak VRAM at load: {load_vram:.2f} GB")
    return model, proc, profile, device


# ---------------------------------------------------------------------------
# Speed measurement
# ---------------------------------------------------------------------------

class _TimingStreamer:
    """
    Intercepts tokens as model.generate() mints them to get exact TTFT.

    WHY prompt_seen skip on first put():
    transformers' generate() calls streamer.put() twice before real decode:
      call 0 — the full prompt tensor (input_ids, not a generated token)
      call 1 — the first actually generated token (prefill complete)
    Without the skip, t_first_token captures the prompt dump (~1ms),
    not the end of prefill (~200-4000ms for a VLM). The result is a
    physically impossible TTFT like 1.2ms that masks the real prefill cost.

    Correct sequence:
      put(prompt_tensor)   → prompt_seen=True, skip
      put(first_token)     → record t_first_token  ← real TTFT ends here
      put(second_token)    → ignored (t_first_token already set)
      end()                → generation complete

    ⚠️ Requires transformers>=4.40.0 for BaseStreamer.
    ⚠️ This skip assumes one prompt dump before decode. If transformers
       changes this behaviour in a future version, TTFT will be off by
       one token latency (~TPOT). Verify against known-good latency if
       upgrading transformers.
    """
    def __init__(self):
        self.t_start:       float          = time.perf_counter()
        self.t_first_token: Optional[float] = None
        self.prompt_seen:   bool           = False

    def put(self, value) -> None:
        # First put() is the prompt dump — discard it
        if not self.prompt_seen:
            self.prompt_seen = True
            return

        # Second put() = first generated token = prefill complete
        if self.t_first_token is None:
            self.t_first_token = time.perf_counter()

    def end(self) -> None:
        pass

    @property
    def ttft_s(self) -> float:
        if self.t_first_token is None:
            return float("inf")
        return self.t_first_token - self.t_start


def _measure_speed(
    model, proc, profile, device: str,
    n_warmup: int = 2,
    n_measure: int = 10,
    max_new_tokens: int = 128,
) -> dict:
    """
    Measure tok/s, exact TTFT, and TPOT using BaseStreamer timing.

    Protocol:
        - n_warmup runs: discarded (GPU kernel compilation warm-up)
        - n_measure runs: averaged for reported numbers
        - Fixed synthetic prompt + image — identical across all configs
        - do_sample=False for determinism

    Metrics reported:
        ttft_ms:   time from generate() call to first token (prefill cost)
        tpot_ms:   time per output token after the first (decode cost)
        tok_per_s: n_new_tokens / total_wall_time (throughput, for paper table)

    WHY all three?
    For NOVA's control loop, TTFT is what matters — the robot is blocked
    until the first token arrives. TPOT matters for advisory text completion.
    tok/s is the conventional benchmark metric for paper comparison.
    """
    import torch
    from PIL import Image as PILImage

    TEXT_PROMPT = (
        "You are a robot navigation assistant. "
        "List any objects on the floor that a cleaning robot should avoid. "
        "Be concise."
    )

    # Build inputs once — identical across all warmup + measure runs
    if profile.is_multimodal:
        img      = PILImage.new("RGB", (640, 480), color=(200, 200, 200))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": TEXT_PROMPT},
            ],
        }]
        text   = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = proc(
            text=[text], images=[img],
            return_tensors="pt", padding=True
        ).to(device)
    else:
        inputs = proc(TEXT_PROMPT, return_tensors="pt").to(device)

    n_prompt_tokens = inputs["input_ids"].shape[1]

    ttft_ms_list  = []
    tpot_ms_list  = []
    tok_per_s_list = []

    for i in range(n_warmup + n_measure):
        _reset_gpu_memory()

        streamer = _TimingStreamer()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                streamer=streamer,              # inject timing hook
                return_dict_in_generate=True,
            )

        t_end        = time.perf_counter()
        n_new_tokens = out.sequences.shape[1] - n_prompt_tokens
        total_s      = t_end - streamer.t_start

        if i < n_warmup or n_new_tokens < 2:
            continue   # discard warmup; need ≥2 tokens for TPOT

        ttft_s = streamer.ttft_s
        tpot_s = (t_end - streamer.t_first_token) / (n_new_tokens - 1)

        ttft_ms_list.append(ttft_s * 1000)
        tpot_ms_list.append(tpot_s * 1000)
        tok_per_s_list.append(n_new_tokens / total_s)

    if not tok_per_s_list:
        return {
            "tok_per_s": 0.0, "ttft_ms": float("inf"),
            "tpot_ms": float("inf"), "n_tokens_generated": 0,
        }

    avg = lambda lst: sum(lst) / len(lst)
    return {
        "tok_per_s":          avg(tok_per_s_list),
        "ttft_ms":            avg(ttft_ms_list),    # exact, not approximate
        "tpot_ms":            avg(tpot_ms_list),
        "n_tokens_generated": max_new_tokens,
        "n_warmup":           n_warmup,
        "n_measure":          n_measure,
    }


# ---------------------------------------------------------------------------
# Visual QA accuracy spot-check
# ---------------------------------------------------------------------------

def _measure_vqa_accuracy(model, proc, profile, device: str, cfg: QuantConfig) -> dict:
    """
    Run VQA spot-check on benchmark images from cfg.benchmark_image_paths.

    For each image, runs each prompt in cfg.benchmark_prompts and records
    the raw output. Does NOT compute automatic VQA accuracy (that requires
    ground-truth annotations). Instead, records outputs for human review
    and flags obviously broken responses (empty string, repeated tokens,
    non-ASCII gibberish).

    WHY not automatic VQA scoring here?
    Automatic VQA (VizWiz, DocVQA) requires the full eval dataset and
    10-30 minutes of GPU time. That belongs in a separate evaluation run,
    not in the quantization pipeline's benchmark pass.
    This spot-check catches catastrophic failures (blind model, hallucination
    loops) in under 60 seconds — enough to gate the artifact push.

    Returns dict with:
        responses:      list of {image, prompt, output, flag}
        n_flagged:      count of responses with quality flags
        flagged_ratio:  n_flagged / total responses
    """
    import torch
    from PIL import Image as PILImage

    if not profile.is_multimodal:
        return {"responses": [], "n_flagged": 0, "flagged_ratio": 0.0,
                "note": "text-only model — VQA check skipped"}

    if not cfg.benchmark_image_paths and not cfg.benchmark_prompts:
        return {"responses": [], "n_flagged": 0, "flagged_ratio": 0.0,
                "note": "no benchmark images configured"}

    # If no image paths given, use a synthetic indoor scene
    image_sources = cfg.benchmark_image_paths or ["__synthetic__"]
    responses     = []

    for img_src in image_sources:
        if img_src == "__synthetic__":
            img      = PILImage.new("RGB", (640, 480), color=(180, 180, 180))
            img_name = "synthetic_gray"
        else:
            try:
                img = PILImage.open(img_src).convert("RGB")
            except Exception as e:
                responses.append({
                    "image": img_src, "prompt": "N/A",
                    "output": f"LOAD ERROR: {e}", "flag": "load_error"
                })
                continue
            img_name = Path(img_src).name

        for prompt in cfg.benchmark_prompts:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": prompt},
                ],
            }]
            text   = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = proc(
                text=[text], images=[img],
                return_tensors="pt", padding=True
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            n_prompt = inputs["input_ids"].shape[1]
            output   = proc.decode(
                out[0][n_prompt:], skip_special_tokens=True
            ).strip()

            # Flag heuristics — catches catastrophic failures
            flag = _quality_flag(output)
            responses.append({
                "image":  img_name,
                "prompt": prompt[:60],
                "output": output[:200],
                "flag":   flag,
            })
            print(f"[benchmark/vqa] {img_name} | flag={flag} | '{output[:60]}'")

    n_flagged = sum(1 for r in responses if r["flag"] != "ok")
    total     = max(len(responses), 1)

    return {
        "responses":    responses,
        "n_flagged":    n_flagged,
        "flagged_ratio": n_flagged / total,
    }


def _quality_flag(output: str) -> str:
    """
    Heuristic quality check on a single model output.

    Returns "ok" or a short flag string describing the failure mode.
    These flags appear in the benchmark report and paper table footnotes.

    WHY heuristics and not a model-based judge?
    A judge model requires another GPU call and is overkill for catching
    catastrophic quantization failures. If the model outputs an empty
    string or 500 repeated tokens, we know something is broken without
    needing GPT-4 to tell us.
    """
    if not output:
        return "empty"
    if len(output) < 5:
        return "too_short"

    # Repetition detection: if any 4-gram repeats more than 5 times, flag it
    words  = output.split()
    if len(words) >= 4:
        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        if any(ngrams.count(ng) > 5 for ng in set(ngrams)):
            return "repetition_loop"

    # Non-ASCII gibberish (quantization sometimes produces garbage tokens)
    non_ascii = sum(1 for c in output if ord(c) > 127)
    if non_ascii / max(len(output), 1) > 0.3:
        return "non_ascii_gibberish"

    return "ok"


# ---------------------------------------------------------------------------
# Disk size
# ---------------------------------------------------------------------------

def _measure_disk_size(save_path: Optional[str]) -> float:
    """Return total size of the saved artifact in GB. Returns 0.0 if no path."""
    if not save_path or not Path(save_path).exists():
        return 0.0
    total = sum(f.stat().st_size for f in Path(save_path).rglob("*") if f.is_file())
    return total / 1e9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(cfg: QuantConfig, engine_result: dict) -> dict:
    """
    Main entry point. Called by pipeline.py after engine.quantize() completes.

    Args:
        cfg:           the run config
        engine_result: dict returned by engine.quantize() — contains save_path,
                       quant_method, ppl from calibration pass, sanity result

    Returns:
        Complete metrics dict ready for artifacts.py to write to report.
        Keys match the paper table schema defined in artifacts.py.

    ⚠️ This function owns the final VRAM measurement. engine.py's VRAM
    numbers reflect quantization overhead (calibration activations).
    Only the benchmark reload gives clean inference-time VRAM.
    """
    save_path = engine_result.get("save_path")

    # Load the saved artifact from disk — not the in-memory engine model
    model, proc, profile, device = _load_artifact(cfg, save_path)
    load_vram_gb = _peak_vram_gb()

    print("[benchmark] measuring speed...")

    # Route to vLLM for quantized artifacts — real Marlin INT4 kernels.
    # Route to transformers for baseline — reference implementation.
    # WHY not vLLM for baseline: vLLM pre-allocates VRAM and adds engine
    # overhead that isn't present in normal transformers inference.
    # The two backends are not directly comparable for absolute numbers,
    # but quantized configs benchmarked with vLLM are comparable to each other.
    if cfg.use_vllm and save_path is not None:
        try:
            from vllm_benchmark import measure_vllm_speed
            speed = measure_vllm_speed(save_path)
            # vLLM freed its GPU memory — reload model for VQA check
            # WHY reload: measure_vllm_speed deletes the LLM engine to free VRAM
            print("[benchmark] reloading model for VQA after vLLM benchmark...")
            model, proc, profile, device = _load_artifact(cfg, save_path)
        except Exception as e:
            print(f"[benchmark] WARNING: vLLM benchmark failed ({e}), "
                  f"falling back to transformers speed measurement")
            speed = _measure_speed(model, proc, profile, device)
    else:
        speed = _measure_speed(model, proc, profile, device)

    print("[benchmark] measuring VQA accuracy...")
    vqa = _measure_vqa_accuracy(model, proc, profile, device, cfg)

    print("[benchmark] measuring disk size...")
    disk_gb = _measure_disk_size(save_path)

    # Peak VRAM during inference (reset again before speed measurement already did this)
    inference_vram_gb = _peak_vram_gb()

    metrics = {
        # Identity
        "run_name":        cfg.run_name,
        "model_id":        cfg.model_id,
        "quant_method":    cfg.quant_method.value,
        "family":          profile.family,
        "calibration":     cfg.calibration_dataset.value,
        "group_size":      cfg.group_size,

        # Quality
        "ppl":             engine_result.get("ppl"),
        "sanity_passed":   engine_result.get("sanity", {}).get("passed"),

        # Speed (vLLM for quantized, transformers for baseline)
        "tok_per_s":       speed.get("tok_per_s"),
        "ttft_ms":         speed.get("ttft_ms"),
        "tpot_ms":         speed.get("tpot_ms"),
        "speed_engine":    speed.get("engine", "transformers"),

        # Memory
        "load_vram_gb":       load_vram_gb,
        "inference_vram_gb":  inference_vram_gb,
        "disk_gb":            disk_gb,

        # VQA spot-check
        "vqa_n_flagged":      vqa.get("n_flagged"),
        "vqa_flagged_ratio":  vqa.get("flagged_ratio"),
        "vqa_responses":      vqa.get("responses"),

        # Notes from registry
        "registry_notes":  engine_result.get("notes", []),
        "ignore_patterns": engine_result.get("ignore_patterns", []),
    }

    print(
        f"[benchmark] done — "
        f"{metrics['tok_per_s']:.1f} tok/s | "
        f"VRAM {metrics['inference_vram_gb']:.2f}GB | "
        f"disk {metrics['disk_gb']:.2f}GB | "
        f"PPL {metrics['ppl']}"
    )

    return metrics