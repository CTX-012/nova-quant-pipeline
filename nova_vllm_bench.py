"""
vllm_benchmark.py — vLLM-based speed measurement for quantized artifacts.

WHY a separate module not integrated into benchmark.py:
vLLM's LLM class takes full ownership of the GPU — it pre-allocates memory,
manages KV cache, and cannot coexist with a transformers model in the same
process. Separating vLLM measurement into its own module allows benchmark.py
to call it as a subprocess-style function with a clean GPU state.

WHY vLLM for quantized models, transformers for baseline:
- Baseline (BF16): transformers is the reference implementation. vLLM would
  introduce engine overhead that isn't present in production transformers usage.
- Quantized (W4A16): vLLM dispatches compressed-tensors pack-quantized artifacts
  to Marlin INT4 GEMM kernels automatically. transformers falls back to software
  dequantization. Only vLLM gives the real hardware throughput number.

This module is called by benchmark.py when cfg.use_vllm=True and a save_path
exists (i.e. the artifact is quantized, not baseline).

⚠️ Requires vllm>=0.7.2 in the Modal image.
⚠️ vLLM V1 engine requires CUDA compute capability >=8.0 (Ampere+).
   A10G is Ampere (8.6) — compatible. T4 is Turing (7.5) — falls back to V0.
⚠️ vLLM pre-allocates gpu_memory_utilization fraction of VRAM at startup.
   VRAM reported here is not comparable to transformers load_vram — vLLM
   reserves a KV cache buffer even if it's unused. Note this in paper.
"""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Optional


# Fixed benchmark prompt — identical to benchmark.py for cross-backend comparability
TEXT_PROMPT = (
    "You are a robot navigation assistant. "
    "List any objects on the floor that a cleaning robot should avoid. "
    "Be concise."
)

# Fixed image: synthetic 640×480 gray — same as benchmark.py
def _make_benchmark_image_b64() -> str:
    """
    Generate a synthetic gray image and return as base64 PNG.

    vLLM's multimodal API accepts images as base64-encoded strings
    in the OpenAI chat format. PIL Image objects are not accepted directly.
    """
    from PIL import Image as PILImage
    img    = PILImage.new("RGB", (640, 480), color=(200, 200, 200))
    buf    = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def measure_vllm_speed(
    save_path: str,
    n_warmup: int = 2,
    n_measure: int = 10,
    max_new_tokens: int = 128,
    gpu_memory_utilization: float = 0.85,
) -> dict:
    """
    Load a quantized artifact with vLLM and measure real INT4 throughput.

    vLLM automatically detects the compressed-tensors quantization_config
    in config.json and dispatches to Marlin kernels for W4A16 artifacts.
    No manual kernel selection needed.

    Args:
        save_path:               path to the quantized model directory on volume
        n_warmup:                warmup runs discarded from measurement
        n_measure:               runs averaged for reported metrics
        max_new_tokens:          tokens to generate per measurement run
        gpu_memory_utilization:  fraction of GPU VRAM vLLM pre-allocates

    Returns:
        dict with tok_per_s, ttft_ms, tpot_ms, engine_info

    ⚠️ TTFT from vLLM is measured differently than TimingStreamer:
    vLLM reports time_to_first_token_ms in RequestOutput.metrics.
    This is exact prefill time measured server-side, not affected by
    the prompt-dump bug we fixed in TimingStreamer. It is the ground truth.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.multimodal.utils import encode_image_base64
    except ImportError:
        raise ImportError(
            "vllm not installed. Add 'vllm>=0.7.2' to the Modal image pip_install list."
        )

    if not Path(save_path).exists():
        raise FileNotFoundError(
            f"Artifact not found at {save_path}. "
            "engine.quantize() must complete before vllm_benchmark runs."
        )

    print(f"[vllm_benchmark] initializing engine from {save_path}")
    print(f"[vllm_benchmark] gpu_memory_utilization={gpu_memory_utilization}")

    # vLLM engine initialization — this pre-allocates VRAM and compiles kernels
    # WHY enforce_eager=False: allows CUDA graph capture for faster decode
    # WHY limit_mm_per_prompt=1: we benchmark single-image inference (NOVA use case)
    llm = LLM(
        model=save_path,
        max_model_len=4096,         # sufficient for our benchmark prompt + 128 output
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,        # enable CUDA graphs for realistic decode speed
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,            # deterministic — same as do_sample=False
        max_tokens=max_new_tokens,
    )

    # Build the multimodal prompt in vLLM's expected format
    img_b64  = _make_benchmark_image_b64()
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
            {"type": "text", "text": TEXT_PROMPT},
        ],
    }]

    # Apply chat template via vLLM's tokenizer
    # WHY use_tqdm=False: suppress per-request progress bars in benchmark loop
    print(f"[vllm_benchmark] warming up ({n_warmup} runs)...")

    ttft_ms_list   = []
    tpot_ms_list   = []
    tok_per_s_list = []

    for i in range(n_warmup + n_measure):
        t_start  = time.perf_counter()
        outputs  = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        t_end    = time.perf_counter()
        output   = outputs[0]

        if i < n_warmup:
            continue

        n_out    = len(output.outputs[0].token_ids)
        total_s  = t_end - t_start

        if n_out < 2:
            continue

        # Extract exact TTFT and TPOT from vLLM's internal Unix timestamps.
        # vLLM tracks arrival_time, first_token_time, finished_time server-side.
        # These are more accurate than client-side perf_counter() because they
        # exclude Python/gRPC overhead and measure the GPU compute directly.
        #
        # WHY not time_to_first_token_ms:
        # That field was removed in vLLM >= 0.6. The correct approach is to
        # subtract Unix timestamps directly from RequestOutputMetrics.
        #
        # ⚠️ first_token_time is None if the request produced 0 tokens.
        #    Guard against this with the n_out < 2 check above.
        metrics = output.metrics
        if (metrics
                and hasattr(metrics, "first_token_time")
                and metrics.first_token_time is not None
                and metrics.arrival_time is not None
                and metrics.finished_time is not None):
            ttft_s  = metrics.first_token_time - metrics.arrival_time
            tpot_s  = (metrics.finished_time - metrics.first_token_time) / max(n_out - 1, 1)
            ttft_ms = ttft_s * 1000
            tpot_ms = tpot_s * 1000
        else:
            # Fallback to wall-clock approximation if metrics unavailable
            # This produces identical TTFT=TPOT — flag in report
            print(f"[vllm_benchmark] WARNING: metrics timestamps unavailable, "
                  f"using wall-clock fallback (TTFT=TPOT will be equal)")
            ttft_ms = (total_s / n_out) * 1000
            tpot_ms = ttft_ms

        tok_per_s = n_out / total_s

        ttft_ms_list.append(ttft_ms)
        tpot_ms_list.append(tpot_ms)
        tok_per_s_list.append(tok_per_s)

        print(f"[vllm_benchmark] run {i-n_warmup+1}/{n_measure}: "
              f"{tok_per_s:.1f} tok/s | TTFT={ttft_ms:.0f}ms | TPOT={tpot_ms:.1f}ms")

    if not tok_per_s_list:
        return {
            "tok_per_s": 0.0, "ttft_ms": float("inf"),
            "tpot_ms": float("inf"), "engine": "vllm", "error": "no valid runs"
        }

    avg = lambda lst: sum(lst) / len(lst)

    result = {
        "tok_per_s":  avg(tok_per_s_list),
        "ttft_ms":    avg(ttft_ms_list),
        "tpot_ms":    avg(tpot_ms_list),
        "engine":     "vllm",
        "n_warmup":   n_warmup,
        "n_measure":  n_measure,
    }

    print(f"[vllm_benchmark] result: {result['tok_per_s']:.1f} tok/s | "
          f"TTFT={result['ttft_ms']:.0f}ms | TPOT={result['tpot_ms']:.1f}ms")

    # Explicitly delete the engine to free VRAM before returning
    # WHY: if benchmark.py runs VQA checks after this, it needs GPU memory
    del llm
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    return result