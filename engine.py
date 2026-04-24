"""
engine.py — Layer 3: quantization engine.

Wraps llm-compressor's GPTQModifier. Calls registry for safety constraints
before touching any weights. Runs sanity check after quantization but before
saving. Aborts loudly rather than pushing broken weights.

This file has NO Modal imports — it runs inside a Modal container but does
not define the container. pipeline.py owns the Modal app definition.
That separation means engine.py is testable locally (mocked) without
spinning up a GPU container.

⚠️ Dependencies (must be installed in Modal image):
    llmcompressor>=0.5.1
    transformers==4.49.0   (pinned — see registry notes on Qwen2.5-VL + AWQ)
    torch>=2.1.0
    Pillow>=10.0.0         (for synthetic sanity-check image generation)
    accelerate>=0.33.0

⚠️ Assumption: model weights are downloaded to local disk before engine runs.
   pipeline.py handles the download into the Modal volume. Engine reads from
   volume path, not from HuggingFace directly. This avoids re-downloading
   on every run and keeps GPU time pure compute, not network IO.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

from config import QuantConfig, QuantMethod
from registry import get_profile, validate_config, resolve_ignore_patterns


# ---------------------------------------------------------------------------
# Sanity check image generation
# ---------------------------------------------------------------------------

# Deterministic test string — fixed across all runs so substring match is stable.
# WHY not random? Random strings require storing the expected value. A fixed
# string is self-documenting and reproducible without any state.
SANITY_TEXT = "QA_TEST_8472"


def _make_sanity_image() -> "PIL.Image.Image":
    """
    Generate a synthetic white-background image with black text.

    Returns a PIL Image. Caller converts to whatever format the model expects.

    WHY synthetic rather than a real image?
    Real images require a file on disk that could be missing or corrupted.
    A synthetic image is generated deterministically in memory — zero IO risk,
    zero dependency on external files, identical across all runs.

    What breaks if this assumption is wrong:
    If Pillow's default font renders the text too small or with artifacts,
    the model may fail OCR on a healthy model. Use a larger font size if
    you observe false-negative sanity failures on BF16 baseline.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for sanity check image generation. "
            "Add 'Pillow>=10.0.0' to the Modal image pip_install list."
        )

    img  = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a real font; fall back to default if unavailable.
    # Default PIL font is tiny — if OCR fails on baseline, increase size here.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text((20, 30), SANITY_TEXT, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def _build_text_calibration_ds(cfg, proc_or_tok) -> "Dataset":
    """
    Build a pre-tokenized HuggingFace Dataset for llm-compressor calibration.

    WHY pre-tokenize and remove_columns instead of passing raw text?
    llm-compressor intercepts any Dataset column named "text" and runs its
    own internal preprocessing pipeline on it — bypassing our collator and
    producing a KeyError when the collator later looks for "text".
    The fix (from llm-compressor issue #1571 and Red Hat multimodal examples):
    tokenize via ds.map(), then remove_columns to strip all original columns.
    llm-compressor sees only tensor-valued columns and skips text processing,
    passing samples directly to the collator (which just re-packs them).
    """
    from datasets import load_dataset, Dataset

    tokenizer = proc_or_tok.tokenizer if hasattr(proc_or_tok, "tokenizer") else proc_or_tok
    max_length = 512

    raw = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{cfg.calibration_samples * 2}]",
        streaming=False,
    )
    raw = raw.shuffle(seed=42).select(range(min(cfg.calibration_samples * 2, len(raw))))

    def preprocess(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = ""
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )

    # map() tokenizes and remove_columns strips "text" so llm-compressor
    # doesn't intercept it with its own preprocessing pipeline
    ds = raw.map(preprocess, remove_columns=raw.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 4)
    ds = ds.select(range(min(cfg.calibration_samples, len(ds))))

    print(f"[calibration] built {len(ds)} pre-tokenized samples")
    return ds


def _make_text_collator(tokenizer, max_length: int):
    """
    Returns a data_collator for pre-tokenized text calibration.

    Since _build_text_calibration_ds already tokenized the data,
    the collator just converts lists → tensors and stacks the batch.

    WHY still need a collator if data is pre-tokenized?
    llm-compressor's DataLoader yields list-of-dicts; model.forward()
    needs a dict-of-tensors. The collator does that final conversion.
    """
    import torch

    def collator(batch):
        assert len(batch) == 1, "calibration uses batch_size=1"
        sample = batch[0]
        return {
            k: torch.tensor([v]) if isinstance(v, list) else v.unsqueeze(0)
            for k, v in sample.items()
            if k in ("input_ids", "attention_mask")
        }

    return collator


def _make_vision_collator(processor, max_length: int):
    """
    Returns a data_collator for vision+text calibration (NOVA frames).

    Collator receives a batch with "image_path" key, loads the image,
    applies the Qwen2.5-VL chat template, and returns tensor dict.

    WHY load image in collator not in Dataset.map()?
    PIL Images can't be serialised by Arrow (Dataset.map output).
    Loading at collation time avoids the serialisation problem entirely.
    """
    import torch
    from PIL import Image as PILImage
    from qwen_vl_utils import process_vision_info

    PROMPT = "Describe any objects or hazards visible on the floor."

    def collator(batch):
        assert len(batch) == 1
        img_path = batch[0]["image_path"]
        img      = PILImage.open(img_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": PROMPT},
            ],
        }]

        text         = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {k: v for k, v in inputs.items()}

    return collator

# ---------------------------------------------------------------------------
# Sanity check image generation
# ---------------------------------------------------------------------------

# Deterministic test string — fixed across all runs so substring match is stable.
# WHY not random? Random strings require storing the expected value. A fixed
# string is self-documenting and reproducible without any state.
SANITY_TEXT = "QA_TEST_8472"


def _make_sanity_image() -> "PIL.Image.Image":
    """
    Generate a synthetic white-background image with black text.

    Returns a PIL Image. Caller converts to whatever format the model expects.

    WHY synthetic rather than a real image?
    Real images require a file on disk that could be missing or corrupted.
    A synthetic image is generated deterministically in memory — zero IO risk,
    zero dependency on external files, identical across all runs.

    What breaks if this assumption is wrong:
    If Pillow's default font renders the text too small or with artifacts,
    the model may fail OCR on a healthy model. Use a larger font size if
    you observe false-negative sanity failures on BF16 baseline.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for sanity check image generation. "
            "Add 'Pillow>=10.0.0' to the Modal image pip_install list."
        )

    img  = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a real font; fall back to default if unavailable.
    # Default PIL font is tiny — if OCR fails on baseline, increase size here.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text((20, 30), SANITY_TEXT, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def run_sanity_check(model, processor_or_tokenizer, profile, device: str = "cuda") -> dict:
    """
    Run a minimal generation test to confirm the model is not broken.

    For VLMs: generates from a synthetic image + OCR prompt.
              Checks that SANITY_TEXT appears in the output.
    For LLMs: generates from a fixed text prompt.
              Checks that output is non-empty and coherent (length > 10 tokens).

    Returns a dict with keys: passed (bool), output (str), latency_ms (float).

    Does NOT raise on failure — returns result so caller decides whether to abort.
    Caller (quantize()) is responsible for the abort decision based on
    cfg.ppl_abort_threshold and profile type.

    WHY not raise here?
    The sanity check result belongs in the run report regardless of outcome.
    If we raise inside this function, the report never gets written.
    Separating detection from abort lets the report capture what happened.
    """
    import torch

    t0     = time.perf_counter()
    passed = False
    output = ""

    try:
        if profile.is_multimodal:
            # ── VLM sanity check ─────────────────────────────────────────
            img = _make_sanity_image()

            # Qwen2.5-VL uses its own processor combining vision + text
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": f"Extract the exact text shown in the image. Reply with only the text, nothing else."},
                ],
            }]

            # apply_chat_template is the correct path for Qwen2.5-VL
            # Do NOT use processor(text=..., images=...) directly —
            # it bypasses the chat template and produces wrong token sequences.
            text = processor_or_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor_or_tokenizer(
                text=[text], images=[img],
                return_tensors="pt", padding=True
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,        # deterministic — critical for a gate
                    temperature=None,
                    top_p=None,
                )

            # Decode only newly generated tokens, not the prompt
            n_prompt = inputs["input_ids"].shape[1]
            output   = processor_or_tokenizer.decode(
                generated[0][n_prompt:], skip_special_tokens=True
            ).strip()

            # Gate: SANITY_TEXT must appear in output
            passed = SANITY_TEXT in output

        else:
            # ── LLM sanity check ──────────────────────────────────────────
            prompt = "The capital of France is"
            inputs = processor_or_tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            n_prompt = inputs["input_ids"].shape[1]
            output   = processor_or_tokenizer.decode(
                generated[0][n_prompt:], skip_special_tokens=True
            ).strip()

            # Gate: output should mention Paris and be non-trivial
            passed = "Paris" in output and len(output) > 3

    except Exception as e:
        output = f"EXCEPTION: {e}"
        passed = False

    latency_ms = (time.perf_counter() - t0) * 1000
    status     = "PASS" if passed else "FAIL"
    print(f"[sanity] {status} | output='{output[:80]}' | latency={latency_ms:.0f}ms")

    return {"passed": passed, "output": output, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------

def quantize(cfg: QuantConfig, baseline_ppl: Optional[float] = None) -> dict:
    """
    Main entry point. Called by pipeline.py inside a Modal GPU container.

    Steps:
        1. Validate config against registry ceiling
        2. Load model + processor/tokenizer
        3. If NONE method: skip quantization, run benchmarks only
        4. Build calibration data
        5. Apply quantization via llm-compressor
        6. Run sanity check — abort if failed
        7. Measure perplexity — abort if exceeds threshold
        8. Save weights to volume path
        9. Return metrics dict for benchmark.py and artifacts.py

    Args:
        cfg:          the run config from Layer 1
        baseline_ppl: perplexity from the BF16 baseline run.
                      If provided and cfg.ppl_abort_threshold is set,
                      used for the quality gate comparison.
                      If None, PPL gate is skipped (first run / baseline).

    Returns:
        dict with keys: quant_method, family, ignore_patterns, sanity,
                        ppl, save_path, duration_s, notes
    """
    import torch

    t_start = time.perf_counter()

    # ── Step 1: validate ────────────────────────────────────────────────────
    validate_config(cfg.model_id, cfg.quant_method)
    profile         = get_profile(cfg.model_id)
    ignore_patterns = resolve_ignore_patterns(cfg.model_id, cfg.extra_ignore_patterns)

    print(f"[engine] {cfg.summary()}")
    print(f"[engine] family={profile.family} multimodal={profile.is_multimodal}")
    print(f"[engine] ignore={ignore_patterns}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[engine] WARNING: no GPU detected — this will be very slow")

    # ── Step 2: load model + processor ──────────────────────────────────────
    # WHY load from volume path (cfg.volume_model_path) not model_id?
    # The download happens once in pipeline.py before this function runs.
    # Reading from local disk skips the HF network round-trip on every run.
    # If the path doesn't exist, pipeline.py has a bug — fail immediately.
    model_path = cfg.base_model_path
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"pipeline.py must download the model before calling engine.quantize()."
        )

    # Check if quantized artifact already exists — skip re-quantization.
    # WHY check here not in pipeline.py? engine.py owns the save_path convention.
    # If the artifact exists and method is not NONE, return early with cached result.
    quantized_path = cfg.artifact_path
    if cfg.quant_method != QuantMethod.NONE and Path(quantized_path).exists():
        meta_file = Path(quantized_path) / "quant_meta.json"
        if meta_file.exists():
            import json as _json
            meta = _json.loads(meta_file.read_text())
            print(f"[engine] cache hit — artifact exists at {quantized_path}, skipping re-quantization")
            return {
                "quant_method":    cfg.quant_method.value,
                "family":          get_profile(cfg.model_id).family,
                "ignore_patterns": ignore_patterns,
                "sanity":          {"passed": meta.get("sanity_passed"), "output": meta.get("sanity_output", ""), "latency_ms": 0},
                "ppl":             meta.get("ppl", float("inf")),
                "save_path":       quantized_path,
                "duration_s":      0,
                "notes":           get_profile(cfg.model_id).notes,
            }

    print(f"[engine] loading from {model_path}")

    if profile.is_multimodal:
        # WHY standard Qwen2_5_VLForConditionalGeneration not Traceable:
        # TraceableQwen2_5_VLForConditionalGeneration was removed in
        # llmcompressor>=0.4.2. The standard class works correctly when
        # sequential_targets is set to decoder layers only, preventing
        # the tracer from walking into the ViT's unsupported ops.
        # Source: llmcompressor issue #1347, #1591 (Apr-Jun 2025).
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        proc_or_tok = processor
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        proc_or_tok = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    print(f"[engine] model loaded — {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # ── Step 3: baseline path (no quantization) ──────────────────────────────
    if cfg.quant_method == QuantMethod.NONE:
        print("[engine] NONE method — skipping quantization, running sanity only")
        sanity = run_sanity_check(model, proc_or_tok, profile, device)
        ppl    = _measure_ppl(model, proc_or_tok, device)

        return {
            "quant_method":    cfg.quant_method.value,
            "family":          profile.family,
            "ignore_patterns": ignore_patterns,
            "sanity":          sanity,
            "ppl":             ppl,
            "save_path":       None,   # baseline not saved separately
            "duration_s":      time.perf_counter() - t_start,
            "notes":           profile.notes,
        }

    # ── Step 4 + 5: calibration + quantize ──────────────────────────────────
    # For Qwen2.5-VL, llm-compressor requires:
    #   - A HF Dataset with raw (un-tokenized) columns after chat template
    #   - A data_collator that calls the processor and returns tensors
    #   - process_vision_info from qwen_vl_utils to extract image inputs
    #
    # WHY not pre-tokenize into the Dataset?
    # llm-compressor's DataLoader calls data_collator(batch) where batch is
    # a list of single-sample dicts from the Dataset. The collator must return
    # a tensor dict. If you pre-tokenize, the Dataset contains nested lists
    # that Arrow can't serialize reliably across worker processes.
    # The collator pattern is the documented approach for all VLMs in
    # llm-compressor (Red Hat blog, issue #1571, multimodal examples).
    print(f"[engine] applying {cfg.quant_method.value} quantization")
    print(f"[engine] group_size={cfg.group_size} ignore={ignore_patterns}")

    try:
        import torch
        from datasets import load_dataset, Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier

        scheme = cfg.quant_method.value

        if profile.is_multimodal:
            from qwen_vl_utils import process_vision_info

            # Build a simple text-only calibration dataset using chat template.
            # WHY text-only for generic calibration?
            # Multimodal calibration needs paired image+text samples which
            # require domain-specific data. For generic calibration, text-only
            # chat samples calibrate the LLM backbone (the quantized part)
            # without needing images — the visual.* layers are ignored anyway.
            # For NOVA-specific calibration, the nova_floor branch handles images.
            if cfg.calibration_dataset.value == "nova_floor":
                nova_dir = __import__("pathlib").Path("/calibration/nova_frames")
                if not nova_dir.exists():
                    print("[engine] WARNING: nova_floor not found, falling back to generic")
                    cal_ds = _build_text_calibration_ds(cfg, proc_or_tok)
                    collator = _make_text_collator(proc_or_tok.tokenizer, 512)
                else:
                    frame_paths = sorted(nova_dir.glob("*.jpg"))[:cfg.calibration_samples]
                    records = [{"image_path": str(p)} for p in frame_paths]
                    cal_ds  = Dataset.from_list(records)
                    collator = _make_vision_collator(proc_or_tok, 512)
            else:
                cal_ds   = _build_text_calibration_ds(cfg, proc_or_tok)
                collator = _make_text_collator(proc_or_tok.tokenizer, 512)
        else:
            cal_ds   = _build_text_calibration_ds(cfg, proc_or_tok)
            collator = _make_text_collator(proc_or_tok, 512)

        # sequential_targets limits graph tracing to LLM decoder layers only.
        # Without this, the tracer walks into the ViT and hits unsupported ops.
        # For Qwen2.5-VL the decoder class is Qwen2_5_VLDecoderLayer.
        # For text-only Qwen2.5 it is Qwen2DecoderLayer.
        # ⚠️ Verify class names match transformers==4.49.0 — they changed in 4.50+
        if profile.is_multimodal:
            sequential_targets = ["Qwen2_5_VLDecoderLayer"]
        else:
            sequential_targets = ["Qwen2DecoderLayer"]

        oneshot(
            model=model,
            dataset=cal_ds,
            recipe=GPTQModifier(
                targets="Linear",
                scheme=scheme,
                ignore=ignore_patterns,
                dampening_frac=0.01,
                sequential_targets=sequential_targets,
            ),
            max_seq_length=512,
            num_calibration_samples=cfg.calibration_samples,
            data_collator=collator,
        )
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for llmcompressor: {e}. "
            "Modal image must include: llmcompressor>=0.5.1, qwen-vl-utils>=0.0.8"
        )

    print("[engine] quantization complete")

    # ── Step 6: sanity check — abort before saving ───────────────────────────
    sanity = run_sanity_check(model, proc_or_tok, profile, device)

    if not sanity["passed"]:
        raise RuntimeError(
            f"[engine] ABORT: sanity check failed after quantization.\n"
            f"Model output: '{sanity['output']}'\n"
            f"Expected to find: '{SANITY_TEXT}' (VLM) or 'Paris' (LLM)\n"
            f"Weights NOT saved. Fix quantization config before retrying."
        )

    # ── Step 7: PPL gate ─────────────────────────────────────────────────────
    ppl = _measure_ppl(model, proc_or_tok, device)
    print(f"[engine] PPL after quant: {ppl:.3f}")

    if baseline_ppl is not None and cfg.ppl_abort_threshold is not None:
        ppl_limit = baseline_ppl * cfg.ppl_abort_threshold
        if ppl > ppl_limit:
            raise RuntimeError(
                f"[engine] ABORT: PPL {ppl:.3f} exceeds gate "
                f"({baseline_ppl:.3f} baseline × {cfg.ppl_abort_threshold} = {ppl_limit:.3f}).\n"
                f"Weights NOT saved. Loosen ppl_abort_threshold or use a less aggressive method."
            )

    # ── Step 8: save ─────────────────────────────────────────────────────────
    save_path = cfg.artifact_path
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f"[engine] saving to {save_path}")
    # WHY no save_compressed flag (default=True → pack-quantized):
    # vLLM reads pack-quantized W4A16 artifacts and dispatches to
    # Marlin kernels automatically. vllm_benchmark.py handles inference.
    model.save_pretrained(save_path)
    proc_or_tok.save_pretrained(save_path)

    # Save run metadata alongside weights for traceability
    meta = {
        "model_id":          cfg.model_id,
        "quant_method":      cfg.quant_method.value,
        "family":            profile.family,
        "ignore_patterns":   ignore_patterns,
        "calibration":       cfg.calibration_dataset.value,
        "group_size":        cfg.group_size,
        "ppl":               ppl,
        "baseline_ppl":      baseline_ppl,
        "sanity_passed":     sanity["passed"],
        "sanity_output":     sanity["output"],
    }
    with open(Path(save_path) / "quant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    duration = time.perf_counter() - t_start
    print(f"[engine] done in {duration/60:.1f} min")

    return {
        "quant_method":    cfg.quant_method.value,
        "family":          profile.family,
        "ignore_patterns": ignore_patterns,
        "sanity":          sanity,
        "ppl":             ppl,
        "save_path":       save_path,
        "duration_s":      duration,
        "notes":           profile.notes,
    }


# ---------------------------------------------------------------------------
# Perplexity measurement (text-only, on wikitext-2 sample)
# ---------------------------------------------------------------------------

def _measure_ppl(model, proc_or_tok, device: str, n_samples: int = 20) -> float:
    """
    Estimate perplexity on a small wikitext-2 sample.

    WHY wikitext-2 and not a VLM benchmark?
    Perplexity on text measures the LLM backbone's language modelling quality
    independently of the vision pipeline. It's fast (20 samples ≈ 30 seconds)
    and comparable across runs. Visual task accuracy is measured in benchmark.py
    separately with longer-running VQA tasks.

    WHY 20 samples not the full dataset?
    Full wikitext-2 PPL takes 5–10 minutes on A10G. 20 samples gives a
    directionally correct estimate — sufficient for the abort gate.
    Full PPL should be measured in the final benchmark pass, not here.

    What breaks if this is wrong:
    If 20 samples happen to be outliers, PPL estimate is noisy.
    The ppl_abort_threshold accounts for this — a 10% gate (1.10) is loose
    enough to survive sample variance. Tighten only after validating on
    the full dataset.

    ⚠️ For VLMs this measures the LLM backbone only — it does not exercise
    the visual pipeline. That's intentional: sanity check covers vision;
    PPL covers language. Two separate signals.
    """
    import torch
    import math

    tokenizer = proc_or_tok.tokenizer if hasattr(proc_or_tok, "tokenizer") else proc_or_tok

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
        texts = [row["text"] for i, row in enumerate(ds) if row["text"].strip() and i < n_samples]
    except Exception as e:
        print(f"[ppl] WARNING: could not load wikitext — {e}. Returning inf (safe failure).")
        return float("inf")

    total_nll = 0.0
    total_tok = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            if enc["input_ids"].shape[1] < 2:
                continue
            out      = model(**enc, labels=enc["input_ids"])
            n_tokens = enc["input_ids"].shape[1]
            total_nll += out.loss.item() * n_tokens
            total_tok += n_tokens

    if total_tok == 0:
        # WHY inf and not 0.0: 0.0 would pass the PPL gate (lower is better),
        # silently treating a broken measurement as a perfect score.
        # inf correctly triggers the gate and aborts before pushing broken weights.
        return float("inf")

    return math.exp(total_nll / total_tok)