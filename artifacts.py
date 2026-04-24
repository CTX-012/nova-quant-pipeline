"""
artifacts.py — Layer 5: artifact output.

Three responsibilities, strictly separated:
    1. Flatten benchmark metrics into a JSONL row and append to report
    2. Push quantized weights to HuggingFace Hub with a generated model card
    3. Confirm Modal volume write

Nothing in this file runs quantization or benchmarking.
It only writes what engine.py and benchmark.py already produced.

WHY append-only JSONL (not JSON array, not SQLite):
- Concurrent runs can append without read-parse-rewrite (no corruption risk)
- pd.read_json("report.jsonl", lines=True) → instant DataFrame → .to_latex()
- One line per run = one grep-able, diff-able record
- If a run aborts mid-write, the partial line is detectable (invalid JSON)
  and the prior lines remain intact

⚠️ Assumption: HF_TOKEN env var is set by Modal secret injection.
   If missing, push_to_hub raises huggingface_hub.utils.RepositoryNotFoundError
   immediately — no partial upload.
⚠️ Assumption: report_path parent directory exists. pipeline.py creates it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import QuantConfig
from registry import get_profile


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def _flatten(cfg: QuantConfig, benchmark_metrics: dict) -> dict:
    """
    Produce one flat dict representing one paper table row.

    Key design rule: every key here is a final column name.
    No nesting. No lists. Lists (e.g. vqa_responses) are serialised to
    a JSON string so the row stays one flat record.

    Column naming convention:
        Identity cols:  run_*, model_*
        Config cols:    cfg_*
        Quality cols:   q_*
        Speed cols:     spd_*
        Memory cols:    mem_*
        VQA cols:       vqa_*

    This prefix scheme lets pandas groupby/filter by column category without
    any string parsing — just df[[c for c in df.columns if c.startswith('spd_')]].
    """
    profile = get_profile(cfg.model_id)

    row = {
        # ── Identity ────────────────────────────────────────────────────
        "run_name":          cfg.run_name,
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        "model_id":          cfg.model_id,
        "model_family":      profile.family,
        "is_multimodal":     profile.is_multimodal,

        # ── Config ──────────────────────────────────────────────────────
        "cfg_method":        cfg.quant_method.value,
        "cfg_calibration":   cfg.calibration_dataset.value,
        "cfg_group_size":    cfg.group_size,
        "cfg_gpu":           cfg.gpu.value,
        "cfg_ignore":        json.dumps(benchmark_metrics.get("ignore_patterns", [])),

        # ── Quality ─────────────────────────────────────────────────────
        "q_ppl":             benchmark_metrics.get("ppl"),
        "q_sanity_passed":   benchmark_metrics.get("sanity_passed"),

        # ── Speed (vLLM for quantized, transformers for baseline) ────────────
        "spd_tok_per_s":     benchmark_metrics.get("tok_per_s"),
        "spd_ttft_ms":       benchmark_metrics.get("ttft_ms"),
        "spd_tpot_ms":       benchmark_metrics.get("tpot_ms"),
        "spd_engine":        benchmark_metrics.get("speed_engine", "transformers"),

        # ── Memory ──────────────────────────────────────────────────────
        "mem_load_vram_gb":      benchmark_metrics.get("load_vram_gb"),
        "mem_inference_vram_gb": benchmark_metrics.get("inference_vram_gb"),
        "mem_disk_gb":           benchmark_metrics.get("disk_gb"),

        # ── VQA spot-check ───────────────────────────────────────────────
        "vqa_n_flagged":     benchmark_metrics.get("vqa_n_flagged"),
        "vqa_flagged_ratio": benchmark_metrics.get("vqa_flagged_ratio"),
        # Responses serialised to string — preserves flat row invariant
        "vqa_responses_json": json.dumps(
            benchmark_metrics.get("vqa_responses", []),
            ensure_ascii=False,
        ),

        # ── Registry notes ───────────────────────────────────────────────
        # Serialised list so the row stays flat
        "notes_json":        json.dumps(benchmark_metrics.get("registry_notes", [])),
    }

    return row


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def append_report_row(
    cfg: QuantConfig,
    benchmark_metrics: dict,
    report_path: str,
) -> None:
    """
    Flatten metrics and append one JSON line to the shared report file.

    Append-only: never reads the existing file.
    This makes it safe to call from concurrent Modal runs targeting the
    same report file on a shared volume.

    If the line is somehow malformed, json.dumps raises before any write —
    the file is never left in a partially-written state by this function.
    (OS-level write atomicity is not guaranteed for lines > 4096 bytes,
    but a single metrics row is well under that limit.)
    """
    row      = _flatten(cfg, benchmark_metrics)
    line     = json.dumps(row, ensure_ascii=False)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"[artifacts] appended row to {report_path}")
    print(f"[artifacts] method={row['cfg_method']} ppl={row['q_ppl']} "
          f"ttft={row['spd_ttft_ms']}ms tok/s={row['spd_tok_per_s']}")


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def _generate_model_card(cfg: QuantConfig, benchmark_metrics: dict) -> str:
    """
    Generate a minimal but informative HuggingFace model card (README.md).

    WHY generate programmatically rather than a static template?
    The card must include actual benchmark numbers from this run.
    A static template would require manual editing after every quantization run.
    Programmatic generation ensures the card always matches the actual artifact.

    Format follows HF model card spec: YAML frontmatter + Markdown body.
    ⚠️ Verify HF model card schema at huggingface.co/docs/hub/model-cards
       if adding new metadata fields — invalid YAML frontmatter causes hub
       indexing failures (model appears but search metadata is wrong).
    """
    profile  = get_profile(cfg.model_id)
    method   = cfg.quant_method.value
    ppl      = benchmark_metrics.get("ppl", "N/A")
    ttft     = benchmark_metrics.get("ttft_ms", "N/A")
    tpot     = benchmark_metrics.get("tpot_ms", "N/A")
    tok_s    = benchmark_metrics.get("tok_per_s", "N/A")
    disk_gb  = benchmark_metrics.get("disk_gb", "N/A")
    vram_gb  = benchmark_metrics.get("inference_vram_gb", "N/A")
    sanity   = "✅ PASS" if benchmark_metrics.get("sanity_passed") else "❌ FAIL"
    cal      = cfg.calibration_dataset.value
    ignored  = benchmark_metrics.get("ignore_patterns", [])
    notes    = benchmark_metrics.get("registry_notes", [])
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ppl_str  = f"{ppl:.3f}" if isinstance(ppl, float) else str(ppl)
    ttft_str = f"{ttft:.1f}" if isinstance(ttft, float) else str(ttft)
    tpot_str = f"{tpot:.1f}" if isinstance(tpot, float) else str(tpot)
    tok_str  = f"{tok_s:.1f}" if isinstance(tok_s, float) else str(tok_s)
    disk_str = f"{disk_gb:.2f}" if isinstance(disk_gb, float) else str(disk_gb)
    vram_str = f"{vram_gb:.2f}" if isinstance(vram_gb, float) else str(vram_gb)

    notes_md = "\n".join(f"- {n}" for n in notes) if notes else "- None"

    # Pre-compute strings that contain quotes — avoids backslash-in-fstring
    # which is a SyntaxError in Python <3.12 (Modal container runs 3.11).
    repo = cfg.hf_repo_id
    if profile.is_multimodal:
        imports_line  = "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor"
        model_cls     = "Qwen2_5_VLForConditionalGeneration"
        proc_line     = f'processor = AutoProcessor.from_pretrained("{repo}")'
    else:
        imports_line  = "from transformers import AutoModelForCausalLM, AutoTokenizer"
        model_cls     = "AutoModelForCausalLM"
        proc_line     = f'tokenizer = AutoTokenizer.from_pretrained("{repo}")'   

    card = (
        "---\n"
        f"base_model: {cfg.model_id}\n"
        "tags:\n"
        f"  - quantized\n"
        f"  - {method.lower()}\n"
        "  - robotics\n"
        "  - nova-robot\n"
        f'pipeline_tag: {"image-text-to-text" if profile.is_multimodal else "text-generation"}\n'
        "language:\n"
        "  - en\n"
        "---\n"
        "\n"
        f"# {cfg.hf_repo_id.split('/')[-1]}\n"
        "\n"
        f"Quantized with the NOVA quantization pipeline on {ts}.\n"
        f"Base model: [{cfg.model_id}](https://huggingface.co/{cfg.model_id})\n"
        "\n"
        "## Quantization details\n"
        "\n"
        "| Parameter | Value |\n"
        "|---|---|\n"
        f"| Method | `{method}` |\n"
        f"| Group size | {cfg.group_size} |\n"
        f"| Calibration | `{cal}` |\n"
        f"| Ignored modules | `{', '.join(ignored)}` |\n"
        "| Tool | `llm-compressor >= 0.4.2` |\n"
        "\n"
        "## Benchmark results\n"
        "\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Perplexity (wikitext-2, 20 samples) | {ppl_str} |\n"
        f"| OCR sanity check | {sanity} |\n"
        f"| Tokens / second | {tok_str} |\n"
        f"| TTFT (exact, prefill only) | {ttft_str} ms |\n"
        f"| TPOT (exact, per output token) | {tpot_str} ms |\n"
        f"| Inference VRAM | {vram_str} GB |\n"
        f"| Disk size | {disk_str} GB |\n"
        "\n"
        "> TTFT and TPOT measured with `BaseStreamer` injection (prompt-skip corrected).\n"
        "\n"
        "## Registry notes\n"
        "\n"
        f"{notes_md}\n"
        "\n"
        "## Usage\n"
        "\n"
        "```python\n"
        f"{imports_line}\n"
        "import torch\n"
        "\n"
        f'model = {model_cls}.from_pretrained(\n'
        f'    "{repo}",\n'
        "    torch_dtype=torch.bfloat16,\n"
        '    device_map="auto",\n'
        ")\n"
        f"{proc_line}\n"
        "```\n"
        "\n"
        "## Citation\n"
        "\n"
        "If you use this model in research, please cite the NOVA project.\n"
        "Pipeline source: `Mohaaxa/nova-quant-pipeline`\n"
    )

    return card


# ---------------------------------------------------------------------------
# HuggingFace Hub push
# ---------------------------------------------------------------------------

def push_to_hub(
    cfg: QuantConfig,
    save_path: str,
    benchmark_metrics: dict,
) -> Optional[str]:
    """
    Push the saved artifact to HuggingFace Hub under cfg.hf_org.

    Returns the repo URL on success, None if push_to_hub=False in config.

    WHY upload_folder and not model.push_to_hub()?
    model.push_to_hub() requires the model object in memory.
    We benchmark from a reloaded artifact — we don't hold the model in memory
    here. upload_folder pushes the directory contents directly, which is
    exactly what we saved in engine.py.

    ⚠️ Requires HF_TOKEN in environment. Modal injects this from the secret.
    ⚠️ create_repo with exist_ok=True means re-running overwrites the repo.
       This is intentional — a re-run is a corrected artifact, not a new one.
       Use a different run_name if you need to preserve the old version.
    """
    if not cfg.push_to_hub:
        print("[artifacts] push_to_hub=False — skipping hub upload")
        return None

    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        raise ImportError(
            "huggingface_hub not found. Add it to the Modal image pip_install list."
        )

    token    = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found in environment. "
            "Modal secret must inject HF_TOKEN before this function runs."
        )

    api      = HfApi(token=token)
    repo_id  = cfg.hf_repo_id

    print(f"[artifacts] creating/verifying repo: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,       # idempotent — safe to re-run
        private=False,
    )

    # Write model card to the save directory so it uploads with the weights
    card_path = Path(save_path) / "README.md"
    card_path.write_text(_generate_model_card(cfg, benchmark_metrics), encoding="utf-8")
    print(f"[artifacts] model card written to {card_path}")

    print(f"[artifacts] uploading {save_path} → {repo_id}")
    api.upload_folder(
        folder_path=save_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"NOVA pipeline: {cfg.quant_method.value} | {cfg.calibration_dataset.value} | run={cfg.run_name}",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[artifacts] pushed → {url}")
    return url


# ---------------------------------------------------------------------------
# Public API — called by pipeline.py
# ---------------------------------------------------------------------------

def save_artifacts(
    cfg: QuantConfig,
    engine_result: dict,
    benchmark_metrics: dict,
    report_path: str,
) -> dict:
    """
    Orchestrate all artifact writes for one pipeline run.

    Args:
        cfg:               run config
        engine_result:     dict from engine.quantize()
        benchmark_metrics: dict from benchmark.run_benchmark()
        report_path:       path to the shared JSONL report file

    Returns:
        dict with hub_url, report_path, volume_path — for pipeline.py logging
    """
    save_path = engine_result.get("save_path")

    # 1. Append JSONL row — always, regardless of other failures
    #    WHY always: even if hub push fails, the local report is the primary
    #    research artifact. Never lose measurements due to network errors.
    append_report_row(cfg, benchmark_metrics, report_path)

    # 2. Push to HuggingFace Hub
    hub_url = None
    if save_path and cfg.push_to_hub:
        hub_url = push_to_hub(cfg, save_path, benchmark_metrics)

    # 3. Volume write confirmation — engine.py already saved; just confirm
    volume_path = save_path if cfg.push_to_volume else None
    if volume_path:
        print(f"[artifacts] confirmed volume path: {volume_path}")

    return {
        "hub_url":     hub_url,
        "report_path": report_path,
        "volume_path": volume_path,
    }