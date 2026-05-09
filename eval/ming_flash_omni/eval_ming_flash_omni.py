#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["LD_PRELOAD"] = ""

import warnings
warnings.filterwarnings("ignore")

_MING_REPO_DIR = os.environ.get("MING_REPO_DIR")
if not _MING_REPO_DIR:
    raise SystemExit(
        "Required environment variable 'MING_REPO_DIR' is not set. "
        "Please set it via the matching .sh launcher (path to cloned Ming repo)."
    )
if _MING_REPO_DIR not in sys.path:
    sys.path.insert(0, _MING_REPO_DIR)

import json
import re
import time
import traceback
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration                

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("MING_MODEL_PATH")
CLEANED_DIR = _require_env("MING_CLEANED_DIR")
VIDEOS_DIR  = _require_env("MING_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("MING_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

SETTINGS: Dict[str, Any] = {
    "model_path":      MODEL_PATH,
    "cleaned_dir":     CLEANED_DIR,
    "videos_dir":      VIDEOS_DIR,
    "output_dir":      OUTPUT_DIR,
    "max_tokens":      512,
    "max_file_mib":    5000,
    "num_layers":      32,

    "max_video_frames": int(os.environ.get("MING_MAX_VIDEO_FRAMES", "64")),
}

@dataclass
class EvaluationMetrics:
    question_id: int
    task_type: str
    video_id: str
    question_type: str
    predicted_options: List[str]
    correct_options: List[str]
    correct: bool
    abstained: bool
    reasoning: str
    api_success: bool
    error_message: Optional[str] = None
    processing_time_sec: float = 0.0
    raw_model_text: Optional[str] = None
    local_video_path: Optional[str] = None
    http_status: Optional[int] = None
    hop_original_event_count: Optional[int] = None
    minute_hop_count: Optional[int] = None
    hop_length_label: Optional[str] = None
    hop_evidence_span_minutes: Optional[float] = None

def hop_metadata_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    oids = item.get("original_event_ids")
    traj = item.get("trajectory_with_timestamps")
    span = None
    if isinstance(traj, list) and len(traj) >= 2:
        vals = []
        for ev in traj:
            tr = ev.get("event_time_range") or {}
            for k in ("start_minute", "end_minute"):
                if tr.get(k) is not None:
                    vals.append(float(tr[k]))
        if len(vals) >= 2:
            span = max(vals) - min(vals)
    return {
        "hop_original_event_count": len(oids) if isinstance(oids, list) else None,
        "minute_hop_count":         item.get("minute_hop_count"),
        "hop_length_label":         str(item.get("hop_length_label")) if item.get("hop_length_label") else None,
        "hop_evidence_span_minutes": span,
    }

def normalize_option_letters(raw: Any) -> List[str]:
    if not raw:
        return []
    chars = re.findall(r"[A-D]", str(raw).upper())
    return sorted(list(set(chars)))

def parse_response_to_prediction(response_text: str) -> List[str]:
    if not response_text:
        return []
    think_tag = "</think>"
    last_think_idx = response_text.rfind(think_tag)
    answer_part = (
        response_text[last_think_idx + len(think_tag):].strip()
        if last_think_idx != -1
        else response_text.strip()
    )
    explicit_match = re.search(
        r"(?:answer|choice|options|selected)[:\s]+([A-D](?:\s*[,，/]\s*[A-D])*)",
        answer_part,
        re.IGNORECASE,
    )
    if explicit_match:
        return normalize_option_letters(explicit_match.group(1))
    return normalize_option_letters(re.findall(r"\b([A-D])\b", answer_part.upper()))

def item_key(it: Dict[str, Any]) -> tuple:
    return (str(it.get("task_type", "")), int(it.get("question_id", -1)))

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_summary(metrics: List[EvaluationMetrics], run_id: str) -> Dict[str, Any]:
    if not metrics:
        return {}
    total    = len(metrics)
    correct  = sum(1 for m in metrics if m.correct)
    api_failed = sum(1 for m in metrics if not m.api_success)
    by_task: Dict[str, Any] = {}
    for tt in sorted({m.task_type for m in metrics}):
        sub = [m for m in metrics if m.task_type == tt]
        c   = sum(1 for m in sub if m.correct)
        by_task[tt] = {"count": len(sub), "correct": c, "accuracy": c / len(sub) if sub else 0.0}
    return {
        "run_id":            run_id,
        "timestamp":         datetime.now().isoformat(),
        "total_items":       total,
        "correct":           correct,
        "api_failed":        api_failed,
        "overall_accuracy":  correct / total if total else 0.0,
        "api_success_rate":  (total - api_failed) / total if total else 0.0,
        "by_task_type":      by_task,
    }

def build_split_model(num_layers: int) -> Dict[str, Any]:
    """Distribute transformer layers across all visible GPUs while budgeting for non-transformer modules.

    Ming's prompt_wrap_navit requires vision/audio encoder, word_embeddings and lm_head
    to live on the same device (GPU 0), so these non-transformer modules stay pinned to GPU 0,
    and the number of transformer layers on GPU 0 is reduced to offload work to the other GPUs.

    4-GPU 32-layer example: gpu0:4 | gpu1:10 | gpu2:10 | gpu3:8
    8-GPU 32-layer example: gpu0:2 | gpu1-6:5 each | gpu7:0 (all offloaded to earlier GPUs)
    """
    import math
    device_map: Dict[str, Any] = {}
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA device found.")

    if world_size == 1:
        for i in range(num_layers):
            device_map[f"model.model.layers.{i}"] = 0
    else:
        avg = num_layers / world_size

        gpu0_layers = max(1, math.floor(avg * 0.5))
        remaining   = num_layers - gpu0_layers
        other_gpus  = world_size - 1
        per_other   = math.ceil(remaining / other_gpus)

        layer_idx = 0
        for _ in range(gpu0_layers):
            device_map[f"model.model.layers.{layer_idx}"] = 0
            layer_idx += 1
        for g in range(1, world_size):
            for _ in range(per_other):
                if layer_idx >= num_layers:
                    break
                device_map[f"model.model.layers.{layer_idx}"] = g
                layer_idx += 1

    for key in (
        "vision", "audio", "linear_proj", "linear_proj_audio",
        "model.model.word_embeddings.weight",
        "model.model.norm.weight",
        "model.lm_head.weight",
        "model.model.norm",
    ):
        device_map[key] = 0

    layer_dist: Dict[int, int] = {}
    for i in range(num_layers):
        g = device_map[f"model.model.layers.{i}"]
        layer_dist[g] = layer_dist.get(g, 0) + 1
    dist_str = " | ".join(f"gpu{g}:{n}" for g, n in sorted(layer_dist.items()))
    print(
        f"[split_model] world_size={world_size}, num_layers={num_layers}: {dist_str} "
        f"| vision/audio/embed/lm_head→gpu0",
        flush=True,
    )
    return device_map

def evaluate_one(item: Dict[str, Any], cfg: Dict[str, Any]) -> EvaluationMetrics:
    t0  = time.time()
    hop = hop_metadata_from_item(item)
    qid  = int(item.get("question_id", -1))
    task = str(item.get("task_type", "unknown"))
    vid  = str(item.get("video_id", ""))
    gold = normalize_option_letters(item.get("correct_options", []))

    def _fail(msg: str) -> EvaluationMetrics:
        return EvaluationMetrics(
            qid, task, vid,
            str(item.get("question_type", "single")),
            [], gold, False, True, "", False,
            msg, time.time() - t0, None, None, None, **hop,
        )

    try:
        rel = vid.replace("\\", "/").lstrip("/")
        videos_dir: Path = cfg["videos_dir"]
        path = None
        for candidate in [
            videos_dir / rel,
            videos_dir / Path(rel).name,
            videos_dir / "videos" / Path(rel).name,
        ]:
            if candidate.exists():
                path = candidate.resolve()
                break
        if not path:
            return _fail(f"Video not found: {vid}")

        max_mib: float = cfg["max_mib"]
        if max_mib > 0 and (path.stat().st_size / 1048576) > max_mib:
            return _fail(f"File too large: {path.stat().st_size}")

        opts = item.get("options", {})
        options_text = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        prompt = (
            "You are given a video. Base your answer only on what you see and hear.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else.\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        processor  = cfg["processor"]
        model      = cfg["model"]
        max_tokens: int = cfg["max_tokens"]

        messages = [{"role": "HUMAN", "content": [
            {"type": "video", "video": str(path)},
            {"type": "text",  "text": prompt},
        ]}]

        text = processor.apply_chat_template(messages)
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        )

        _dev = torch.device("cuda:0")
        for k in list(inputs.keys()):
            v = inputs[k]
            if not isinstance(v, torch.Tensor):
                continue
            if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
                inputs[k] = v.to(device=_dev, dtype=torch.bfloat16)
            else:
                inputs[k] = v.to(device=_dev)

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    eos_token_id=processor.gen_terminator,
                    num_logits_to_keep=1,
                )
        except torch.cuda.OutOfMemoryError as oom_e:
            torch.cuda.empty_cache()
            return _fail(f"CUDA OOM: {oom_e}")

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        pred = parse_response_to_prediction(raw_text)

        return EvaluationMetrics(
            qid, task, vid,
            str(item.get("question_type", "single")),
            pred, gold,
            (pred == gold and len(gold) > 0),
            not bool(pred),
            raw_text, True, None,
            time.time() - t0, raw_text, str(path), None, **hop,
        )
    except Exception:
        return _fail(traceback.format_exc())

def main() -> None:
    s = SETTINGS
    model_path      = Path(s["model_path"])
    cleaned_dir     = Path(s["cleaned_dir"])
    videos_dir      = Path(s["videos_dir"])
    output_dir      = Path(s["output_dir"])
    max_tokens        = int(s["max_tokens"])
    max_mib           = float(s["max_file_mib"])
    num_layers        = int(s["num_layers"])
    max_video_frames  = int(s.get("max_video_frames", 64))

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
        print(f"[subset] manifest {len(manifest_keys)} -> pool {len(pool)}", flush=True)
    else:
        print(f"[subset] No manifest found, evaluating full pool = {len(pool)}", flush=True)

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S_lv_ming")
    run_dir = output_dir / "eval_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = run_dir / "detailed_results.json"
    summary_path  = run_dir / "summary.json"
    config_path   = run_dir / "run_config.json"
    to_run = pool
    print(f"Loading Ming-flash-omni-2.0 from: {model_path}", flush=True)
    print(f"  CUDA devices visible: {torch.cuda.device_count()}", flush=True)

    device_map = build_split_model(num_layers)

    try:
        import flash_attn              
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    if attn_impl == "eager":

        import json as _json
        from configuration_bailingmm2 import BailingMM2Config                

        with open(model_path / "config.json", "r", encoding="utf-8") as _f:
            _cfg_dict = _json.load(_f)
        top_cfg = BailingMM2Config(**{
            k: v for k, v in _cfg_dict.items()
            if k not in ("architectures", "model_type", "transformers_version")
        })

    else:
        top_cfg = None

    from_pretrained_kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device_map,
        load_image_gen=False,
        load_talker=False,
        local_files_only=True,
    )
    if top_cfg is not None:
        from_pretrained_kwargs["config"] = top_cfg

    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        str(model_path), **from_pretrained_kwargs
    ).to(dtype=torch.bfloat16)
    model.eval()

    if attn_impl == "eager":
        patched = 0
        for _mod in model.modules():
            _cfg = getattr(_mod, "config", None)
            if _cfg is not None:
                if hasattr(_cfg, "_attn_implementation"):
                    _cfg._attn_implementation = "eager"
                    patched += 1
                if hasattr(_cfg, "attn_implementation"):
                    _cfg.attn_implementation = "eager"

    ming_repo = Path(_MING_REPO_DIR)
    processor = AutoProcessor.from_pretrained(
        str(ming_repo),
        trust_remote_code=True,
        local_files_only=True,
    )

    cfg = {
        "videos_dir":       videos_dir,
        "max_mib":          max_mib,
        "model":            model,
        "processor":        processor,
        "max_tokens":       max_tokens,
        "max_video_frames": max_video_frames,
    }

    save_json(config_path, {
        "script":          "eval_ming.py",
        "model_path":      str(model_path),
        "cleaned_dir":     str(cleaned_dir),
        "videos_dir":      str(videos_dir),
        "output_dir":      str(run_dir),
        "max_tokens":        max_tokens,
        "max_file_mib":      max_mib,
        "num_layers":        num_layers,
        "max_video_frames":  max_video_frames,
        "item_count":        len(pool),
        "cuda_devices":      torch.cuda.device_count(),
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        save_json(detailed_path, [asdict(m) for m in results_obj])
        save_json(summary_path,  compute_summary(results_obj, run_id))

    results: List[EvaluationMetrics] = []
    for it in tqdm(to_run, desc="LV-Bench (Ming-flash-omni-2.0)"):
        res_obj = evaluate_one(it, cfg)
        results.append(res_obj)
        _flush(results)

    _flush(results)
    sm = compute_summary(results, run_id)
    print("=" * 72)
    print(f"Run:      {run_id}")
    print(f"Items:    {len(results)} | accuracy: {sm.get('overall_accuracy', 0):.2%} | api_ok: {sm.get('api_success_rate', 0):.2%}")
    print(f"Output:   {run_dir}")
    print("=" * 72)

if __name__ == "__main__":
    main()
