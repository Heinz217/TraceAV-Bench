#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DECORD_NUM_THREADS", "1")
os.environ.setdefault("DECORD_EOF_RETRY_COUNT", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCHCODEC", "1")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["LD_PRELOAD"] = ""

_REPO_DIR = os.environ.get("SALMONN_REPO_DIR")
if not _REPO_DIR:
    raise SystemExit(
        "Required environment variable 'SALMONN_REPO_DIR' is not set. "
        "Please set it via the matching .sh launcher (path to video_SALMONN2_plus repo)."
    )

if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import json
import re
import time
import traceback
import dataclasses
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("SALMONN_MODEL_PATH")
CLEANED_DIR = _require_env("SALMONN_CLEANED_DIR")
VIDEOS_DIR  = _require_env("SALMONN_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("SALMONN_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

_MAX_FRAMES = int(os.environ.get("SALMONN_MAX_FRAMES", "384"))
_MAX_PIXELS = int(os.environ.get("SALMONN_MAX_PIXELS", "7000"))

SETTINGS: Dict[str, Any] = {
    "model_path":      MODEL_PATH,
    "cleaned_dir":     CLEANED_DIR,
    "videos_dir":      VIDEOS_DIR,
    "output_file":     f"{OUTPUT_DIR}/salmonn_results_{datetime.now().strftime('%m%d_%H%M')}.json",
    "max_tokens":      512,
    "max_frames":      _MAX_FRAMES,
    "max_pixels":      _MAX_PIXELS,
    "max_file_mib":    5000,
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
        "minute_hop_count": item.get("minute_hop_count"),
        "hop_length_label": str(item.get("hop_length_label")) if item.get("hop_length_label") else None,
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
    total = len(metrics)
    correct = sum(1 for m in metrics if m.correct)
    api_failed = sum(1 for m in metrics if not m.api_success)
    by_task: Dict[str, Any] = {}
    for tt in sorted({m.task_type for m in metrics}):
        sub = [m for m in metrics if m.task_type == tt]
        c = sum(1 for m in sub if m.correct)
        by_task[tt] = {"count": len(sub), "correct": c, "accuracy": c / len(sub) if sub else 0.0}
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total_items": total,
        "correct": correct,
        "api_failed": api_failed,
        "overall_accuracy": correct / total if total else 0.0,
        "api_success_rate": (total - api_failed) / total if total else 0.0,
        "by_task_type": by_task,
    }

def evaluate_one(item: Dict[str, Any], cfg: Dict[str, Any]) -> EvaluationMetrics:
    t0 = time.time()
    hop = hop_metadata_from_item(item)
    qid  = int(item.get("question_id", -1))
    task = str(item.get("task_type", "unknown"))
    vid  = str(item.get("video_id", ""))
    gold = normalize_option_letters(item.get("correct_options", []))

    def _fail(msg: str, status: int = 0) -> EvaluationMetrics:
        return EvaluationMetrics(
            qid, task, vid,
            str(item.get("question_type", "single")),
            [], gold, False, True, "", False,
            msg, time.time() - t0, None, None, status, **hop,
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

        if cfg["max_mib"] > 0 and (path.stat().st_size / 1048576) > cfg["max_mib"]:
            return _fail(f"File too large: {path.stat().st_size / 1048576:.1f} MiB")

        opts = item.get("options", {})
        options_text = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        prompt = (
            "You are given a video. Base your answer only on what you see and hear.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else.\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        model        = cfg["model"]
        dataset      = cfg["dataset"]
        tokenizer    = cfg["tokenizer"]
        max_new_tokens = cfg["max_tokens"]

        source = {
            "video": str(path),
            "use_audio": True,                  
            "conversations": [
                {"from": "human", "value": f"<video>\n{prompt}"},
                {"from": "gpt",   "value": ""},
            ],
        }
        inputs = dataset._get_item(source)

        for drop_key in ("video", "image", "prompt", "ref", "audio", "use_audio", "should_use", "labels"):
            inputs.pop(drop_key, None)

        _device = torch.device("cuda:0")

        processed = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                processed[k] = v
                continue
            if v.dim() <= 1 and k in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
                processed[k] = v.reshape(1, -1).to(device=_device)
            elif k in ("pixel_values_videos", "pixel_values"):
                processed[k] = v.to(device=_device, dtype=torch.bfloat16)
            elif k in ("video_grid_thw", "image_grid_thw"):
                processed[k] = v.to(device=_device, dtype=torch.long)
            elif v.is_floating_point():
                processed[k] = v.to(device=_device, dtype=torch.bfloat16)
            else:
                processed[k] = v.to(device=_device)
        inputs = processed

        if "attention_mask" in inputs and isinstance(inputs["attention_mask"], list):
            inputs["attention_mask"] = torch.ones(
                inputs["input_ids"].shape, dtype=torch.long, device=_device
            )
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones(
                inputs["input_ids"].shape, dtype=torch.long, device=_device
            )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[:, input_len:]
        raw_text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

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
    max_tokens      = int(s["max_tokens"])
    max_frames      = int(s["max_frames"])
    max_pixels      = int(s["max_pixels"])
    max_mib         = float(s["max_file_mib"])

    print(f"CUDA_VISIBLE_DEVICES={_CUDA_DEVICES}", flush=True)
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
        print(f"[subset] manifest {len(manifest_keys)} items -> pool {len(pool)} items", flush=True)
    else:
        print(f"[subset] no manifest found, evaluating full pool = {len(pool)} items", flush=True)

    out_root = Path(OUTPUT_DIR)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S_lv_salmonn")
    run_dir  = out_root / "eval_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = run_dir / "detailed_results.json"
    summary_path  = run_dir / "summary.json"
    config_path   = run_dir / "run_config.json"
    to_run = pool
    print(f"To evaluate: {len(to_run)}", flush=True)

    print(f"Loading video-SALMONN 2+ model: {model_path}", flush=True)
    print(f"  max_frames={max_frames}, max_pixels={max_pixels}", flush=True)
    print(f"  repo_dir={_REPO_DIR}", flush=True)

    model = video_SALMONN2_plus.from_pretrained(
        str(model_path),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        model_max_length=131072,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )

    from transformers import WhisperFeatureExtractor
    from qwenvl.data.dataset import make_supervised_data_module
    from qwenvl.train.argument import DataArguments

    def _load_image_processor(model_path_str):
        from transformers import AutoImageProcessor

        try:
            from transformers import Qwen2VLImageProcessor
            proc = Qwen2VLImageProcessor.from_pretrained(
                model_path_str, local_files_only=True, use_fast=False
            )
            print("[image_proc] Using Qwen2VLImageProcessor (slow, supports video)", flush=True)
            return proc
        except Exception as e:
            print(f"[image_proc] Qwen2VLImageProcessor failed: {e}", flush=True)

        try:
            from transformers import Qwen2VLVideoProcessor
            proc = Qwen2VLVideoProcessor.from_pretrained(
                model_path_str, local_files_only=True
            )
            print("[image_proc] Using Qwen2VLVideoProcessor", flush=True)
            return proc
        except Exception as e:
            print(f"[image_proc] Qwen2VLVideoProcessor failed: {e}", flush=True)

        from transformers import Qwen2VLImageProcessorFast
        proc = Qwen2VLImageProcessorFast.from_pretrained(
            model_path_str, local_files_only=True
        )
        print("[image_proc] Fallback to Qwen2VLImageProcessorFast (video deprecated)", flush=True)
        return proc

    data_args = DataArguments()
    data_args.video_max_frames = max_frames
    data_args.video_min_frames = 16
    data_args.base_interval    = 0.1
    data_args.max_pixels       = max_pixels
    data_args.video_max_frame_pixels = max_pixels
    data_args.run_test         = True
    data_args.image_processor  = _load_image_processor(str(model_path))
    data_args.audio_processor  = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]

    devices = {p.device for p in model.parameters()}
    print(f"[OK] Model loaded. Parameters on: {devices}", flush=True)

    cfg = {
        "videos_dir":  videos_dir,
        "max_mib":     max_mib,
        "max_tokens":  max_tokens,
        "max_frames":  max_frames,
        "max_pixels":  max_pixels,
        "model":       model,
        "tokenizer":   tokenizer,
        "dataset":     dataset,
    }

    save_json(config_path, {
        "script":            "eval_salmonn.py",
        "model_path":        str(model_path),
        "cleaned_dir":       str(cleaned_dir),
        "videos_dir":        str(videos_dir),
        "output_dir":        str(run_dir),
        "max_tokens":        max_tokens,
        "max_frames":        max_frames,
        "max_pixels":        max_pixels,
        "max_file_mib":      max_mib,
        "item_count":        len(pool),
        "cuda_visible_devices": _CUDA_DEVICES,
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        try:
            rows = []
            for m in results_obj:
                d = asdict(m)

                rows.append({k: (None if v is dataclasses.MISSING else v) for k, v in d.items()})
            save_json(detailed_path, rows)
            save_json(summary_path,  compute_summary(results_obj, run_id))
        except Exception as e:
            print(f"[WARNING] _flush failed: {e}\n{traceback.format_exc()}", flush=True)

    results: List[EvaluationMetrics] = []
    _first_errors_shown = 0
    for it in tqdm(to_run, desc="LV-Bench (video-SALMONN 2+)"):
        res_obj = evaluate_one(it, cfg)
        results.append(res_obj)
        if not res_obj.api_success and _first_errors_shown < 5:
            print(f"\n[ERROR] qid={res_obj.question_id} task={res_obj.task_type}: {res_obj.error_message}", flush=True)
            _first_errors_shown += 1
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
