#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DECORD_NUM_THREADS", "1")
os.environ["LD_PRELOAD"] = ""

_REPO_DIR_MOUNTED  = os.environ.get("VL2AV_REPO_DIR")
if not _REPO_DIR_MOUNTED:
    raise SystemExit(
        "Required environment variable 'VL2AV_REPO_DIR' is not set. "
        "Please set it via the matching .sh launcher (path to cloned VideoLLaMA2 repo)."
    )
_REPO_DIR_WRITABLE = "/tmp/VideoLLaMA2"
_REPO_DIR          = _REPO_DIR_WRITABLE

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tqdm import tqdm

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("VL2AV_MODEL_PATH")
CLEANED_DIR = _require_env("VL2AV_CLEANED_DIR")
VIDEOS_DIR  = _require_env("VL2AV_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("VL2AV_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

SETTINGS: Dict[str, Any] = {
    "model_path":      MODEL_PATH,
    "cleaned_dir":     CLEANED_DIR,
    "videos_dir":      VIDEOS_DIR,
    "max_tokens":      512,
    "max_file_mib":    5000,
    "modal":           "av",
}

@dataclass
class EvaluationMetrics:
    question_id:               int
    task_type:                 str
    video_id:                  str
    question_type:             str
    predicted_options:         List[str]
    correct_options:           List[str]
    correct:                   bool
    abstained:                 bool
    reasoning:                 str
    api_success:               bool
    error_message:             Optional[str]   = None
    processing_time_sec:       float           = 0.0
    raw_model_text:            Optional[str]   = None
    local_video_path:          Optional[str]   = None
    http_status:               Optional[int]   = None
    hop_original_event_count:  Optional[int]   = None
    minute_hop_count:          Optional[int]   = None
    hop_length_label:          Optional[str]   = None
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
        "hop_original_event_count":  len(oids) if isinstance(oids, list) else None,
        "minute_hop_count":          item.get("minute_hop_count"),
        "hop_length_label":          str(item.get("hop_length_label")) if item.get("hop_length_label") else None,
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

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_summary(metrics: List[EvaluationMetrics], run_id: str) -> Dict[str, Any]:
    if not metrics:
        return {}
    total      = len(metrics)
    correct    = sum(1 for m in metrics if m.correct)
    api_failed = sum(1 for m in metrics if not m.api_success)
    by_task: Dict[str, Any] = {}
    for tt in sorted({m.task_type for m in metrics}):
        sub = [m for m in metrics if m.task_type == tt]
        c   = sum(1 for m in sub if m.correct)
        by_task[tt] = {"count": len(sub), "correct": c,
                       "accuracy": c / len(sub) if sub else 0.0}
    return {
        "run_id":           run_id,
        "timestamp":        datetime.now().isoformat(),
        "total_items":      total,
        "correct":          correct,
        "api_failed":       api_failed,
        "overall_accuracy": correct / total if total else 0.0,
        "api_success_rate": (total - api_failed) / total if total else 0.0,
        "by_task_type":     by_task,
    }

def evaluate_one(item: Dict[str, Any], cfg: Dict[str, Any]) -> EvaluationMetrics:
    t0  = time.time()
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
            return _fail(f"File too large: {path.stat().st_size}")

        opts         = item.get("options", {})
        options_text = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        instruct = (
            "You are given a video. Base your answer only on what you see and hear.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else.\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        processor      = cfg["processor"]
        model          = cfg["model"]
        tokenizer      = cfg["tokenizer"]
        modal          = cfg["modal"]                   
        max_new_tokens = cfg["max_tokens"]

        video_path_str = str(path)

        preprocess = processor["video"]
        try:
            if modal == "av":
                tensor = preprocess(video_path_str, va=True)
            else:
                tensor = preprocess(video_path_str, va=False)
        except Exception:

            modal = "v"
            tensor = preprocess(video_path_str, va=False)

        raw_text = mm_infer(
            tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal="video",                                    
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

        pred = parse_response_to_prediction(raw_text)

        return EvaluationMetrics(
            qid, task, vid,
            str(item.get("question_type", "single")),
            pred, gold,
            (pred == gold and len(gold) > 0),
            not bool(pred),
            raw_text, True, None,
            time.time() - t0, raw_text, video_path_str, None, **hop,
        )

    except Exception:
        return _fail(traceback.format_exc())

def main() -> None:
    s = SETTINGS
    model_path      = Path(s["model_path"])
    cleaned_dir     = Path(s["cleaned_dir"])
    videos_dir      = Path(s["videos_dir"])
    max_tokens      = int(s["max_tokens"])
    max_mib         = float(s["max_file_mib"])
    modal           = str(s["modal"])

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
    out_root = Path(OUTPUT_DIR)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S_lv_videollama2av")
    run_dir  = out_root / "eval_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = run_dir / "detailed_results.json"
    summary_path  = run_dir / "summary.json"
    config_path   = run_dir / "run_config.json"
    to_run = pool
    print(f"[CUDA] torch.__version__       = {torch.__version__}", flush=True)
    print(f"[CUDA] torch.cuda.is_available = {torch.cuda.is_available()}", flush=True)
    print(f"[CUDA] CUDA_VISIBLE_DEVICES    = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)
    try:
        print(f"[CUDA] device_count            = {torch.cuda.device_count()}", flush=True)
        print(f"[CUDA] torch.version.cuda      = {torch.version.cuda}", flush=True)
    except Exception as _e:
        print(f"[CUDA] diagnostic failed: {_e}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check：\n"
            "  1. --device /dev/nvidiaN mounted into the container?\n"
            "  2. CUDA_VISIBLE_DEVICES set correctly?\n"
            "  3. libcuda.so / libnvidia-ml.so mounted?\n"
            "  4. Does the torch version match the CUDA driver?"
        )

    n_gpus = torch.cuda.device_count()

    print(f"🚀 Loading  VideoLLaMA2.1-7B-AV model: {model_path}", flush=True)
    disable_torch_init()

    if n_gpus > 1:
        device_map = "auto"
        print(f"[device] multi-GPU mode, device_map=auto（total {n_gpus} ）", flush=True)
    else:
        device_map = {"": "cuda:0"}
        print("[device] single-GPU mode, device_map={'': 'cuda:0'}", flush=True)

    model, processor, tokenizer = model_init(
        str(model_path),
        device_map=device_map,
    )
    model.eval()

    if modal == "v":
        model.model.audio_tower = None
    cfg = {
        "videos_dir": videos_dir,
        "max_mib":    max_mib,
        "max_tokens": max_tokens,
        "modal":      modal,
        "model":      model,
        "processor":  processor,
        "tokenizer":  tokenizer,
    }

    save_json(config_path, {
        "script":          "eval_videollama2av.py",
        "model_path":      str(model_path),
        "cleaned_dir":     str(cleaned_dir),
        "videos_dir":      str(videos_dir),
        "output_dir":      str(run_dir),
        "max_tokens":      max_tokens,
        "modal":           modal,
        "max_file_mib":    max_mib,
        "item_count":      len(pool),
        "n_gpus":          n_gpus,
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        save_json(detailed_path, [asdict(m) for m in results_obj])
        save_json(summary_path,  compute_summary(results_obj, run_id))

    results: List[EvaluationMetrics] = []
    for it in tqdm(to_run, desc="LV-Bench (VideoLLaMA2.1-7B-AV)"):
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
