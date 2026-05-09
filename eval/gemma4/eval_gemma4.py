#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["LD_PRELOAD"] = ""

import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*torchcodec.*")
_warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
_warnings.filterwarnings("ignore", message=".*audioread.*")
_warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
_warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import av

import torch
from tqdm import tqdm

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("GEMMA4_MODEL_PATH")
CLEANED_DIR = _require_env("GEMMA4_CLEANED_DIR")
VIDEOS_DIR  = _require_env("GEMMA4_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("GEMMA4_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

SETTINGS: Dict[str, Any] = {
    "model_path":          MODEL_PATH,
    "cleaned_dir":         CLEANED_DIR,
    "videos_dir":          VIDEOS_DIR,
    "output_file":         f"{OUTPUT_DIR}/gemma4_results_{datetime.now().strftime('%m%d_%H%M')}.json",
    "max_tokens":          512,
    "load_audio_from_video": True,                 
    "num_video_frames":    32,                   
    "max_file_mib":        5000,
}

@dataclass
class EvaluationMetrics:
    question_id:             int
    task_type:               str
    video_id:                str
    question_type:           str
    predicted_options:       List[str]
    correct_options:         List[str]
    correct:                 bool
    abstained:               bool
    reasoning:               str
    api_success:             bool
    error_message:           Optional[str]  = None
    processing_time_sec:     float          = 0.0
    raw_model_text:          Optional[str]  = None
    local_video_path:        Optional[str]  = None
    http_status:             Optional[int]  = None
    hop_original_event_count: Optional[int] = None
    minute_hop_count:        Optional[int]  = None
    hop_length_label:        Optional[str]  = None
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
        "hop_original_event_count":    len(oids) if isinstance(oids, list) else None,
        "minute_hop_count":            item.get("minute_hop_count"),
        "hop_length_label":            str(item.get("hop_length_label")) if item.get("hop_length_label") else None,
        "hop_evidence_span_minutes":   span,
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
        "run_id":            run_id,
        "timestamp":         datetime.now().isoformat(),
        "total_items":       total,
        "correct":           correct,
        "api_failed":        api_failed,
        "overall_accuracy":  correct / total if total else 0.0,
        "api_success_rate":  (total - api_failed) / total if total else 0.0,
        "by_task_type":      by_task,
    }

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

        max_mib = cfg["max_mib"]
        if max_mib > 0 and (path.stat().st_size / 1048576) > max_mib:
            return _fail(f"File too large: {path.stat().st_size} bytes")

        opts         = item.get("options", {})
        options_text = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        prompt = (
            "You are given a video. Base your answer only on what you see and hear.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else.\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        processor    = cfg["processor"]
        model        = cfg["model"]
        device       = cfg["device"]
        max_tokens   = cfg["max_tokens"]
        num_frames   = cfg["num_frames"]

        frames: List[Any] = []
        with av.open(str(path)) as container:
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream is None:
                return _fail("No video stream found")

            total = video_stream.frames or 0

            if total > 0:
                indices = set(
                    int(round(i * (total - 1) / (num_frames - 1)))
                    for i in range(num_frames)
                ) if num_frames > 1 else {0}
            else:
                indices = None                     

            frame_idx = 0
            for frame in container.decode(video=0):
                if indices is None:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) >= num_frames:
                        break
                else:
                    if frame_idx in indices:
                        frames.append(frame.to_ndarray(format="rgb24"))
                    frame_idx += 1
                    if len(frames) >= num_frames:
                        break

        if not frames:
            return _fail("No frames decoded from video")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text",  "text":  prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated  = output_ids[0][input_len:]
        raw_text   = processor.decode(generated, skip_special_tokens=True).strip()
        pred       = parse_response_to_prediction(raw_text)

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
    max_mib         = float(s["max_file_mib"])
    load_audio      = bool(s.get("load_audio_from_video", True))

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
    out_root = Path(OUTPUT_DIR)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S_lv_gemma4")
    run_dir  = out_root / "eval_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = run_dir / "detailed_results.json"
    summary_path  = run_dir / "summary.json"
    config_path   = run_dir / "run_config.json"
    to_run = pool
    print(f"🚀 Loading  Gemma 4 E4B-it model: {model_path}", flush=True)

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
        _ModelCls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModel
        _ModelCls = AutoModel

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
            "CUDA is not available. See the [CUDA] diagnostics above.\n"
            "  1. --device /dev/nvidiaN correctly mounted into the container?\n"
            "  2. CUDA_VISIBLE_DEVICES set correctly?\n"
            "  3. libcuda.so / libnvidia-ml.so mounted?\n"
            "  4. Does the torch version match the CUDA driver?"
        )
    device = "cuda"

    model = _ModelCls.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},                                                   
        local_files_only=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    num_frames = int(s.get("num_video_frames", 32))

    cfg = {
        "videos_dir":          videos_dir,
        "max_mib":             max_mib,
        "model":               model,
        "processor":           processor,
        "device":              device,
        "max_tokens":          max_tokens,
        "load_audio_from_video": load_audio,
        "num_frames":          num_frames,
    }

    save_json(config_path, {
        "script":              "eval_gemma4.py",
        "model_path":          str(model_path),
        "cleaned_dir":         str(cleaned_dir),
        "videos_dir":          str(videos_dir),
        "output_dir":          str(run_dir),
        "max_tokens":          max_tokens,
        "load_audio_from_video": load_audio,
        "num_video_frames":    num_frames,
        "max_file_mib":        max_mib,
        "item_count":          len(pool),
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        save_json(detailed_path, [asdict(m) for m in results_obj])
        save_json(summary_path,  compute_summary(results_obj, run_id))

    results: List[EvaluationMetrics] = []
    fields = EvaluationMetrics.__dataclass_fields__
    for it in tqdm(to_run, desc="LV-Bench (Gemma4-E4B)"):
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
