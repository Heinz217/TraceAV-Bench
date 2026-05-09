#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("DECORD_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHCODEC", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["LD_PRELOAD"] = ""

import json
import re
import time

import traceback
try:
    import ujson
except ImportError:
    ujson = json                
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("BAICHUAN_MODEL_PATH")
CLEANED_DIR = _require_env("BAICHUAN_CLEANED_DIR")
VIDEOS_DIR  = _require_env("BAICHUAN_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("BAICHUAN_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))
CACHE_DIR   = os.environ.get("BAICHUAN_CACHE_DIR", str(Path(__file__).resolve().parent / "cache"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

_MAX_FRAMES_DEFAULT    = int(os.environ.get("BAICHUAN_MAX_FRAMES",    "512"))
_MAX_PIXELS_DEFAULT    = int(os.environ.get("BAICHUAN_MAX_PIXELS",    "200704"))                               

SETTINGS: Dict[str, Any] = {
    "model_path":        MODEL_PATH,
    "cleaned_dir":       CLEANED_DIR,
    "videos_dir":        VIDEOS_DIR,
    "output_file":       f"{OUTPUT_DIR}/baichuan_omni_results_{datetime.now().strftime('%m%d_%H%M')}.json",
    "max_tokens":        512,
    "max_file_mib":      5000,
    "max_frames":        _MAX_FRAMES_DEFAULT,
    "max_pixels":        _MAX_PIXELS_DEFAULT,
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
            return _fail(f"File too large: {path.stat().st_size}")

        opts = item.get("options", {})
        options_text = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        question_text = (
            "You are given a video. Base your answer only on what you see and hear.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else.\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        model     = cfg["model"]
        tokenizer = cfg["tokenizer"]
        max_new_tokens = cfg["max_tokens"]

        video_token_start = tokenizer.convert_ids_to_tokens(
            model.config.video_config.video_start_token_id
        )
        video_token_end = tokenizer.convert_ids_to_tokens(
            model.config.video_config.video_end_token_id
        )
        video_tag = (
            video_token_start
            + ujson.dumps({"local": str(path)}, ensure_ascii=False)
            + video_token_end
        )

        system_content = "You are a helpful assistant."
        content = f"<|im_start|>system\n{system_content}<|im_end|>\n"
        content += f"<|im_start|>user\n{video_tag}{question_text}<|im_end|>\n"
        content += "<|im_start|>assistant\n"

        pret = model.processor([content])

        first_device = next(model.parameters()).device
        videos = (
            [torch.tensor(v, dtype=torch.float32).to(first_device) for v in pret.videos]
            if pret.videos is not None else None
        )

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=pret.input_ids.to(first_device),
                    attention_mask=pret.attention_mask.to(first_device) if pret.attention_mask is not None else None,
                    videos=videos,
                    videos_patch_nums=pret.videos_patch_nums if pret.videos_patch_nums is not None else None,
                    videos_grid=pret.videos_grid if pret.videos_grid is not None else None,
                    images=None,
                    audios=None,
                    encoder_length=pret.encoder_length.to(first_device) if pret.encoder_length is not None else None,
                    bridge_length=pret.bridge_length.to(first_device) if pret.bridge_length is not None else None,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=False,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            err_str = str(oom_err)
            if "out of memory" in err_str.lower() or isinstance(oom_err, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                return _fail(f"CUDA OOM: {oom_err}", status=0)
            raise

        prompt_len = pret.input_ids.shape[1]
        new_token_ids = output_ids[0, prompt_len:]

        raw_text_full = tokenizer.decode(new_token_ids, skip_special_tokens=False)
        raw_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if stop in raw_text:
                raw_text = raw_text[:raw_text.index(stop)]
        raw_text = raw_text.strip()
        if not raw_text:

            print(f"[warn] qid={qid} empty output; full_tokens={raw_text_full!r}", flush=True)
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
    max_mib         = float(s["max_file_mib"])
    max_frames      = int(s.get("max_frames", 512))     
    max_pixels      = int(s.get("max_pixels", 200704))

    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("image", "audio", "video"):
        (cache_dir / sub).mkdir(exist_ok=True)

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
    out_root = Path(OUTPUT_DIR)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S_lv_baichuan_omni")
    run_dir  = out_root / "eval_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = run_dir / "detailed_results.json"
    summary_path  = run_dir / "summary.json"
    config_path   = run_dir / "run_config.json"
    to_run = pool
    print(f"🚀 Loading  Baichuan-Omni-1.5 model: {model_path}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",                            
        device_map="balanced",                                 
        local_files_only=True,
    )
    model.training = False

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    orig_max_frame_num = getattr(model.config.video_config, "max_frame_num", None)
    orig_max_pixels    = getattr(model.config.video_config, "max_pixels",    None)
    model.config.video_config.max_frame_num = max_frames
    model.config.video_config.max_pixels    = max_pixels
    print(f"[config] video_config.max_frame_num: {orig_max_frame_num} → {max_frames}", flush=True)
    print(f"[config] video_config.max_pixels:    {orig_max_pixels}    → {max_pixels}", flush=True)

    model.bind_processor(tokenizer, training=False, relative_path=str(cache_dir))

    model.eval()

    cfg = {
        "videos_dir": videos_dir,
        "max_mib":    max_mib,
        "max_tokens": max_tokens,
        "max_frames": max_frames,
        "model":      model,
        "tokenizer":  tokenizer,
    }

    save_json(config_path, {
        "script": "eval_baichuan_omni.py",
        "model_path": str(model_path),
        "cleaned_dir": str(cleaned_dir),
        "videos_dir": str(videos_dir),
        "output_dir": str(run_dir),
        "max_tokens": max_tokens,
        "max_file_mib": max_mib,
        "max_frames": max_frames,
        "max_pixels": max_pixels,
        "item_count": len(pool),
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        save_json(detailed_path, [asdict(m) for m in results_obj])
        save_json(summary_path,  compute_summary(results_obj, run_id))

    results: List[EvaluationMetrics] = []
    for it in tqdm(to_run, desc="LV-Bench (Baichuan-Omni-1.5)"):
        res_obj = evaluate_one(it, cfg)
        results.append(res_obj)
        _flush(results)
        torch.cuda.empty_cache()

    _flush(results)
    sm = compute_summary(results, run_id)
    print("=" * 72)
    print(f"Run:      {run_id}")
    print(f"Items:    {len(results)} | accuracy: {sm.get('overall_accuracy', 0):.2%} | api_ok: {sm.get('api_success_rate', 0):.2%}")
    print(f"Output:   {run_dir}")
    print("=" * 72)

if __name__ == "__main__":
    main()
