#!/usr/bin/env python3
from __future__ import annotations

import faulthandler
faulthandler.enable()

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["LD_PRELOAD"] = ""

import warnings
warnings.filterwarnings("ignore")

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import librosa
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("QWEN2AUDIO_MODEL_PATH")
CLEANED_DIR = _require_env("QWEN2AUDIO_CLEANED_DIR")
VIDEOS_DIR  = _require_env("QWEN2AUDIO_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("QWEN2AUDIO_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

SETTINGS: Dict[str, Any] = {
    "model_path":      MODEL_PATH,
    "cleaned_dir":     CLEANED_DIR,
    "videos_dir":      VIDEOS_DIR,
    "output_dir":      OUTPUT_DIR,
    "max_tokens":      512,
    "max_file_mib":    5000,
    "max_audio_sec":   int(os.environ.get("QWEN2AUDIO_MAX_SEC", "600")),
}

@dataclass
class EvaluationMetrics:
    question_id:              int
    task_type:                str
    video_id:                 str
    question_type:            str
    predicted_options:        List[str]
    correct_options:          List[str]
    correct:                  bool
    abstained:                bool
    reasoning:                str
    api_success:              bool
    error_message:            Optional[str]  = None
    processing_time_sec:      float          = 0.0
    raw_model_text:           Optional[str]  = None
    local_video_path:         Optional[str]  = None
    http_status:              Optional[int]  = None
    hop_original_event_count: Optional[int]  = None
    minute_hop_count:         Optional[int]  = None
    hop_length_label:         Optional[str]  = None
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

def extract_audio_from_video(video_path: str, target_sr: int, max_sec: int = 0):
    """Extract an audio waveform from a video file. Returns a numpy array of shape (samples,).

    Prefer loading via librosa (with the ffmpeg backend);
    falls back to av decoding + manual resampling on failure.
    """
    import numpy as np

    try:
        waveform, sr = librosa.load(video_path, sr=target_sr, mono=True)
        if max_sec > 0:
            max_samples = target_sr * max_sec
            waveform = waveform[:max_samples]
        return waveform
    except Exception:
        pass

    try:
        import av
        container = av.open(video_path)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            return None

        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)        
            frames.append(arr)
            if max_sec > 0 and len(frames) * len(arr) / frame.sample_rate > max_sec * 1.1:
                break
        container.close()

        if not frames:
            return None

        waveform = np.concatenate(frames).astype(np.float32)

        orig_sr = audio_stream.rate or 16000
        if orig_sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        if max_sec > 0:
            waveform = waveform[:target_sr * max_sec]
        return waveform
    except Exception:
        return None

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
            "You are given an audio track from a video. Base your answer only on what you hear.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else. "
            "Do not include the full text of the option; do not provide any explanation. "
            "The problem could be a single-choice question or a multiple-choice question. "
            "If multiple options are correct, return letters joined by commas (example: A,C).\n\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        processor  = cfg["processor"]
        model      = cfg["model"]
        max_tokens: int = cfg["max_tokens"]
        target_sr: int  = cfg["target_sr"]
        max_audio_sec: int = cfg["max_audio_sec"]

        waveform = extract_audio_from_video(str(path), target_sr, max_audio_sec)
        if waveform is None or len(waveform) == 0:
            return _fail("No audio track found in video")

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(path)},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[waveform], return_tensors="pt", padding=True)

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        try:
            _gen_t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            print(f"  [infer] qid={qid} generate took {time.time()-_gen_t0:.1f}s", flush=True)
        except torch.cuda.OutOfMemoryError as oom_e:
            torch.cuda.empty_cache()
            return _fail(f"CUDA OOM: {oom_e}")

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        raw_text  = processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
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
    max_tokens      = int(s["max_tokens"])
    max_mib         = float(s["max_file_mib"])
    max_audio_sec   = int(s.get("max_audio_sec", 600))

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S_lv_qwen2audio")
    run_dir = output_dir / "eval_results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = run_dir / "detailed_results.json"
    summary_path  = run_dir / "summary.json"
    config_path   = run_dir / "run_config.json"
    to_run = pool
    print("*** Mode: AUDIO ONLY - extract audio track from video, ignore visuals ***", flush=True)

    print(f"[CUDA] torch.__version__       = {torch.__version__}", flush=True)
    print(f"[CUDA] torch.cuda.is_available = {torch.cuda.is_available()}", flush=True)
    print(f"[CUDA] CUDA_VISIBLE_DEVICES    = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)
    print(f"[CUDA] device_count            = {torch.cuda.device_count()}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"🚀 Loading  Qwen2-Audio-7B-Instruct: {model_path}", flush=True)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    target_sr = processor.feature_extractor.sampling_rate

    cfg = {
        "videos_dir":    videos_dir,
        "max_mib":       max_mib,
        "model":         model,
        "processor":     processor,
        "max_tokens":    max_tokens,
        "target_sr":     target_sr,
        "max_audio_sec": max_audio_sec,
    }

    save_json(config_path, {
        "script":          "eval_qwen2audio.py",
        "model_path":      str(model_path),
        "cleaned_dir":     str(cleaned_dir),
        "videos_dir":      str(videos_dir),
        "output_dir":      str(run_dir),
        "max_tokens":      max_tokens,
        "max_file_mib":    max_mib,
        "max_audio_sec":   max_audio_sec,
        "target_sr":       target_sr,
        "item_count":      len(pool),
        "cuda_devices":    torch.cuda.device_count(),
        "modality":        "audio_only",
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        save_json(detailed_path, [asdict(m) for m in results_obj])
        save_json(summary_path,  compute_summary(results_obj, run_id))

    results: List[EvaluationMetrics] = []
    fields = EvaluationMetrics.__dataclass_fields__
    for it in tqdm(to_run, desc="LV-Bench (Qwen2-Audio)"):
        res_obj = evaluate_one(it, cfg)
        results.append(res_obj)
        _flush(results)

    _flush(results)
    sm = compute_summary(results, run_id)
    print("=" * 72)
    print(f"Run:      {run_id}")
    print("Mode:     AUDIO ONLY (from video)")
    print(f"Items:    {len(results)} | accuracy: {sm.get('overall_accuracy', 0):.2%} | api_ok: {sm.get('api_success_rate', 0):.2%}")
    print(f"Output:   {run_dir}")
    print("=" * 72)

if __name__ == "__main__":
    main()
