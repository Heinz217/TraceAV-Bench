#!/usr/bin/env python3
from __future__ import annotations

import faulthandler
faulthandler.enable()                                                     

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
from qwen_vl_utils import process_vision_info                

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Please set it via the matching .sh launcher."
        )
    return v

MODEL_PATH  = _require_env("QWEN3VL_MODEL_PATH")
CLEANED_DIR = _require_env("QWEN3VL_CLEANED_DIR")
VIDEOS_DIR  = _require_env("QWEN3VL_VIDEOS_DIR")
OUTPUT_DIR  = os.environ.get("QWEN3VL_OUTPUT_DIR", str(Path(__file__).resolve().parent / "runs"))

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH!r}")

SETTINGS: Dict[str, Any] = {
    "model_path":      MODEL_PATH,
    "cleaned_dir":     CLEANED_DIR,
    "videos_dir":      VIDEOS_DIR,
    "output_dir":      OUTPUT_DIR,
    "max_tokens":      512,
    "max_file_mib":    5000,
    "video_fps":       float(os.environ.get("QWEN3VL_VIDEO_FPS",   "1.0")),
    "max_pixels":      int(os.environ.get("QWEN3VL_MAX_PIXELS",    str(360 * 420))),

    "nframes":         int(os.environ.get("QWEN3VL_NFRAMES",        "64")),
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

def item_key(it: Dict[str, Any]) -> tuple:
    return (str(it.get("task_type", "")), int(it.get("question_id", -1)))

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

        max_mib: float = cfg["max_mib"]
        if max_mib > 0 and (path.stat().st_size / 1048576) > max_mib:
            return _fail(f"File too large: {path.stat().st_size} bytes")

        opts = item.get("options", {})
        options_text = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items()))
        prompt = (
            "You are given a video. Base your answer only on what you see.\n\n"
            "Directly provide the letter representing your choice (A/B/C/D) and nothing else. "
            "Do not include the full text of the option; do not provide any explanation. "
            "The problem could be a single-choice question or a multiple-choice question. "
            "If multiple options are correct, return letters joined by commas (example: A,C).\n\n"
            f"Question:\n{item.get('question', '')}\n\n"
            f"Options:\n{options_text}"
        )

        processor  = cfg["processor"]
        model      = cfg["model"]
        nframes:    int   = cfg["nframes"]
        max_pixels: int   = cfg["max_pixels"]

        from PIL import Image as _PILImage

        frames: List[Any] = []
        n = max(1, nframes) if nframes > 0 else 32

        def _resize(pil: Any) -> Any:
            w, h = pil.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                pil = pil.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    _PILImage.BILINEAR,
                )
            return pil

        try:
            import decord as _decord
            _decord.bridge.set_bridge("native")
            vr = _decord.VideoReader(str(path), ctx=_decord.cpu(0))
            total = len(vr)
            if total <= n:
                idx = list(range(total))
            else:
                idx = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
            raw = vr.get_batch(idx).asnumpy()                
            frames = [_resize(_PILImage.fromarray(raw[i])) for i in range(len(idx))]
        except Exception as _de:

            try:
                import av as _av
                with _av.open(str(path)) as container:
                    vs = next((s for s in container.streams if s.type == "video"), None)
                    if vs is None:
                        return _fail("No video stream found")

                    dur = container.duration                                   
                    if dur and dur > 0:
                        targets = [int(dur * i / n) for i in range(n)]
                        for t in targets:
                            try:
                                container.seek(t)
                                for frame in container.decode(video=0):
                                    frames.append(_resize(
                                        _PILImage.fromarray(
                                            frame.to_ndarray(format="rgb24")
                                        )
                                    ))
                                    break
                            except Exception:
                                pass
                    else:

                        for frame in container.decode(video=0):
                            frames.append(_resize(
                                _PILImage.fromarray(frame.to_ndarray(format="rgb24"))
                            ))
                            if len(frames) >= n:
                                break
            except Exception as _ae:
                return _fail(f"Video decode error: decord={_de}; av={_ae}")

        if not frames:
            return _fail("No frames decoded from video")

        _video_fps = float(max(0.5, cfg.get("video_fps", 1.0)))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type":  "video",
                        "video": frames,

                        "sample_fps": _video_fps,
                        "raw_fps":    _video_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs_raw = process_vision_info(
            messages, return_video_metadata=True
        )

        if video_inputs_raw is not None:
            video_tensors = [v[0] for v in video_inputs_raw]
            video_metas   = [v[1] for v in video_inputs_raw]
        else:
            video_tensors = None
            video_metas   = None

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_tensors,
            video_metadata=video_metas,                              
            do_sample_frames=False,                      
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        _ids_shape = inputs.get("input_ids", torch.empty(0)).shape
        _vid_shape = inputs.get("pixel_values_videos", torch.empty(0)).shape
        print(f"  [infer] qid={qid} input_ids={list(_ids_shape)} "
              f"pixel_values_videos={list(_vid_shape)} device={device}",
              flush=True)

        try:
            _gen_t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_tokens"],
                    do_sample=False,
                )
            print(f"  [infer] qid={qid} generate took {time.time()-_gen_t0:.1f}s", flush=True)
        except torch.cuda.OutOfMemoryError as oom_e:
            torch.cuda.empty_cache()
            return _fail(f"CUDA OOM: {oom_e}")

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        raw_text  = processor.decode(generated, skip_special_tokens=True).strip()
        pred      = parse_response_to_prediction(raw_text)

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

    pool: List[Dict[str, Any]] = []
    for f in sorted(cleaned_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as j:
            for it in json.load(j).get("items", []):
                it["task_type"] = f.stem
                pool.append(it)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S_lv_qwen3vl")
    run_dir = output_dir / "eval_results" / run_id
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
            "  1. CUDA_VISIBLE_DEVICES set correctly?\n"
            "  2. nvidia-smi able to see the GPU?\n"
            "  3. Does the torch version match the CUDA driver?"
        )

    print(f"🚀 Loading  Qwen3-VL-32B-Instruct: {model_path}", flush=True)
    print(f"   GPU count: {torch.cuda.device_count()}，device_map=auto", flush=True)

    from transformers import AutoProcessor

    trust_remote = os.environ.get("QWEN3VL_TRUST_REMOTE_CODE", "1") != "0"
    print(f"[model] trust_remote_code={trust_remote}", flush=True)

    try:
        from transformers import Qwen3VLForConditionalGeneration
        _ModelCls = Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText
            _ModelCls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModel
            _ModelCls = AutoModel

    try:
        import flash_attn              
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = _ModelCls.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=trust_remote,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote,
        local_files_only=True,
    )

    cfg = {
        "videos_dir": videos_dir,
        "max_mib":    max_mib,
        "model":      model,
        "processor":  processor,
        "max_tokens": max_tokens,
        "video_fps":  s["video_fps"],
        "nframes":    s["nframes"],
        "max_pixels": s["max_pixels"],
    }

    save_json(config_path, {
        "script":          "eval_qwen3vl.py",
        "model_path":      str(model_path),
        "cleaned_dir":     str(cleaned_dir),
        "videos_dir":      str(videos_dir),
        "output_dir":      str(run_dir),
        "max_tokens":      max_tokens,
        "max_file_mib":    max_mib,
        "video_fps":       s["video_fps"],
        "nframes":         s["nframes"],
        "max_pixels":      s["max_pixels"],
        "item_count":      len(pool),
        "cuda_devices":    torch.cuda.device_count(),
        "attn_impl":       attn_impl,
    })

    def _flush(results_obj: List[EvaluationMetrics]) -> None:
        save_json(detailed_path, [asdict(m) for m in results_obj])
        save_json(summary_path,  compute_summary(results_obj, run_id))

    results: List[EvaluationMetrics] = []
    fields = EvaluationMetrics.__dataclass_fields__
    for it in tqdm(to_run, desc="LV-Bench (Qwen3-VL-32B)"):
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
