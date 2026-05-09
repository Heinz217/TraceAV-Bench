#!/usr/bin/env python3
"""LV-Bench evaluator for Gemini-compatible generateContent endpoints (full benchmark)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

GOOGLE_API_BASE = "https://generativelanguage.googleapis.com"
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"})

# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

class BenchmarkLoader:
    def __init__(self, cleaned_dir: Path) -> None:
        self.cleaned_dir = cleaned_dir

    def build_item_pool(self, task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []
        files = sorted(self.cleaned_dir.glob("*.json"))
        available = {p.stem: p for p in files}
        targets = task_types if task_types else sorted(available.keys())
        for tt in targets:
            if tt not in available:
                raise FileNotFoundError(f"Task file not found: {self.cleaned_dir / (tt + '.json')}")
            with open(available[tt], "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("items", []) or []:
                enriched = dict(item)
                enriched["task_type"] = tt
                pool.append(enriched)
        return pool

def item_key(it: Dict[str, Any]) -> Tuple[str, int]:
    return (str(it.get("task_type", "")), int(it.get("question_id", -1)))

# ---------------------------------------------------------------------------
# Prompt / answer parsing
# ---------------------------------------------------------------------------

def normalize_option_letters(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for x in raw:
        c = str(x).strip().upper()
        if c in {"A", "B", "C", "D"} and c not in seen:
            seen.add(c)
            out.append(c)
    return sorted(out)

def build_user_prompt(question: str, options: Dict[str, str]) -> str:
    options_text = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    return (
        "You are given a video. Base your answer only on what you see and hear.\n\n"
        "Directly provide the letter representing your choice (A/B/C/D) and nothing else. "
        "Do not include the full text of the option; do not provide any explanation. "
        "The problem could be a single-choice question or a multiple-choice question. "
        "If multiple options are correct, return letters joined by commas (example: A,C).\n\n"
        f"Question:\n{question}\n\nOptions:\n{options_text}"
    )

def parse_response_to_prediction(text: str) -> List[str]:
    s = (text or "").strip()
    if s.startswith("```json"):
        s = s[7:].strip()
    if s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    try:
        obj = json.loads(s)
        ans = obj.get("answer")
        if isinstance(ans, list):
            return normalize_option_letters(ans)
        if isinstance(ans, str):
            return normalize_option_letters(re.findall(r"[A-D]", ans.upper()))
    except Exception:
        pass
    m = re.search(r"Answer:\s*([A-D](?:\s*[,;/]\s*[A-D])*)", text or "", re.IGNORECASE | re.DOTALL)
    if m:
        return normalize_option_letters(re.findall(r"[A-D]", m.group(1).upper()))
    return normalize_option_letters(re.findall(r"\b([A-D])\b", (text or "").upper()))

def parse_response_text(resp_json: Dict[str, Any]) -> str:
    """Extract assistant text from Gemini generateContent response."""
    if not isinstance(resp_json, dict):
        return ""
    for cand in resp_json.get("candidates", []) or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list):
            texts = [p.get("text") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
            joined = "\n".join(t for t in texts if t and t.strip()).strip()
            if joined:
                return joined
    return ""

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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

def compute_summary(metrics: List[EvaluationMetrics], run_id: str) -> Dict[str, Any]:
    if not metrics:
        return {}
    total = len(metrics)
    correct = sum(1 for m in metrics if m.correct)
    api_failed = sum(1 for m in metrics if not m.api_success)
    by_task: Dict[str, Dict[str, Any]] = {}
    for tt in sorted({m.task_type for m in metrics}):
        xs = [m for m in metrics if m.task_type == tt]
        c = sum(1 for m in xs if m.correct)
        by_task[tt] = {"count": len(xs), "correct": c, "accuracy": c / len(xs) if xs else 0.0}
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

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Media (ffmpeg frames + audio) -> base64 inlineData parts
# ---------------------------------------------------------------------------

def _ffmpeg_bin() -> str:
    w = shutil.which("ffmpeg")
    if not w:
        raise RuntimeError("ffmpeg not found in PATH (needed for video frame + audio extraction)")
    return w

def _run_ffmpeg(argv: List[str], *, timeout_sec: int) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: {err[:2000]}")

def _inline_part(raw: bytes, mime: str) -> Dict[str, Any]:
    return {"inlineData": {"mimeType": mime, "data": base64.standard_b64encode(raw).decode("ascii")}}

def build_media_parts(
    video_path: Path,
    *,
    long_edge: int,
    fps: float,
    max_frames: int,
    ffmpeg_timeout_sec: int,
    include_audio: bool,
    jpeg_q: int,
    audio_bitrate_kbps: int,
) -> List[Dict[str, Any]]:
    ff = _ffmpeg_bin()
    parts: List[Dict[str, Any]] = []
    vf = (
        f"fps={fps},"
        f"scale=w=min({long_edge}\\,iw):h=min({long_edge}\\,ih):force_original_aspect_ratio=decrease"
    )
    with tempfile.TemporaryDirectory(prefix="lvbench_inline_") as tdir:
        wd = Path(tdir)
        _run_ffmpeg(
            [
                ff, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path),
                "-vf", vf,
                "-q:v", str(jpeg_q),
                str(wd / "frame_%06d.jpg"),
            ],
            timeout_sec=ffmpeg_timeout_sec,
        )
        frames = sorted(wd.glob("frame_*.jpg"))
        if max_frames > 0:
            frames = frames[:max_frames]
        for fp in frames:
            parts.append(_inline_part(fp.read_bytes(), "image/jpeg"))

        if include_audio:
            audio_path = wd / "audio.m4a"
            try:
                _run_ffmpeg(
                    [
                        ff, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(video_path),
                        "-vn",
                        "-c:a", "aac",
                        "-b:a", f"{audio_bitrate_kbps}k",
                        str(audio_path),
                    ],
                    timeout_sec=ffmpeg_timeout_sec,
                )
            except RuntimeError:
                audio_path.unlink(missing_ok=True)
            if audio_path.exists() and audio_path.stat().st_size >= 256:
                parts.append(_inline_part(audio_path.read_bytes(), "audio/mp4"))

    return parts

# ---------------------------------------------------------------------------
# generateContent call
# ---------------------------------------------------------------------------

def build_payload(
    media_parts: List[Dict[str, Any]],
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: Optional[int],
) -> Dict[str, Any]:
    gen_cfg: Dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    if thinking_budget is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return {
        "contents": [{"role": "user", "parts": list(media_parts) + [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }

def call_generate(
    *,
    generate_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_sec: int,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    if "generativelanguage.googleapis.com" not in generate_url:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(generate_url, headers=headers, json=payload, timeout=timeout_sec)
    if r.status_code >= 400:
        raise RuntimeError(f"generateContent HTTP {r.status_code}: {r.text[:2000]}")
    return r.json()

# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def resolve_video_path(benchmark_dir: Path, videos_dir: Path, video_id: str) -> Path:
    rel = str(video_id).replace("\\", "/").lstrip("/")
    if rel.startswith("videos/"):
        return benchmark_dir / rel
    return videos_dir / rel

def evaluate_one(
    item: Dict[str, Any],
    *,
    benchmark_dir: Path,
    videos_dir: Path,
    generate_url: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
    generate_timeout_sec: int,
    media_long_edge: int,
    media_fps: float,
    max_inline_frames: int,
    ffmpeg_timeout_sec: int,
    include_audio: bool,
    jpeg_q: int,
    audio_bitrate_kbps: int,
    thinking_budget: Optional[int],
) -> EvaluationMetrics:
    t0 = time.time()
    question_id = int(item.get("question_id", -1))
    task_type = str(item.get("task_type", "unknown"))
    video_id = str(item.get("video_id", ""))
    question = str(item.get("question", ""))
    options = item.get("options", {}) or {}
    question_type = str(item.get("question_type", "single"))
    correct = normalize_option_letters(item.get("correct_options", []))

    def _fail(msg: str) -> EvaluationMetrics:
        return EvaluationMetrics(
            question_id=question_id,
            task_type=task_type,
            video_id=video_id,
            question_type=question_type,
            predicted_options=[],
            correct_options=correct,
            correct=False,
            abstained=True,
            reasoning="",
            api_success=False,
            error_message=msg,
            processing_time_sec=time.time() - t0,
        )

    if not question:
        return _fail("empty question")
    path = resolve_video_path(benchmark_dir, videos_dir, video_id)
    if not path.exists():
        return _fail(f"media not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return _fail(f"unsupported media type: {path.suffix}")

    try:
        media_parts = build_media_parts(
            path,
            long_edge=media_long_edge,
            fps=media_fps,
            max_frames=max_inline_frames,
            ffmpeg_timeout_sec=ffmpeg_timeout_sec,
            include_audio=include_audio,
            jpeg_q=jpeg_q,
            audio_bitrate_kbps=audio_bitrate_kbps,
        )
    except Exception as e:
        return _fail(f"ffmpeg: {e}")

    if not media_parts:
        return _fail("no media parts produced")

    prompt = build_user_prompt(question, {str(k): str(v) for k, v in options.items()})
    payload = build_payload(
        media_parts,
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
    )
    try:
        resp = call_generate(
            generate_url=generate_url,
            api_key=api_key,
            payload=payload,
            timeout_sec=generate_timeout_sec,
        )
    except Exception as e:
        return _fail(str(e))

    raw_text = parse_response_text(resp)
    predicted = parse_response_to_prediction(raw_text) if raw_text.strip() else []
    return EvaluationMetrics(
        question_id=question_id,
        task_type=task_type,
        video_id=video_id,
        question_type=question_type,
        predicted_options=predicted,
        correct_options=correct,
        correct=(len(correct) > 0 and predicted == correct),
        abstained=(len(predicted) == 0),
        reasoning=raw_text,
        api_success=True,
        error_message=None,
        processing_time_sec=time.time() - t0,
        raw_model_text=raw_text,
        local_video_path=str(path),
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LV-Bench Gemini generateContent evaluator")
    p.add_argument("--benchmark-dir", type=Path, required=True,
                   help="Root containing cleaned/*.json and videos/")
    p.add_argument("--cleaned-dir", type=Path, default=None,
                   help="Override cleaned dir (default: <benchmark-dir>/cleaned)")
    p.add_argument("--videos-dir", type=Path, default=None,
                   help="Override videos dir (default: <benchmark-dir>)")
    p.add_argument("--tasks", nargs="+", default=[],
                   help="Restrict to these task stems (matches cleaned/<stem>.json)")

    p.add_argument("--model", type=str, default="gemini-2.5-pro")
    p.add_argument("--generate-url", type=str, default=None,
                   help="Full URL to generateContent endpoint; "
                        "default = https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent")
    p.add_argument("--api-key", type=str,
                   default=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""))

    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--generate-timeout-sec", type=int, default=600)
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="generationConfig.thinkingConfig.thinkingBudget (omit for API default)")

    p.add_argument("--media-long-edge", type=int, default=360,
                   help="Max width/height of sampled frames (px)")
    p.add_argument("--media-fps", type=float, default=1.0,
                   help="Video frame sampling rate for ffmpeg")
    p.add_argument("--max-inline-frames", type=int, default=0,
                   help="Cap number of JPEG frames sent (0 = no cap)")
    p.add_argument("--jpeg-q", type=int, default=8,
                   help="ffmpeg mjpeg -q:v (higher = more compression)")
    p.add_argument("--audio-kbps", type=int, default=48,
                   help="AAC bitrate for extracted audio")
    p.add_argument("--ffmpeg-timeout-sec", type=int, default=3600)
    p.add_argument("--no-video-audio", action="store_true",
                   help="Do not send the extracted audio track")

    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    cleaned_dir = (args.cleaned_dir or (benchmark_dir / "cleaned")).resolve()
    videos_dir = (args.videos_dir or benchmark_dir).resolve()
    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Missing cleaned dir: {cleaned_dir}")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing videos dir: {videos_dir}")
    if args.media_fps <= 0:
        raise ValueError("--media-fps must be > 0")

    api_key = (args.api_key or "").strip()
    if not api_key:
        raise ValueError("Missing --api-key (or GEMINI_API_KEY / GOOGLE_API_KEY)")
    generate_url = (args.generate_url or "").strip()
    if not generate_url:
        generate_url = f"{GOOGLE_API_BASE}/v1beta/models/{args.model}:generateContent"

    loader = BenchmarkLoader(cleaned_dir)
    pool = loader.build_item_pool(args.tasks if args.tasks else None)

    if not pool:
        raise ValueError("No items to evaluate")

    output_root = (args.output_root or (benchmark_dir / "eval_results")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_gemini")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        run_dir / "run_config.json",
        {
            "script": "eval_gemini.py",
            "benchmark_dir": str(benchmark_dir),
            "cleaned_dir": str(cleaned_dir),
            "videos_dir": str(videos_dir),
            "tasks": args.tasks,
            "model": args.model,
            "generate_url": generate_url,
            "max_workers": args.max_workers,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "generate_timeout_sec": args.generate_timeout_sec,
            "thinking_budget": args.thinking_budget,
            "media_long_edge": args.media_long_edge,
            "media_fps": args.media_fps,
            "max_inline_frames": args.max_inline_frames,
            "jpeg_q": args.jpeg_q,
            "audio_kbps": args.audio_kbps,
            "ffmpeg_timeout_sec": args.ffmpeg_timeout_sec,
            "include_audio": not args.no_video_audio,
            "item_count": len(pool),
        },
    )

    detailed_path = run_dir / "detailed_results.json"
    summary_path = run_dir / "summary.json"

    metrics_by_key: Dict[Tuple[str, int], EvaluationMetrics] = {}
    lock = threading.Lock()

    pbar = tqdm(total=len(pool), desc="eval_gemini", dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {
            ex.submit(
                evaluate_one,
                it,
                benchmark_dir=benchmark_dir,
                videos_dir=videos_dir,
                generate_url=generate_url,
                api_key=api_key,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                generate_timeout_sec=args.generate_timeout_sec,
                media_long_edge=args.media_long_edge,
                media_fps=args.media_fps,
                max_inline_frames=args.max_inline_frames,
                ffmpeg_timeout_sec=args.ffmpeg_timeout_sec,
                include_audio=not args.no_video_audio,
                jpeg_q=args.jpeg_q,
                audio_bitrate_kbps=args.audio_kbps,
                thinking_budget=args.thinking_budget,
            ): it
            for it in pool
        }
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                m = fut.result()
            except Exception as e:
                m = EvaluationMetrics(
                    question_id=int(it.get("question_id", -1)),
                    task_type=str(it.get("task_type", "unknown")),
                    video_id=str(it.get("video_id", "")),
                    question_type=str(it.get("question_type", "single")),
                    predicted_options=[],
                    correct_options=normalize_option_letters(it.get("correct_options", [])),
                    correct=False,
                    abstained=True,
                    reasoning="",
                    api_success=False,
                    error_message=str(e),
                    processing_time_sec=0.0,
                )
            with lock:
                metrics_by_key[item_key(it)] = m
            pbar.update(1)
    pbar.close()

    ordered = [metrics_by_key[item_key(it)] for it in pool if item_key(it) in metrics_by_key]
    save_json(detailed_path, [asdict(m) for m in ordered])
    summary = compute_summary(ordered, run_id)
    save_json(summary_path, summary)
    print("=" * 72)
    print(f"Run: {run_id}")
    print(f"Items: {len(ordered)} | accuracy: {summary.get('overall_accuracy', 0):.2%} "
          f"| api_ok: {summary.get('api_success_rate', 0):.2%}")
    print(f"Output: {run_dir}")
    print("=" * 72)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
