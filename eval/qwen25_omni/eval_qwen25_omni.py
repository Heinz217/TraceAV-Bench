#!/usr/bin/env python3
"""LV-Bench MCQ eval client against an OpenAI-compatible chat endpoint (full benchmark)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

MODEL_DEFAULT = "qwen25-omni"

TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _retry_sleep_sec(attempt: int) -> float:
    import random
    return min(10.0, 0.8 * (2 ** (attempt - 1))) + random.random() * 0.2


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


def parse_openai_assistant_text(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict):
        return ""
    for ch in resp_json.get("choices", []) or []:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message") or {}
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") in (None, "text"):
                        t = blk.get("text")
                        if isinstance(t, str) and t.strip():
                            return t.strip()
    err = resp_json.get("error")
    if isinstance(err, dict):
        m = err.get("message") or err.get("code")
        if m is not None:
            return f"[error] {m}"
    return ""


def hop_metadata_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    oids = item.get("original_event_ids")
    hop_original_event_count = len(oids) if isinstance(oids, list) else None
    minute_hop_count = None
    minute_raw = item.get("minute_hop_count")
    if minute_raw is not None:
        try:
            minute_hop_count = int(minute_raw)
        except (ValueError, TypeError):
            minute_hop_count = None
    hop_length_label = item.get("hop_length_label")
    hop_length_label = str(hop_length_label) if hop_length_label is not None else None
    span = None
    traj = item.get("trajectory_with_timestamps")
    if isinstance(traj, list):
        vals: List[float] = []
        for ev in traj:
            if not isinstance(ev, dict):
                continue
            tr = ev.get("event_time_range") or {}
            for k in ("start_minute", "end_minute"):
                v = tr.get(k)
                if v is not None:
                    vals.append(float(v))
            ts = ev.get("timestamp_minute")
            if ts is not None:
                vals.append(float(ts))
        if len(vals) >= 2:
            span = max(vals) - min(vals)
    return {
        "hop_original_event_count": hop_original_event_count,
        "minute_hop_count": minute_hop_count,
        "hop_length_label": hop_length_label,
        "hop_evidence_span_minutes": span,
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
    hop_original_event_count: Optional[int] = None
    minute_hop_count: Optional[int] = None
    hop_length_label: Optional[str] = None
    hop_evidence_span_minutes: Optional[float] = None
    processing_time_sec: float = 0.0
    raw_model_text: Optional[str] = None
    local_video_path: Optional[str] = None
    http_status: Optional[int] = None


def item_key(it: Dict[str, Any]) -> Tuple[str, int]:
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
    by_task: Dict[str, Dict[str, Any]] = {}
    for tt in sorted({m.task_type for m in metrics}):
        xs = [m for m in metrics if m.task_type == tt]
        c = sum(1 for m in xs if m.correct)
        by_task[tt] = {"count": len(xs), "correct": c,
                       "accuracy": c / len(xs) if xs else 0.0}
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
                raise FileNotFoundError(f"Missing task file: {self.cleaned_dir / (tt + '.json')}")
            with open(available[tt], "r", encoding="utf-8") as f:
                task_data = json.load(f)
            for item in task_data.get("items", []) or []:
                row = dict(item)
                row["task_type"] = tt
                pool.append(row)
        return pool


def resolve_local_video(benchmark_dir: Optional[Path], videos_dir: Path, video_id: str) -> Optional[Path]:
    rel = str(video_id).replace("\\", "/").lstrip("/")
    bn = Path(rel).name
    candidates: List[Path] = []
    if benchmark_dir is not None and rel.startswith("videos/"):
        candidates.append(benchmark_dir / rel)
    candidates.extend([videos_dir / rel, videos_dir / bn])
    seen: Set[str] = set()
    for c in candidates:
        try:
            rp = c.resolve()
        except OSError:
            continue
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return rp
    return None


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".webm":
        return "video/webm"
    if ext == ".mov":
        return "video/quicktime"
    return "video/mp4"


def post_chat_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_sec: int,
    max_attempts: int = 5,
) -> Tuple[requests.Response, Optional[str]]:
    last_exc: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if r.status_code in TRANSIENT_HTTP_STATUSES and attempt < max_attempts:
                time.sleep(_retry_sleep_sec(attempt))
                continue
            return r, None
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError) as e:
            last_exc = f"{type(e).__name__}: {e}"
            if attempt >= max_attempts:
                break
            time.sleep(_retry_sleep_sec(attempt))
    fake = requests.Response()
    fake.status_code = 0
    return fake, last_exc or "POST failed"


def evaluate_one(
    item: Dict[str, Any],
    *,
    benchmark_dir: Optional[Path],
    videos_dir: Path,
    chat_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> EvaluationMetrics:
    t0 = time.time()
    hop = hop_metadata_from_item(item)
    qid = int(item.get("question_id", -1))
    task_type = str(item.get("task_type", "unknown"))
    video_id = str(item.get("video_id", ""))
    question = str(item.get("question", ""))
    options = item.get("options", {}) or {}
    qtype = str(item.get("question_type", "single"))
    gold = normalize_option_letters(item.get("correct_options", []))

    def _fail(msg: str, *, http_status: Optional[int] = None) -> EvaluationMetrics:
        return EvaluationMetrics(
            question_id=qid, task_type=task_type, video_id=video_id,
            question_type=qtype, predicted_options=[], correct_options=gold,
            correct=False, abstained=True, reasoning="", api_success=False,
            error_message=msg, processing_time_sec=time.time() - t0,
            http_status=http_status, **hop,
        )

    if not question:
        return _fail("empty question")
    path = resolve_local_video(benchmark_dir, videos_dir, video_id)
    if not path:
        return _fail(f"video not found for video_id={video_id!r}")

    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    mime = guess_mime(path)

    prompt = build_user_prompt(question, {str(k): str(v) for k, v in options.items()})
    content = [
        {"type": "text", "text": prompt},
        {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r, err = post_chat_with_retries(chat_url, headers, payload, timeout_sec=timeout_sec)
    if err:
        return _fail(err)
    if r.status_code >= 400:
        return _fail(f"HTTP {r.status_code}: {(r.text or '')[:2000]}", http_status=r.status_code)

    try:
        body_json = r.json()
    except Exception as e:
        return _fail(f"invalid JSON: {e}", http_status=r.status_code)

    raw_text = parse_openai_assistant_text(body_json)
    pred = parse_response_to_prediction(raw_text) if raw_text.strip() else []
    return EvaluationMetrics(
        question_id=qid, task_type=task_type, video_id=video_id,
        question_type=qtype, predicted_options=pred, correct_options=gold,
        correct=(len(gold) > 0 and pred == gold),
        abstained=(len(pred) == 0),
        reasoning=raw_text,
        api_success=True, error_message=None,
        processing_time_sec=time.time() - t0,
        raw_model_text=raw_text, local_video_path=str(path),
        http_status=r.status_code, **hop,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LV-Bench MCQ eval via OpenAI-compatible chat + inline video")
    p.add_argument("--benchmark-dir", type=Path, default=None,
                   help="LV-Bench root (contains cleaned/ + videos/)")
    p.add_argument("--cleaned-dir", type=Path, default=None,
                   help="Directory of cleaned *.json (default: <benchmark-dir>/cleaned)")
    p.add_argument("--videos-dir", type=Path, default=None,
                   help="Directory of videos (default: <benchmark-dir>)")
    p.add_argument("--tasks", nargs="+", default=[],
                   help="Restrict to these task stems when loading cleaned/")
    p.add_argument("--base-url", type=str,
                   default=os.environ.get("LVBENCH_BASE_URL", ""),
                   help="OpenAI-compatible API root (env: LVBENCH_BASE_URL)")
    p.add_argument("--chat-path", type=str, default="/v1/chat/completions")
    p.add_argument("--model", type=str, default=MODEL_DEFAULT)
    p.add_argument("--api-key", type=str,
                   default=os.environ.get("LVBENCH_API_KEY", ""),
                   help="Bearer token for the chat endpoint (env: LVBENCH_API_KEY)")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout-sec", type=int, default=600)
    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve() if args.benchmark_dir else None

    cleaned_dir = args.cleaned_dir.resolve() if args.cleaned_dir else None
    if cleaned_dir is None and benchmark_dir is not None:
        cleaned_dir = benchmark_dir / "cleaned"
    videos_dir = args.videos_dir.resolve() if args.videos_dir else None
    if videos_dir is None and benchmark_dir is not None:
        videos_dir = benchmark_dir
    if videos_dir is None or not videos_dir.is_dir():
        print("Need --videos-dir and/or --benchmark-dir.", file=sys.stderr)
        return 2
    if cleaned_dir is None or not cleaned_dir.is_dir():
        print("Need --cleaned-dir or --benchmark-dir (with <root>/cleaned present).", file=sys.stderr)
        return 2

    base = (args.base_url or "").strip().rstrip("/")
    if not base:
        print("Need --base-url or LVBENCH_BASE_URL (e.g. http://127.0.0.1:8000).", file=sys.stderr)
        return 2
    chat_path = args.chat_path if args.chat_path.startswith("/") else "/" + args.chat_path
    chat_url = f"{base}{chat_path}"

    api_key = (args.api_key or "").strip()

    loader = BenchmarkLoader(cleaned_dir)
    pool = loader.build_item_pool(args.tasks if args.tasks else None)
    if not pool:
        print("No items to evaluate.", file=sys.stderr)
        return 2

    out_root = (args.output_root or (Path(__file__).resolve().parent / "runs")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_lv_openai_chat")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        run_dir / "run_config.json",
        {
            "script": Path(__file__).name,
            "chat_url": chat_url,
            "model": args.model,
            "benchmark_dir": str(benchmark_dir) if benchmark_dir else None,
            "cleaned_dir": str(cleaned_dir),
            "videos_dir": str(videos_dir),
            "tasks_filter": args.tasks,
            "max_workers": args.max_workers,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout_sec": args.timeout_sec,
            "item_count": len(pool),
        },
    )

    detailed_path = run_dir / "detailed_results.json"
    summary_path = run_dir / "summary.json"
    metrics_by_key: Dict[Tuple[str, int], EvaluationMetrics] = {}
    lock = threading.Lock()

    pbar = tqdm(total=len(pool), desc=f"lv_{args.model}", dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {
            ex.submit(
                evaluate_one, it,
                benchmark_dir=benchmark_dir,
                videos_dir=videos_dir,
                chat_url=chat_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
            ): it
            for it in pool
        }
        for fut in as_completed(futs):
            it = futs[fut]
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
                    correct=False, abstained=True, reasoning="",
                    api_success=False, error_message=str(e),
                    processing_time_sec=0.0,
                    **hop_metadata_from_item(it),
                )
            with lock:
                metrics_by_key[item_key(it)] = m
            pbar.update(1)
    pbar.close()

    ordered = [metrics_by_key[item_key(it)] for it in pool if item_key(it) in metrics_by_key]
    save_json(detailed_path, [asdict(m) for m in ordered])
    s = compute_summary(ordered, run_id)
    save_json(summary_path, s)
    print("=" * 72)
    print(f"Run: {run_id}")
    print(f"Items: {len(ordered)} | accuracy: {s.get('overall_accuracy', 0):.2%} "
          f"| api_ok: {s.get('api_success_rate', 0):.2%}")
    print(f"Output: {run_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
