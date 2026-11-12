"""Shared AV-preserving media sampling for evaluation API clients."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import atexit
import hashlib
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

_MEDIA_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="traceav_media_cache_"))
_MEDIA_CACHE_LOCK = threading.Lock()
atexit.register(shutil.rmtree, _MEDIA_CACHE_ROOT, ignore_errors=True)


def _tool(name: str) -> str:
    value = shutil.which(name)
    if not value:
        raise RuntimeError(f"{name} is required for AV-aware evaluation sampling")
    return value


def probe_media(path: Path, timeout_sec: int = 3600) -> Dict[str, object]:
    proc = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    if proc.returncode:
        raise RuntimeError(f"ffprobe failed: {(proc.stderr or '').strip()[:1500]}")
    import json
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    duration = float((data.get("format") or {}).get("duration") or 0)
    if duration <= 0:
        raise RuntimeError(f"Invalid media duration for {path}: {duration}")
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream in {path}")
    audio = any(s.get("codec_type") == "audio" for s in streams)
    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        num, den = rate_text.split("/", 1)
        source_fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        source_fps = 30.0
    return {"duration_sec": duration, "has_audio": audio, "source_fps": source_fps}


def inspect_encoded_media(path: Path, timeout_sec: int = 3600) -> Dict[str, object]:
    proc = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
            "format=duration:stream=codec_type,width,height,nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    if proc.returncode:
        raise RuntimeError(f"ffprobe failed to inspect sampled input: {(proc.stderr or '').strip()[:1500]}")
    import json
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format") or {}
    return {
        "encoded_video_frame_count": int(video.get("nb_read_frames") or 0),
        "encoded_video_resolution": [int(video.get("width") or 0), int(video.get("height") or 0)],
        "encoded_audio_duration_sec": float(audio.get("duration") or 0),
        "encoded_media_duration_sec": float(fmt.get("duration") or 0),
    }


def model_input_profile(input_ids: object = None, tensors: object = None) -> Dict[str, object]:
    """Record effective token length and tensor shapes without storing tensor values."""
    def _shape(value: object):
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return [int(dim) for dim in shape]
            except (TypeError, ValueError):
                pass
        if isinstance(value, dict):
            return {str(k): _shape(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_shape(v) for v in value]
        return None

    result: Dict[str, object] = {}
    shape = _shape(input_ids)
    if isinstance(shape, list) and shape:
        result["input_sequence_length"] = shape[-1]
        result["input_token_count"] = shape[-1]
    if tensors is not None:
        result["processor_tensor_shapes"] = _shape(tensors)
    return result


def _even_windows(duration: float, budget_sec: int) -> List[Tuple[float, float]]:
    count = max(1, budget_sec // 30)
    if count * 30 >= duration:
        return [(0.0, duration)]
    max_start = duration - 30.0
    starts = [
        round(i * max_start / max(1, count - 1), 3)
        for i in range(count)
    ] if count > 1 else [max_start / 2]
    return [(start, min(30.0, duration - start)) for start in starts]


def _uniform_subsample(items: List[Tuple[float, float]], count: int) -> List[Tuple[float, float]]:
    if count >= len(items):
        return items
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def timeline_coverage_profile(
    windows: List[Tuple[float, float]], duration: float, *, full_track: bool = False,
) -> Dict[str, object]:
    if full_track:
        return {
            "audio_timeline_coverage": "full_track",
            "audio_timeline_coverage_fraction": 1.0,
        }
    intervals = sorted((max(0.0, start), min(duration, start + length)) for start, length in windows)
    covered = 0.0
    right_edge = 0.0
    for start, end in intervals:
        if end <= start:
            continue
        covered += max(0.0, end - max(start, right_edge))
        right_edge = max(right_edge, end)
    fraction = min(1.0, covered / duration) if duration > 0 else 0.0
    if fraction >= 0.999:
        label = "full_timeline_windows"
    elif len(intervals) == 1:
        label = "single_midpoint_window_partial_timeline"
    else:
        label = "uniformly_distributed_partial_timeline_windows"
    return {
        "audio_timeline_coverage": label,
        "audio_timeline_coverage_fraction": fraction,
    }


def fallback_profiles(
    duration: float, frame_budget: int, audio_cap_sec: int = 0,
) -> List[Tuple[int, int, int]]:
    """Resolution → halved frame budgets → synchronized, uniformly sampled AV windows.

    Audio windows are limited by the current frame budget. This keeps video and
    audio on matching source intervals and never raises the frame budget again.
    """
    initial_audio = 0
    if audio_cap_sec > 0 and duration > audio_cap_sec:
        initial_audio = (audio_cap_sec // 30) * 30
    levels: List[Tuple[int, int, int]] = [
        (edge, frame_budget, initial_audio) for edge in (1280, 960, 640, 480)
    ]
    counts: List[int] = []
    current = frame_budget
    while current > 1:
        current //= 2
        if current not in counts:
            counts.append(current)
            levels.append((480, current, initial_audio))
    minimum_frames = counts[-1] if counts else frame_budget
    seconds = (initial_audio if initial_audio > 0 else int(duration)) // 2
    seconds = (seconds // 30) * 30
    seen_budgets = set()
    while seconds >= 30:
        effective_audio = min(seconds, max(30, minimum_frames * 30))
        if effective_audio not in seen_budgets:
            levels.append((480, minimum_frames, effective_audio))
            seen_budgets.add(effective_audio)
        seconds = ((seconds // 2) // 30) * 30
    # Avoid repeated retries when the synchronized-window frame cap already
    # reduces several nominal audio budgets to the same selected AV windows.
    deduped: List[Tuple[int, int, int]] = []
    seen_profiles = set()
    for edge, frames, budget in levels:
        if budget <= 0:
            windows_key = "full_track"
        else:
            candidate_windows = _even_windows(duration, budget)
            selected = _uniform_subsample(candidate_windows, min(frames, len(candidate_windows)))
            windows_key = tuple((round(start, 3), round(length, 3)) for start, length in selected)
        signature = (edge, frames, windows_key)
        if signature not in seen_profiles:
            deduped.append((edge, frames, budget))
            seen_profiles.add(signature)
    return deduped


def is_capacity_error(status_code: int, error_text: str) -> bool:
    if status_code == 413:
        return True
    message = (error_text or "").lower()
    return any(token in message for token in (
        "too large", "exceeds the maximum", "context length", "input is too long",
        "too many tokens", "maximum number of frames", "payload size",
        "out of memory", "cuda oom", "sequence length",
    ))


def _run(argv: List[str], timeout_sec: int) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: {(proc.stderr or '').strip()[:1500]}")


@contextmanager
def prepare_av_video(
    source: Path,
    *,
    frame_budget: int,
    long_edge: int,
    audio_budget_sec: int = 0,
    timeout_sec: int = 3600,
) -> Iterator[Tuple[Path, Dict[str, object]]]:
    """Sample AV uniformly; fallback windows cut matching audio/video intervals."""
    info = probe_media(source, timeout_sec)
    duration = float(info["duration_sec"])
    if not info["has_audio"]:
        raise RuntimeError(f"No audio track in AV benchmark input: {source}")
    if frame_budget < 1 or long_edge < 1:
        raise ValueError("frame_budget and long_edge must be positive")
    windows = [(0.0, duration)] if audio_budget_sec <= 0 else _even_windows(duration, audio_budget_sec)
    requested_window_count = len(windows)
    if audio_budget_sec > 0:
        # Preserve full-timeline coverage if there are more audio windows than
        # video frames. Never take a prefix of the candidate window list.
        windows = _uniform_subsample(windows, min(len(windows), frame_budget))
    quotas = [frame_budget // len(windows)] * len(windows)
    for i in range(frame_budget % len(windows)):
        quotas[i] += 1
    quotas = [max(1, q) for q in quotas]

    key_material = (
        f"{source.resolve()}|{source.stat().st_size}|{source.stat().st_mtime_ns}|"
        f"{frame_budget}|{long_edge}|{audio_budget_sec}"
    )
    cache_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    output_dir = _MEDIA_CACHE_ROOT / cache_key
    output = output_dir / "sampled_av.mp4"
    ffmpeg = _tool("ffmpeg")
    with _MEDIA_CACHE_LOCK:
        if not output.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            work = output_dir
            segments: List[Path] = []
            try:
                for index, ((start, length), quota) in enumerate(zip(windows, quotas)):
                    segment = work / f"segment_{index:04d}.mp4"
                    fps = min(max(float(info["source_fps"]), 0.001), quota / max(length, 0.001))
                    scale = (
                        f"fps={fps:.10f},"
                        f"scale=w=min({long_edge}\\,iw):h=min({long_edge}\\,ih):"
                        "force_original_aspect_ratio=decrease"
                    )
                    _run(
                        [
                            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", f"{start:.3f}", "-i", str(source),
                            "-t", f"{length:.3f}", "-map", "0:v:0", "-map", "0:a:0",
                            "-vf", scale, "-frames:v", str(quota),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "48k",
                            "-ar", "16000", "-ac", "1", "-movflags", "+faststart", str(segment),
                        ],
                        timeout_sec,
                    )
                    segments.append(segment)

                if len(segments) == 1:
                    shutil.copy2(segments[0], output)
                else:
                    concat_list = work / "segments.txt"
                    concat_list.write_text(
                        "".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8"
                    )
                    _run(
                        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                         "-safe", "0", "-i", str(concat_list), "-c", "copy", "-movflags",
                         "+faststart", str(output)],
                        timeout_sec,
                    )
            except Exception:
                shutil.rmtree(output_dir, ignore_errors=True)
                raise
    profile = {
        "fallback_policy": "resolution_then_frame_halving_then_synchronized_av_window_budget_halving; selected_timeline_coverage_is_recorded",
        "long_edge_px": long_edge,
        "frame_budget": sum(quotas),
            "requested_audio_budget_sec": int(audio_budget_sec or duration),
            "audio_budget_sec": int(sum(length for _, length in windows)),
        "requested_audio_window_count": requested_window_count,
        "encoded_audio_window_count": len(windows),
        "audio_window_selection": (
            "full_track" if audio_budget_sec <= 0 else
            "single_midpoint_30s_window_partial_timeline" if len(windows) == 1 and sum(length for _, length in windows) < duration * 0.999 else
            "uniform_30s_windows" if len(windows) == requested_window_count else
            "uniform_30s_windows_capped_by_frame_budget"
        ),
        "audio_windows": windows,
    }
    profile.update(timeline_coverage_profile(windows, duration, full_track=audio_budget_sec <= 0))
    profile.update(inspect_encoded_media(output, timeout_sec))
    yield output, profile
