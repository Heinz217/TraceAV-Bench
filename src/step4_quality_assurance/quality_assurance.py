import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

from quality_assurance_prompts import (
    DIRECT_ANSWER_FALLBACK,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    VERIFICATION_FALLBACK,
    VERIFICATION_SYSTEM_PROMPT,
    build_direct_answer_user_prompt,
    build_verification_user_prompt,
)


@dataclass
class QAConfig:
    input_dir: Path
    output_root: Path
    model: str
    base_url: str
    api_key: str
    workers: int
    max_items_per_task: int
    drop_failed_verification: bool
    tasks: List[str]


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_client(cfg: QAConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120.0, max_retries=3)


def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return fallback


def normalize_option_letters(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for x in raw:
        c = str(x).strip().upper()
        if c in {"A", "B", "C", "D"} and c not in seen:
            out.append(c)
            seen.add(c)
    return sorted(out)


def normalize_confidence(raw: Any) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def list_task_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.json"))


def run_blindfold_detection_for_item(client: OpenAI, cfg: QAConfig, item: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_direct_answer_user_prompt(
        question=str(item.get("question", "")),
        options=item.get("options", {}) or {},
    )
    pred = call_llm_json(
        client=client,
        model=cfg.model,
        system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        fallback=DIRECT_ANSWER_FALLBACK,
    )
    predicted = normalize_option_letters(pred.get("predicted_options"))
    confidence = normalize_confidence(pred.get("confidence", 0.0))
    correct = normalize_option_letters(item.get("correct_options"))
    abstained = len(predicted) == 0
    blindfold_hit_direct = (predicted == correct and len(correct) > 0)
    blindfold_hit = blindfold_hit_direct
    return {
        "question_id": item.get("question_id"),
        "predicted_options": predicted,
        "correct_options": correct,
        "abstained": abstained,
        "confidence": confidence,
        "blindfold_hit_direct": blindfold_hit_direct,
        "blindfold_hit": blindfold_hit,
        "reason": pred.get("reason", ""),
    }


def run_logical_verification_for_item(client: OpenAI, cfg: QAConfig, task_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_verification_user_prompt(
        task_type=task_type,
        question=str(item.get("question", "")),
        options=item.get("options", {}) or {},
        question_type=str(item.get("question_type", "single")),
        correct_options=normalize_option_letters(item.get("correct_options")),
        answer_text=str(item.get("answer_text", "")),
        trajectory_with_timestamps=item.get("trajectory_with_timestamps", []) or [],
    )
    raw = call_llm_json(
        client=client,
        model=cfg.model,
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        fallback=VERIFICATION_FALLBACK,
    )

    option_quality = raw.get("option_quality", {}) if isinstance(raw.get("option_quality"), dict) else {}
    should_drop = bool(raw.get("should_drop", False))
    drop_reasons = raw.get("drop_reasons", [])
    if not isinstance(drop_reasons, list):
        drop_reasons = [str(drop_reasons)]

    verdict = {
        "is_real_multihop": bool(raw.get("is_real_multihop", True)),
        "multihop_issue": str(raw.get("multihop_issue", "")),
        "is_pseudo_multihop": bool(raw.get("is_pseudo_multihop", False)),
        "pseudo_multihop_reason": str(raw.get("pseudo_multihop_reason", "")),
        "option_quality": {
            "has_obvious_pattern": bool(option_quality.get("has_obvious_pattern", False)),
            "distractors_relevant": bool(option_quality.get("distractors_relevant", True)),
            "distractors_plausible": bool(option_quality.get("distractors_plausible", True)),
            "overall_quality": str(option_quality.get("overall_quality", "fair")),
            "reason": str(option_quality.get("reason", "")),
        },
        "answer_leakage_detected": bool(raw.get("answer_leakage_detected", False)),
        "answer_leakage_reason": str(raw.get("answer_leakage_reason", "")),
        "should_drop": should_drop,
        "drop_reasons": drop_reasons,
        "severity": str(raw.get("severity", "low")),
        "summary": str(raw.get("summary", "")),
    }
    return verdict


def process_task_file(
    cfg: QAConfig,
    task_file: Path,
    progress_position: int,
) -> Tuple[str, Dict[str, Any]]:
    task_payload = read_json(task_file)
    task_type = str(task_payload.get("task_type", task_file.stem))
    items = task_payload.get("items", []) or []
    if cfg.max_items_per_task > 0:
        items = items[: cfg.max_items_per_task]

    client = get_client(cfg)

    stage_blindfold_records: List[Dict[str, Any]] = []
    kept_after_blindfold: List[Dict[str, Any]] = []
    removed_by_blindfold = 0
    abstained_in_blindfold = 0
    blindfold_hit_direct_count = 0
    with tqdm(
        total=len(items),
        desc=f"{task_type} [blindfold]",
        position=progress_position,
        leave=False,
        dynamic_ncols=True,
    ) as pbar_blindfold:
        for item in items:
            rec = run_blindfold_detection_for_item(client, cfg, item)
            stage_blindfold_records.append(rec)
            if rec.get("abstained", False):
                abstained_in_blindfold += 1
            if rec.get("blindfold_hit_direct", False):
                blindfold_hit_direct_count += 1
            if rec["blindfold_hit"]:
                removed_by_blindfold += 1
            else:
                kept_after_blindfold.append(item)
            pbar_blindfold.update(1)
            pbar_blindfold.set_postfix(
                kept=len(kept_after_blindfold),
                removed=removed_by_blindfold,
                abstained=abstained_in_blindfold,
                hit_direct=blindfold_hit_direct_count,
                refresh=False,
            )

    stage_blindfold_payload = {
        "task_type": task_type,
        "source_file": str(task_file),
        "total_items_input": len(items),
        "removed_by_blindfold": removed_by_blindfold,
        "abstained_in_blindfold": abstained_in_blindfold,
        "blindfold_hit_direct": blindfold_hit_direct_count,
        "remaining_after_blindfold": len(kept_after_blindfold),
        "records": stage_blindfold_records,
        "items_after_blindfold": kept_after_blindfold,
    }

    stage_logical_records: List[Dict[str, Any]] = []
    final_items: List[Dict[str, Any]] = []
    removed_by_logical = 0
    with tqdm(
        total=len(kept_after_blindfold),
        desc=f"{task_type} [logical]",
        position=progress_position,
        leave=False,
        dynamic_ncols=True,
    ) as pbar_logical:
        for item in kept_after_blindfold:
            verdict = run_logical_verification_for_item(client, cfg, task_type, item)
            enriched = dict(item)
            enriched["verification"] = verdict
            stage_logical_records.append(
                {
                    "question_id": item.get("question_id"),
                    "verification": verdict,
                }
            )
            should_drop_now = cfg.drop_failed_verification and verdict.get("should_drop", False)
            if should_drop_now:
                removed_by_logical += 1
            else:
                final_items.append(enriched)
            pbar_logical.update(1)
            pbar_logical.set_postfix(
                kept=len(final_items),
                dropped=removed_by_logical,
                refresh=False,
            )

    for i, x in enumerate(final_items, start=1):
        x["question_id"] = i

    final_payload = {
        "task_type": task_type,
        "video_count": task_payload.get("video_count", len({x.get("video_id") for x in final_items})),
        "question_count": len(final_items),
        "items": final_items,
    }

    stage_logical_payload = {
        "task_type": task_type,
        "input_after_blindfold": len(kept_after_blindfold),
        "removed_by_logical_verification": removed_by_logical,
        "remaining_after_logical_verification": len(final_items),
        "drop_failed_verification": cfg.drop_failed_verification,
        "records": stage_logical_records,
    }

    return task_type, {
        "blindfold_payload": stage_blindfold_payload,
        "logical_payload": stage_logical_payload,
        "final_payload": final_payload,
        "stats": {
            "total_items_input": len(items),
            "removed_by_blindfold": removed_by_blindfold,
            "abstained_in_blindfold": abstained_in_blindfold,
            "blindfold_hit_direct": blindfold_hit_direct_count,
            "removed_by_logical_verification": removed_by_logical,
            "final_count": len(final_items),
        },
    }


def run(cfg: QAConfig) -> None:
    task_files = list_task_files(cfg.input_dir)
    if not task_files:
        raise FileNotFoundError(f"No json files found in {cfg.input_dir}")
    if cfg.tasks:
        wanted = set(cfg.tasks)
        task_files = [p for p in task_files if p.stem in wanted]
        if not task_files:
            raise ValueError(f"No matching task files for --tasks={sorted(wanted)}")

    run_dir = cfg.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    blindfold_dir = run_dir / "blindfold_shortcut_detection"
    logical_dir = run_dir / "logical_verification"
    final_dir = run_dir / "final_cleaned"
    for p in (blindfold_dir, logical_dir, final_dir):
        p.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {
            ex.submit(process_task_file, cfg, fp, idx + 1): fp
            for idx, fp in enumerate(task_files)
        }
        pbar = tqdm(total=len(futures), desc="Quality Assurance tasks", position=0, dynamic_ncols=True)
        for fut in as_completed(futures):
            task_file = futures[fut]
            task_type = task_file.stem
            result = fut.result()
            task_type, payloads = result
            write_json(blindfold_dir / f"{task_type}.json", payloads["blindfold_payload"])
            write_json(logical_dir / f"{task_type}.json", payloads["logical_payload"])
            write_json(final_dir / f"{task_type}.json", payloads["final_payload"])
            summary_rows.append({"task_type": task_type, **payloads["stats"]})
            pbar.update(1)
        pbar.close()

    total_input = sum(x["total_items_input"] for x in summary_rows)
    total_drop_blindfold = sum(x["removed_by_blindfold"] for x in summary_rows)
    total_abstained = sum(x.get("abstained_in_blindfold", 0) for x in summary_rows)
    total_blindfold_hit = sum(x.get("blindfold_hit_direct", 0) for x in summary_rows)
    total_drop_logical = sum(x["removed_by_logical_verification"] for x in summary_rows)
    total_final = sum(x["final_count"] for x in summary_rows)

    run_summary = {
        "input_dir": str(cfg.input_dir),
        "run_dir": str(run_dir),
        "model": cfg.model,
        "drop_failed_verification": cfg.drop_failed_verification,
        "workers": cfg.workers,
        "max_items_per_task": cfg.max_items_per_task,
        "task_stats": sorted(summary_rows, key=lambda x: x["task_type"]),
        "totals": {
            "total_items_input": total_input,
            "removed_by_blindfold": total_drop_blindfold,
            "abstained_in_blindfold": total_abstained,
            "blindfold_hit_direct": total_blindfold_hit,
            "removed_by_logical_verification": total_drop_logical,
            "final_count": total_final,
        },
    }
    write_json(run_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LV-Bench Step 4 Quality Assurance: Blindfold Shortcut Detection + Logical Verification.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./quality_assurance_input"),
        help="Input benchmark directory containing per-task pre-QA json files produced by Step 3 aggregation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./quality_assurance_output"),
        help="Output root directory for all run artifacts.",
    )
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible endpoint URL (or set OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
    )
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument(
        "--max-items-per-task",
        type=int,
        default=0,
        help="If >0, only process first N items per task (debug mode).",
    )
    parser.add_argument(
        "--drop-failed-verification",
        action="store_true",
        help="Drop questions with verification.should_drop=true in final output.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="Optional task list to run, e.g. --tasks av_sequencing av_tracking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")
    cfg = QAConfig(
        input_dir=args.input_dir,
        output_root=args.output_root,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        workers=max(1, args.workers),
        max_items_per_task=max(0, args.max_items_per_task),
        drop_failed_verification=bool(args.drop_failed_verification),
        tasks=[str(x).strip() for x in args.tasks if str(x).strip()],
    )
    run(cfg)


if __name__ == "__main__":
    main()
