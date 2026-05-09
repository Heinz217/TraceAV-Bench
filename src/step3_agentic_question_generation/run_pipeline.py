import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from agentic_question_generation_prompts import (
    TASK_PROMPTS,
    TASK_TYPES_AUDIO_ONLY,
    TASK_TYPES_AV,
    TASK_TYPES_HALLUCINATION,
    TASK_TYPES_VISUAL_ONLY,
)
from common import (
    PipelineConfig,
    build_client,
    read_json,
    write_json,
)
from trajectory_proposal_agent import (
    filter_trajectories,
    propose_trajectories,
    task_setup,
)
from question_generation_agent import (
    generate_questions,
    verify_questions,
)


PIPELINE_STAGES = ["setup", "propose", "filter", "generate", "verify"]


def run_pipeline(cfg: PipelineConfig, pbar: "tqdm | None" = None) -> None:
    def _tick(label: str) -> None:
        if pbar is not None:
            pbar.set_postfix_str(label)
            pbar.update(1)

    qa_path = cfg.output_dir / "qa_generation/stage3_qa_generation.json"
    verified_path = cfg.output_dir / "qa_generation/stage3_qa_verified.json"
    if verified_path.exists():
        if pbar is not None:
            pbar.update(len(PIPELINE_STAGES))
        return

    source = read_json(cfg.input_json)
    source["event_blocks"] = (source.get("event_blocks") or [])

    if qa_path.exists():
        qa_generation = read_json(qa_path)
        for lbl in ("setup: cached", "propose: cached", "filter: cached", "generate: loaded disk"):
            _tick(lbl)
    else:
        client = build_client(cfg)

        _tick("setup")
        task_context = task_setup(cfg, source)
        write_json(cfg.output_dir / "qa_generation/stage2_setup.json", task_context)

        _tick("propose: trajectory proposal agent")
        proposal = propose_trajectories(client, cfg, source, task_context)
        write_json(cfg.output_dir / "qa_generation/stage2_proposal.json", proposal)

        _tick("filter: trajectory filter")
        filtered = filter_trajectories(source, proposal, cfg)
        write_json(cfg.output_dir / "qa_generation/stage2_filtered.json", filtered)

        _tick("generate: question generation agent")
        qa_generation = generate_questions(client, cfg, source, filtered)
        write_json(qa_path, qa_generation)

    _tick("verify: label & choice verification")
    verified = verify_questions(qa_generation, cfg)
    write_json(verified_path, verified)


def discover_verified_paths(output_root: Path) -> List[Path]:
    seen: Set[Path] = set()

    def video_key(p: Path) -> str:
        return str(p.parent.parent.resolve())

    patterns = (
        "*/*/qa_generation/stage3_qa_verified.json",
        "*/qa_generation/stage3_qa_verified.json",
        "*/*/qa_generation/stage3_qa_generation.json",
        "*/qa_generation/stage3_qa_generation.json",
    )
    for pattern in patterns:
        for p in output_root.glob(pattern):
            seen.add(p.resolve())

    rank = {"stage3_qa_verified.json": 0, "stage3_qa_generation.json": 1}
    best_by_folder: Dict[str, Tuple[int, Path]] = {}
    for p in sorted(seen):
        r = rank.get(p.name)
        if r is None:
            continue
        k = video_key(p)
        if k not in best_by_folder or r < best_by_folder[k][0]:
            best_by_folder[k] = (r, p)
    return sorted(p for _, p in best_by_folder.values())


def shuffle_mcq_options(item: Dict[str, Any]) -> None:
    opts = item.get("options")
    if not isinstance(opts, dict):
        return
    letters = ["A", "B", "C", "D"]
    pairs: List[Tuple[str, str]] = [(k, str(opts.get(k, ""))) for k in letters]
    random.shuffle(pairs)
    new_opts = {letters[i]: pairs[i][1] for i in range(4)}
    old_to_new: Dict[str, str] = {}
    for i, new_l in enumerate(letters):
        old_l, _ = pairs[i]
        old_to_new[old_l] = new_l

    co_raw = item.get("correct_options", [])
    if isinstance(co_raw, str):
        co_raw = [co_raw]
    if not isinstance(co_raw, list):
        co_raw = []
    new_co_set: Set[str] = set()
    for c in co_raw:
        oc = str(c).strip().upper()
        if oc in old_to_new:
            new_co_set.add(old_to_new[oc])
    new_co: List[str] = sorted(new_co_set)

    item["options"] = new_opts
    item["correct_options"] = new_co

    qtype = str(item.get("question_type", "single")).strip().lower()
    if qtype == "multiple" and new_co:
        item["answer_text"] = " | ".join(new_opts.get(c, "") for c in new_co)
    elif new_co:
        item["answer_text"] = new_opts.get(new_co[0], "")


def aggregate_by_task(output_root: Path, target_task: str = None) -> List[Path]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for verified_path in discover_verified_paths(output_root):
        try:
            verified = read_json(verified_path)
        except (OSError, json.JSONDecodeError):
            continue
        video_id = verified.get("video_id", "")

        setup_path = verified_path.parent / "stage2_setup.json"
        task_type = "unknown"
        if setup_path.exists():
            try:
                s0 = read_json(setup_path)
                task_type = str(s0.get("task_type") or "unknown")
            except (OSError, json.JSONDecodeError):
                pass

        if target_task and task_type != target_task:
            continue

        bucket = by_task.setdefault(task_type, [])
        qa_items = verified.get("qa_items", [])
        if not isinstance(qa_items, list):
            qa_items = []

        for qa in qa_items:
            if not isinstance(qa, dict):
                continue
            verification = qa.get("verification", {})
            if isinstance(verification, dict):
                if verification.get("passed") is False:
                    continue
            else:
                continue
            rest = {
                k: v
                for k, v in qa.items()
                if k
                not in (
                    "video_subdir",
                    "output_rel_path",
                    "trajectory_id",
                    "trajectory_id_local",
                    "label_verification",
                    "choice_verification",
                    "verification",
                )
            }
            row: Dict[str, Any] = {"video_id": video_id, **rest}
            bucket.append(row)

    written: List[Path] = []
    for task_type, items in sorted(by_task.items()):
        video_ids = {it.get("video_id") for it in items if it.get("video_id")}
        for idx, it in enumerate(items, start=1):
            shuffle_mcq_options(it)
            new_item = {"question_id": idx}
            new_item.update({k: v for k, v in it.items() if k != "question_id"})
            items[idx - 1] = new_item

        payload = {
            "task_type": task_type,
            "video_count": len(video_ids),
            "question_count": len(items),
            "items": items,
        }
        tt_safe = str(task_type).replace("/", "_")
        agg_dir = output_root / tt_safe / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)
        out_path = agg_dir / f"{tt_safe}.json"
        write_json(out_path, payload)
        written.append(out_path)
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic long-video multi-hop QA synthesis pipeline (Step 3, stages 2-3).")
    parser.add_argument(
        "--input_dir",
        default="./event_blocks",
        help="Directory containing per-video event_blocks JSON files produced by event_segmentation_agent.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="./benchmark",
        help="Root directory. Per-video outputs go to {output_dir}/{task_type}/<video_stem>/qa_generation/.",
    )
    parser.add_argument("--task_type", default=None, choices=sorted(list(TASK_PROMPTS.keys())), help="Run a single specific task.")
    parser.add_argument(
        "--task_group",
        choices=["av", "visual", "audio", "halluc", "all"],
        help="Run a group of tasks based on taxonomy.",
    )
    parser.add_argument("--model", default=os.getenv("QGEN_MODEL", "gpt-5.1-2025-11-13"))
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", ""),
                        help="OpenAI-compatible endpoint URL (or set OPENAI_BASE_URL).")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""),
                        help="API key (or set OPENAI_API_KEY).")
    parser.add_argument("--workers", type=int, default=60, help="Number of parallel worker threads.")
    parser.add_argument("--file-bars", action="store_true", default=False, help="Show per-file stage progress bars.")
    parser.add_argument("--no-file-bars", dest="file_bars", action="store_false", help="Disable per-file stage progress bars.")
    parser.add_argument("--min_hops", type=int, default=2)
    parser.add_argument("--max_hops", type=int, default=10)
    parser.add_argument("--min_span", type=int, default=2)
    parser.add_argument("--max_candidates", type=int, default=10)
    parser.add_argument(
        "--max_questions_per_video",
        type=int,
        default=1,
        help="Maximum number of final questions generated per video.",
    )
    parser.add_argument("--short_hop_max", type=int, default=3)
    parser.add_argument("--medium_hop_max", type=int, default=6)
    parser.add_argument("--debug", type=int, default=0, metavar="N", help="Only process first N input json files (0 = all).")
    parser.add_argument("--no-aggregate", action="store_true", help="Do not write task-specific aggregated JSON after batch run.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only run aggregation from existing outputs under --output_dir; skip pipeline.")
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set --api_key or OPENAI_API_KEY.")
    return args


def process_one(json_path: Path, output_root: Path, args, position: int, total_pbar: "tqdm") -> str:
    stem = json_path.stem
    out_dir = output_root / args.task_type / stem

    file_bar = tqdm(
        total=len(PIPELINE_STAGES),
        desc=stem[:40],
        position=position,
        leave=True,
        unit="stage",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        disable=not args.file_bars,
    )

    if (out_dir / "qa_generation/stage3_qa_verified.json").exists():
        file_bar.set_postfix_str("skipped")
        file_bar.update(len(PIPELINE_STAGES))
        file_bar.close()
        total_pbar.update(1)
        return f"[SKIP] {stem}"

    cfg_obj = PipelineConfig(
        input_json=json_path,
        output_dir=out_dir,
        task_type=args.task_type,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        min_span=args.min_span,
        max_candidates=args.max_candidates,
        max_questions_per_video=args.max_questions_per_video,
        short_hop_max=args.short_hop_max,
        medium_hop_max=args.medium_hop_max,
    )
    try:
        run_pipeline(cfg_obj, pbar=file_bar)
        file_bar.set_postfix_str("done")
        result = f"[DONE] {stem}"
    except Exception as exc:
        file_bar.set_postfix_str(f"ERROR: {exc}")
        result = f"[ERR ] {stem}: {exc}"
    finally:
        file_bar.close()
        total_pbar.update(1)
    return result


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    tasks_to_run: List[str] = []
    if args.task_type:
        tasks_to_run = [args.task_type]
    elif args.task_group:
        if args.task_group == "av":
            tasks_to_run = sorted(list(TASK_TYPES_AV))
        elif args.task_group == "visual":
            tasks_to_run = sorted(list(TASK_TYPES_VISUAL_ONLY))
        elif args.task_group == "audio":
            tasks_to_run = sorted(list(TASK_TYPES_AUDIO_ONLY))
        elif args.task_group == "halluc":
            tasks_to_run = sorted(list(TASK_TYPES_HALLUCINATION))
        elif args.task_group == "all":
            tasks_to_run = sorted(list(TASK_PROMPTS.keys()))
    else:
        print("Please specify either --task_type or --task_group.")
        raise SystemExit(1)

    if args.aggregate_only:
        for t in tasks_to_run:
            paths = aggregate_by_task(output_root, target_task=t)
            for p in paths:
                print(f"[{t} aggregate] wrote {p}")
        raise SystemExit(0)

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("*.json"))
    if args.debug > 0:
        json_files = json_files[: args.debug]
    n_files = len(json_files)

    print(f"Running pipeline for tasks: {', '.join(tasks_to_run)}")
    print(f"Total files: {n_files}, workers={args.workers}")

    task_pbar = tqdm(total=len(tasks_to_run), desc="overall tasks", position=0, unit="task", leave=True)

    for task_name in tasks_to_run:
        task_pbar.set_postfix_str(f"current: {task_name}")
        args.task_type = task_name

        file_pbar = tqdm(total=n_files, desc=f"files ({task_name[:15]})", position=1, unit="file", leave=False)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, p, output_root, args, idx + 2, file_pbar): p
                for idx, p in enumerate(json_files)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    task_pbar.write(f"[{task_name}] [ERR ] {futures[future].stem}: {exc}")

        file_pbar.close()

        if not args.no_aggregate:
            agg_paths = aggregate_by_task(output_root, target_task=task_name)
            for p in agg_paths:
                task_pbar.write(f"[{task_name} aggregate] {p}")

        task_pbar.update(1)

    task_pbar.close()


if __name__ == "__main__":
    main()
