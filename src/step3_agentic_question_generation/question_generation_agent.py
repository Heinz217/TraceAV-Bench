import json
from typing import Any, Dict, List

from openai import OpenAI

from agentic_question_generation_prompts import (
    get_option_selection_constraint_text,
    get_option_selection_mode,
    get_step3_system_prompt,
    get_step3_user_payload_schema,
    get_task_instruction,
    verify_final_trajectory_labels,
)
from common import (
    PipelineConfig,
    call_llm_json,
    classify_hop_length,
    evidence_to_minute,
)
from trajectory_proposal_agent import select_top_trajectories


def collect_entity_pool(
    events_by_id: Dict[int, Dict[str, Any]],
    event_ids: List[int],
) -> Dict[str, Dict[str, str]]:
    pool: Dict[str, Dict[str, str]] = {}
    for eid in event_ids:
        involved = events_by_id[eid].get("involved_entities", {}) or {}
        for name, info in involved.items():
            if not isinstance(info, dict):
                continue
            if name not in pool:
                pool[name] = {
                    "description": info.get("description", ""),
                    "attribute": info.get("attribute", "visual"),
                }
    return pool


def _build_trajectory_event(ev: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    start = int(ev.get("start_minute", 0))
    raw_segs = ev.get("raw_segments", []) or []
    if task_type == "av_localization":
        annotated = [
            {"minute": start + i, "text": str(seg)}
            for i, seg in enumerate(raw_segs)
        ]
        return {
            "event_id": ev.get("event_id"),
            "start_minute": ev.get("start_minute"),
            "end_minute": ev.get("end_minute"),
            "summary": ev.get("summary", ""),
            "minute_captions": annotated,
        }
    return {
        "event_id": ev.get("event_id"),
        "start_minute": ev.get("start_minute"),
        "end_minute": ev.get("end_minute"),
        "summary": ev.get("summary", ""),
        "raw_segments": raw_segs,
    }


def _resolve_timestamp(node: Dict[str, Any], ev: Dict[str, Any]) -> int:
    start = int(ev.get("start_minute", 0))
    end = int(ev.get("end_minute", 0))
    llm_ts = node.get("timestamp_minute")
    if llm_ts is not None:
        try:
            ts = int(llm_ts)
            if start <= ts <= end:
                return ts
        except (ValueError, TypeError):
            pass
    return evidence_to_minute(ev, node.get("evidence", ""))


def generate_questions(
    client: OpenAI,
    cfg: PipelineConfig,
    source: Dict[str, Any],
    filtered: Dict[str, Any],
) -> Dict[str, Any]:
    events = source.get("event_blocks", []) or []
    events_by_id = {int(e.get("event_id")): e for e in events if str(e.get("event_id", "")).isdigit()}
    task_prompt = get_task_instruction(cfg.task_type)
    outputs = []
    system_prompt = get_step3_system_prompt(cfg.task_type)
    selected = select_top_trajectories(
        client=client,
        cfg=cfg,
        source=source,
        filtered=filtered,
        limit=cfg.max_questions_per_video,
    )
    for idx, item in enumerate(selected, start=1):
        traj = item.get("trajectory", {})
        event_ids = [int(x) for x in traj.get("event_ids", []) if str(x).isdigit() and int(x) in events_by_id]
        trajectory_events = [
            _build_trajectory_event(events_by_id[eid], cfg.task_type)
            for eid in event_ids
        ]
        entity_pool = collect_entity_pool(events_by_id, event_ids)
        question_direction = traj.get("question_direction", {})
        user_prompt = json.dumps(
            {
                "task_type": cfg.task_type,
                "task_instruction": task_prompt,
                "trajectory_id": traj.get("trajectory_id", f"traj_{idx}"),
                "question_direction": question_direction,
                "option_selection_mode": get_option_selection_mode(cfg.task_type),
                "option_selection_constraint": get_option_selection_constraint_text(cfg.task_type),
                "trajectory_events": trajectory_events,
                "entity_pool": entity_pool,
                "required_output_schema": get_step3_user_payload_schema(),
            },
            ensure_ascii=False,
        )
        llm = call_llm_json(
            client,
            cfg.model,
            system_prompt,
            user_prompt,
            fallback={
                "final_trajectory": [],
                "task_specific_key": {"key_name": "", "key_value": ""},
                "question_type": "single",
                "question": "",
                "options": {"A": "", "B": "", "C": "", "D": ""},
                "correct_options": [],
                "answer_text": "",
            },
        )
        task_specific_key = llm.get("task_specific_key", {})
        if not isinstance(task_specific_key, dict):
            task_specific_key = {"key_name": "", "key_value": ""}
        options = llm.get("options", {})
        if not isinstance(options, dict):
            options = {}
        normalized_options = {
            "A": str(options.get("A", "")),
            "B": str(options.get("B", "")),
            "C": str(options.get("C", "")),
            "D": str(options.get("D", "")),
        }
        question_type = str(llm.get("question_type", "")).strip().lower()
        if question_type not in {"single", "multiple"}:
            question_type = "single"

        correct_options_raw = llm.get("correct_options", [])
        if isinstance(correct_options_raw, str):
            correct_options_raw = [correct_options_raw]
        if not isinstance(correct_options_raw, list):
            correct_options_raw = []
        correct_options: List[str] = []
        for opt in correct_options_raw:
            c = str(opt).strip().upper()
            if c in {"A", "B", "C", "D"} and c not in correct_options:
                correct_options.append(c)
        if not correct_options:
            correct_options = ["A"] if normalized_options.get("A") else []

        mode = get_option_selection_mode(cfg.task_type)
        if mode == "single_choice_only":
            question_type = "single"
            correct_options = correct_options[:1]
        else:
            if question_type == "single":
                correct_options = correct_options[:1]
            else:
                if len(correct_options) < 2:
                    question_type = "single"
                    correct_options = correct_options[:1]

        answer_text = llm.get("answer_text", "")
        if not answer_text:
            if question_type == "single" and correct_options:
                answer_text = normalized_options.get(correct_options[0], "")
            elif question_type == "multiple":
                answer_text = " | ".join(normalized_options.get(c, "") for c in correct_options if c)

        raw_traj = llm.get("final_trajectory", [])
        if not isinstance(raw_traj, list):
            raw_traj = []
        refined_traj = []
        for node in raw_traj:
            if not isinstance(node, dict):
                continue
            eid = node.get("event_id")
            if not str(eid).isdigit():
                continue
            eid_int = int(eid)
            if eid_int not in events_by_id:
                continue
            ev = events_by_id[eid_int]
            minute = _resolve_timestamp(node, ev)
            refined_traj.append(
                {
                    "event_id": eid_int,
                    "evidence": node.get("evidence", ""),
                    "label": node.get("label", ""),
                    "reason": node.get("reason", ""),
                    "timestamp_minute": minute,
                    "event_time_range": {
                        "start_minute": ev.get("start_minute"),
                        "end_minute": ev.get("end_minute"),
                    },
                }
            )

        minute_hop_count = 0
        if len(refined_traj) >= 2:
            for i in range(len(refined_traj) - 1):
                minute_hop_count += abs(
                    int(refined_traj[i + 1]["timestamp_minute"]) - int(refined_traj[i]["timestamp_minute"])
                )

        outputs.append(
            {
                "trajectory_id": traj.get("trajectory_id", f"traj_{idx}"),
                "original_event_ids": event_ids,
                "task_specific_key": {
                    "key_name": str(task_specific_key.get("key_name", "")),
                    "key_value": str(task_specific_key.get("key_value", "")),
                },
                "question": llm.get("question", ""),
                "options": normalized_options,
                "question_type": question_type,
                "correct_options": correct_options,
                "answer_text": answer_text,
                "minute_hop_count": minute_hop_count,
                "hop_length_label": classify_hop_length(minute_hop_count, cfg),
                "trajectory_with_timestamps": refined_traj,
            }
        )
    return {
        "video_id": source.get("video_id"),
        "task_type": cfg.task_type,
        "qa_items": outputs,
    }


def verify_questions(qa_generation: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    task_type = str(qa_generation.get("task_type") or cfg.task_type)
    verified_items: List[Dict[str, Any]] = []
    stats = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "label_failed": 0,
        "choice_failed": 0,
    }
    for qa in qa_generation.get("qa_items", []) or []:
        if not isinstance(qa, dict):
            continue
        traj = qa.get("trajectory_with_timestamps", [])
        if not isinstance(traj, list):
            traj = []
        labels = [str(node.get("label", "")) if isinstance(node, dict) else "" for node in traj]
        label_passed, label_reasons = verify_final_trajectory_labels(task_type, labels)

        question_type = str(qa.get("question_type", "")).strip().lower()
        correct_options = qa.get("correct_options", [])
        if isinstance(correct_options, str):
            correct_options = [correct_options]
        if not isinstance(correct_options, list):
            correct_options = []
        normalized_correct_options: List[str] = []
        for opt in correct_options:
            c = str(opt).strip().upper()
            if c in {"A", "B", "C", "D"} and c not in normalized_correct_options:
                normalized_correct_options.append(c)

        choice_reasons: List[str] = []
        if question_type not in {"single", "multiple"}:
            choice_reasons.append("invalid_question_type")
        if question_type == "single" and len(normalized_correct_options) != 1:
            choice_reasons.append("single_requires_exactly_one_correct_option")
        if question_type == "multiple" and len(normalized_correct_options) < 2:
            choice_reasons.append("multiple_requires_at_least_two_correct_options")

        choice_passed = len(choice_reasons) == 0
        passed = label_passed and choice_passed
        reasons = sorted(set(label_reasons + choice_reasons))
        stats["total"] += 1
        if passed:
            stats["passed"] += 1
        else:
            stats["failed"] += 1
        if not label_passed:
            stats["label_failed"] += 1
        if not choice_passed:
            stats["choice_failed"] += 1
        verified_items.append(
            {
                **qa,
                "label_verification": {
                    "passed": label_passed,
                    "reasons": label_reasons,
                    "labels_raw": labels,
                },
                "choice_verification": {
                    "passed": choice_passed,
                    "reasons": choice_reasons,
                    "question_type": question_type,
                    "correct_options": normalized_correct_options,
                },
                "verification": {"passed": passed, "reasons": reasons},
            }
        )
    return {
        "video_id": qa_generation.get("video_id"),
        "task_type": task_type,
        "qa_items": verified_items,
        "label_verification_stats": stats,
    }
