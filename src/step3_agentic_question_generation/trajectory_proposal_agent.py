import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from agentic_question_generation_prompts import (
    get_option_selection_constraint_text,
    get_option_selection_mode,
    get_step1_system_prompt,
    get_step1_user_payload_schema,
    get_task_instruction,
)
from common import (
    PipelineConfig,
    build_event_catalog,
    call_llm_json,
)


def task_setup(cfg: PipelineConfig, source: Dict[str, Any]) -> Dict[str, Any]:
    task_prompt = get_task_instruction(cfg.task_type)
    return {
        "video_id": source.get("video_id"),
        "audio_id": source.get("audio_id"),
        "task_type": cfg.task_type,
        "task_prompt": task_prompt,
        "constraints": {
            "min_hops": cfg.min_hops,
            "max_hops": cfg.max_hops,
            "min_span_events": cfg.min_span,
            "max_candidates": cfg.max_candidates,
            "max_questions_per_video": cfg.max_questions_per_video,
        },
    }


def propose_trajectories(
    client: OpenAI,
    cfg: PipelineConfig,
    source: Dict[str, Any],
    task_context: Dict[str, Any],
) -> Dict[str, Any]:
    events = source.get("event_blocks", []) or []
    catalog = build_event_catalog(events)
    system_prompt = get_step1_system_prompt(cfg.task_type)
    user_prompt = json.dumps(
        {
            "task_type": task_context["task_type"],
            "task_prompt": task_context["task_prompt"],
            "constraints": task_context["constraints"],
            "option_selection_mode": get_option_selection_mode(cfg.task_type),
            "option_selection_constraint": get_option_selection_constraint_text(cfg.task_type),
            "events": catalog,
            "required_output_schema": get_step1_user_payload_schema(),
        },
        ensure_ascii=False,
    )
    llm = call_llm_json(
        client,
        cfg.model,
        system_prompt,
        user_prompt,
        fallback={"trajectory_candidates": []},
    )
    candidates = llm.get("trajectory_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    for c in candidates:
        qd = c.get("question_direction")
        if not isinstance(qd, dict):
            c["question_direction"] = {
                "focus": "",
                "answer_hint": "",
                "distractor_hint": "",
                "answer_mode_hint": "",
            }
        else:
            c["question_direction"] = {
                "focus": str(qd.get("focus", "")),
                "answer_hint": str(qd.get("answer_hint", "")),
                "distractor_hint": str(qd.get("distractor_hint", "")),
                "answer_mode_hint": str(qd.get("answer_mode_hint", "")),
            }
    return {
        "video_id": source.get("video_id"),
        "trajectory_candidates": candidates[: cfg.max_candidates],
        "event_catalog_size": len(catalog),
    }


def is_valid_multihop(
    traj: Dict[str, Any],
    events_by_id: Dict[int, Dict[str, Any]],
    cfg: PipelineConfig,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ids = traj.get("event_ids", [])
    if not isinstance(ids, list):
        return False, ["event_ids_not_list"]
    ids = [int(x) for x in ids if str(x).isdigit()]
    if len(ids) < cfg.min_hops:
        reasons.append("too_few_hops")
    if len(ids) > cfg.max_hops:
        reasons.append("too_many_hops")
    if len(set(ids)) != len(ids):
        reasons.append("duplicate_event_ids")
    if any(eid not in events_by_id for eid in ids):
        reasons.append("unknown_event_id")
        return False, reasons
    if ids:
        span = max(ids) - min(ids)
        if span < cfg.min_span:
            reasons.append("span_too_short")
    shared_entity = False
    for i in range(len(ids) - 1):
        e1 = set((events_by_id[ids[i]].get("involved_entities") or {}).keys())
        e2 = set((events_by_id[ids[i + 1]].get("involved_entities") or {}).keys())
        if e1 & e2:
            shared_entity = True
            break
    if not shared_entity:
        reasons.append("no_shared_entity_bridge")
    return len(reasons) == 0, reasons


def filter_trajectories(
    source: Dict[str, Any],
    proposal: Dict[str, Any],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    events = source.get("event_blocks", []) or []
    events_by_id = {int(e.get("event_id")): e for e in events if str(e.get("event_id", "")).isdigit()}
    kept, removed = [], []
    for traj in proposal.get("trajectory_candidates", []):
        valid, reasons = is_valid_multihop(traj, events_by_id, cfg)
        item = {"trajectory": traj, "reasons": reasons}
        if valid:
            kept.append(item)
        else:
            removed.append(item)
    return {
        "video_id": source.get("video_id"),
        "kept_trajectories": kept,
        "removed_trajectories": removed,
        "stats": {"kept": len(kept), "removed": len(removed)},
    }


def select_top_trajectories(
    client: OpenAI,
    cfg: PipelineConfig,
    source: Dict[str, Any],
    filtered: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    kept = filtered.get("kept_trajectories", []) or []
    if not isinstance(kept, list):
        return []
    if limit <= 0:
        return []
    if len(kept) <= limit:
        return [x for x in kept if isinstance(x, dict)]

    compact = []
    for idx, item in enumerate(kept, start=1):
        traj = item.get("trajectory", {}) if isinstance(item, dict) else {}
        compact.append(
            {
                "rank_index": idx,
                "trajectory_id": traj.get("trajectory_id", f"traj_{idx}"),
                "event_ids": traj.get("event_ids", []),
                "why_multihop": traj.get("why_multihop", ""),
                "range_type": traj.get("range_type", ""),
                "bridge_type": traj.get("bridge_type", ""),
                "question_direction": traj.get("question_direction", {}),
            }
        )

    selector_system = (
        "You are selecting trajectories for hard, high-quality QA generation.\n"
        "Pick trajectories that can produce the most difficult yet answerable questions.\n"
        "Prioritize: non-trivial multi-hop reasoning, grounded evidence, low guessability,\n"
        "and clear but confusing distractor opportunities.\n"
        "Return ONLY JSON: {\"selected_trajectory_ids\": [\"...\"]}"
    )
    selector_user = json.dumps(
        {
            "task_type": cfg.task_type,
            "video_id": source.get("video_id"),
            "selection_goal": "Select the highest-quality trajectories for difficult questions.",
            "max_questions_per_video": limit,
            "candidates": compact,
        },
        ensure_ascii=False,
    )
    llm = call_llm_json(
        client,
        cfg.model,
        selector_system,
        selector_user,
        fallback={"selected_trajectory_ids": []},
    )
    chosen_ids = llm.get("selected_trajectory_ids", [])
    if not isinstance(chosen_ids, list):
        chosen_ids = []
    chosen_set = {str(x) for x in chosen_ids}

    selected: List[Dict[str, Any]] = []
    for item in kept:
        if not isinstance(item, dict):
            continue
        traj = item.get("trajectory", {})
        tid = str(traj.get("trajectory_id", ""))
        if tid and tid in chosen_set:
            selected.append(item)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in kept:
            if not isinstance(item, dict) or item in selected:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected[:limit]
