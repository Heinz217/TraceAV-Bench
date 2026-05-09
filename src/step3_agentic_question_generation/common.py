import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


@dataclass
class PipelineConfig:
    input_json: Path
    output_dir: Path
    task_type: str
    model: str
    base_url: str
    api_key: str
    min_hops: int
    max_hops: int
    min_span: int
    max_candidates: int
    short_hop_max: int
    medium_hop_max: int
    max_questions_per_video: int


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_client(cfg: PipelineConfig) -> OpenAI:
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


def normalize_text(text: str) -> List[str]:
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return [t for t in clean.split() if t]


def jaccard(a: str, b: str) -> float:
    ta, tb = set(normalize_text(a)), set(normalize_text(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def event_duration(event: Dict[str, Any]) -> int:
    return int(event.get("end_minute", 0)) - int(event.get("start_minute", 0)) + 1


def build_event_catalog(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    catalog = []
    for ev in events:
        eid = ev.get("event_id")
        involved = ev.get("involved_entities", {}) or {}
        entity_names = sorted(list(involved.keys()))
        entity_attrs = {
            name: (info.get("attribute", "visual") if isinstance(info, dict) else "visual")
            for name, info in involved.items()
        }
        catalog.append(
            {
                "event_id": eid,
                "start_minute": ev.get("start_minute"),
                "end_minute": ev.get("end_minute"),
                "duration_min": event_duration(ev),
                "summary": ev.get("summary", ""),
                "entities": entity_names,
                "entity_attributes": entity_attrs,
            }
        )
    return catalog


def evidence_to_minute(event: Dict[str, Any], evidence: str) -> int:
    start = int(event.get("start_minute", 0))
    end = int(event.get("end_minute", 0))
    duration = end - start + 1
    if duration <= 1:
        return start
    raw_segments = event.get("raw_segments", []) or []
    if not raw_segments:
        return start
    best_idx = 0
    best_score = -1.0
    for idx, seg in enumerate(raw_segments):
        score = jaccard(evidence, seg)
        if score > best_score:
            best_score = score
            best_idx = idx
    mapped = start + best_idx
    if mapped > end:
        mapped = end
    return mapped


def classify_hop_length(minute_hop_count: int, cfg: PipelineConfig) -> str:
    if minute_hop_count <= cfg.short_hop_max:
        return "short"
    if minute_hop_count <= cfg.medium_hop_max:
        return "medium"
    return "long"
