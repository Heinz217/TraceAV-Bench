import os
import json
import re
import base64
from tqdm import tqdm
import subprocess
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Edit this script header or export the variable before running."
        )
    return v


# Paths (configure via environment).
SOURCE_DIR     = _require_env("STEP1_SOURCE_DIR")      # raw videos root
OUT_DIR        = _require_env("STEP1_OUT_DIR")         # output: per-minute captions
TEMP_SEG_DIR   = os.environ.get("STEP1_TEMP_SEG_DIR", "/tmp/traceavbench_step1_segments")

# Extensions to pick up when scanning SOURCE_DIR.
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi")

# OpenAI-compatible endpoint hosting a Qwen3-VL (or similar) server.
VLLM_API_URL   = os.environ.get("STEP1_VLLM_API_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("STEP1_VLLM_MODEL_NAME", "Qwen3-VL-32B-Instruct")
MAX_WORKERS    = int(os.environ.get("STEP1_MAX_WORKERS", "8"))

CONSISTENT_VIDEO_PROMPT = """
Task: Analyze this 1-minute video segment as a continuous narrative.

[Identity Consistency Context]
Refer to these previously identified entities to maintain naming consistency:
{entity_cache}

[Analysis Requirements]
- Narrative Flow: Describe the events as a dynamic sequence, focusing on how actions evolve from start to finish (minute_summary).
- Entity Tracking: 
    1. Identify which entities from the Context above are actively visible or interacting in this specific minute (present_entities).
    2. If any known entity has changed appearance/state/role, update its description in present_entities to reflect the latest state.
    3. Identify any NEW prominent people or objects not listed in the Context (new_entities).
- Detail Capture: Capture subtle gestures, clothing details and any on-screen text.
- Environmental Shift: Note changes in setting, lighting, or atmosphere if applicable.
- Notations: In the minute_summary, use consistent names for entities based on the Context.
- For every entity in present_entities and new_entities, always return a non-empty distinguishing description (not just names).

[Strict Constraints]
- DO NOT list any entities from the Context that are NOT visible in this segment.

[Output Format]
Your response must be a valid JSON:
{{
  "minute_summary": "A high-density paragraph describing the visual narrative.",
  "present_entities": {{"EntityName": "Brief distinguishing description (can be updated if the entity has evolved)"}},
  "new_entities": {{"NewEntityName": "Brief distinguishing description for future tracking"}}
}}
"""


def encode_video_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')


def run_vllm_inference(client, video_path, cache_str):
    prompt = CONSISTENT_VIDEO_PROMPT.format(entity_cache=cache_str)
    video_base64 = encode_video_base64(video_path)

    response = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=3500,
    )
    return response.choices[0].message.content


def normalize_entity_dict(entity_obj):
    if not isinstance(entity_obj, dict):
        return {}
    cleaned = {}
    for raw_name, raw_desc in entity_obj.items():
        name = str(raw_name).strip()
        desc = str(raw_desc).strip()
        if name and desc:
            cleaned[name] = desc
    return cleaned


def process_single_video(video_name, worker_id):
    video_id = os.path.splitext(video_name)[0]
    video_path = os.path.join(SOURCE_DIR, video_name)
    output_file = os.path.join(OUT_DIR, f"{video_id}_captions.json")

    client = OpenAI(api_key="EMPTY", base_url=VLLM_API_URL)

    seg_folder = os.path.join(TEMP_SEG_DIR, video_id)
    os.makedirs(seg_folder, exist_ok=True)
    seg_pattern = os.path.join(seg_folder, "seg_%03d.mp4")
    subprocess.run(
        f"ffmpeg -i {video_path} -an -f segment -segment_time 60 -reset_timestamps 1 -c copy -y {seg_pattern} -loglevel quiet",
        shell=True,
    )

    segments = sorted([os.path.join(seg_folder, f) for f in os.listdir(seg_folder) if f.endswith('.mp4')])

    pbar_inner = tqdm(total=len(segments), desc=f" -> {video_id[:12]}", position=worker_id + 1, leave=False, unit="min")

    full_video_results = []
    global_entity_cache = {}

    for i, seg_path in enumerate(segments):
        cache_text = json.dumps(global_entity_cache, indent=2, ensure_ascii=False) if global_entity_cache else "None"
        try:
            raw_output = run_vllm_inference(client, seg_path, cache_text)
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                existing_entity_names = set(global_entity_cache.keys())
                model_present = normalize_entity_dict(parsed.get("present_entities", {}))
                model_new = normalize_entity_dict(parsed.get("new_entities", {}))

                merged_entities = dict(model_present)
                for name, desc in model_new.items():
                    if name not in merged_entities:
                        merged_entities[name] = desc

                corrected_present = {}
                corrected_new = {}
                for name, desc in merged_entities.items():
                    if name in existing_entity_names:
                        corrected_present[name] = desc
                    else:
                        corrected_new[name] = desc

                if corrected_present:
                    global_entity_cache.update(corrected_present)
                if corrected_new:
                    global_entity_cache.update(corrected_new)

                full_video_results.append({
                    "minute": i + 1,
                    "caption": parsed.get("minute_summary", ""),
                    "active_entities": corrected_present,
                    "discovered_new_entities": corrected_new,
                })
        except Exception:
            continue
        finally:
            pbar_inner.update(1)

    pbar_inner.close()

    if full_video_results:
        final_data = {
            "video_id": video_id,
            "segments": full_video_results,
            "entity_library": global_entity_cache,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

    subprocess.run(f"rm -rf {seg_folder}", shell=True)
    return video_id


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEMP_SEG_DIR, exist_ok=True)

    all_videos = sorted(
        f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(VIDEO_EXTS)
    )
    todo_list = [
        f for f in all_videos
        if not os.path.exists(
            os.path.join(OUT_DIR, f"{os.path.splitext(f)[0]}_captions.json")
        )
    ]

    print(f"Starting Multi-Process Pipeline...")
    print(f"Total: {len(todo_list)} videos | Workers: {MAX_WORKERS}\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_video, video_name, i % MAX_WORKERS): video_name
            for i, video_name in enumerate(todo_list)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="OVERALL PROGRESS", position=0):
            try:
                future.result()
            except Exception:
                pass

    print("\n" * (MAX_WORKERS + 1))
    print("All tasks completed.")


if __name__ == "__main__":
    main()
