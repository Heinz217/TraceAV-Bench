import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Edit this script header or export the variable before running."
        )
    return v


API_KEY    = _require_env("STEP3_API_KEY")
API_URL    = _require_env("STEP3_API_BASE_URL")
MODEL_NAME = os.environ.get("STEP3_MODEL_NAME", "gpt-5.1-2025-11-13")

INPUT_DIR  = _require_env("STEP3_INPUT_DIR")    # fused captions from step 2
OUTPUT_DIR = _require_env("STEP3_OUTPUT_DIR")   # event blocks

MAX_WORKERS = int(os.environ.get("STEP3_MAX_WORKERS", "4"))

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
    timeout=60.0,
    max_retries=3,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print_lock = threading.Lock()
available_positions = list(range(1, MAX_WORKERS + 1))

EVENT_SEGMENTATION_PROMPT = """
Task: Analyze the narrative continuity between a past event summary and the current video segment.

[Identity Consistency Context]
Active entities in the current minute:
{current_entities}

[Past State Memory (Ongoing Event)]
{past_summary}

[Future Vision (Look-ahead Context)]
T+1: {future_1}
T+2: {future_2}

[Analysis Requirements]
Determine the relationship of the Current Minute (Minute T) to the [Past State Memory]:
- CONTINUE: Direct continuation of narrative and characters.
- OVERLAP_TRANSITION: Minute T contains the resolution of the past and start of the future.
- HARD_CUT: Total break from the past, aligns with the new context in Future Vision.

[Output Format]
Your response must be a valid JSON:
{{
  "action": "CONTINUE" | "OVERLAP_TRANSITION" | "HARD_CUT",
  "reason": "Brief logical basis.",
  "updated_summary": "A high-density narrative paragraph."
}}
"""


def get_llm_decision(past_summary, current_seg, future_1, future_2):
    current_entities_list = ", ".join(current_seg.get("processed_entities", {}).keys())

    user_prompt = EVENT_SEGMENTATION_PROMPT.format(
        current_entities=current_entities_list if current_entities_list else "None",
        past_summary=past_summary,
        future_1=future_1,
        future_2=future_2,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a specialized video analyst. Output ONLY valid JSON."},
                    {"role": "user", "content": f"[Current Minute T Caption]: {current_seg['integrated_caption']}\n" + user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 ** attempt)
                continue
            with print_lock:
                print(f"API Error on attempt {attempt+1}: {e}")
            if attempt == 2:
                break

    return {
        "action": "CONTINUE",
        "reason": "Fallback due to processing error",
        "updated_summary": past_summary + " " + current_seg["integrated_caption"],
    }


def process_single_video(file_path):
    with print_lock:
        worker_id = available_positions.pop(0)

    filename = os.path.basename(file_path)
    video_id = filename.split('.')[0]
    out_path = os.path.join(OUTPUT_DIR, filename)
    pbar_inner = None

    try:
        if os.path.exists(out_path):
            return True

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            return False

        pbar_inner = tqdm(
            total=len(segments),
            desc=f" -> {video_id[:12]}",
            position=worker_id,
            leave=False,
            unit="min",
        )

        events = []
        first_seg = segments[0]
        current_event = {
            "start_minute": first_seg["minute"],
            "end_minute": None,
            "captions": [first_seg["integrated_caption"]],
            "summary": first_seg["integrated_caption"],
            "event_entity_library": dict(first_seg.get("processed_entities", {})),
        }
        pbar_inner.update(1)

        for i in range(1, len(segments)):
            seg_t = segments[i]
            f1 = segments[i + 1]["integrated_caption"] if i + 1 < len(segments) else "END_OF_VIDEO"
            f2 = segments[i + 2]["integrated_caption"] if i + 2 < len(segments) else "END_OF_VIDEO"

            decision = get_llm_decision(current_event["summary"], seg_t, f1, f2)
            action = decision.get("action", "CONTINUE")
            new_summary = decision.get("updated_summary", seg_t["integrated_caption"])

            if action == "CONTINUE":
                current_event["captions"].append(seg_t["integrated_caption"])
                current_event["summary"] = new_summary
                current_event["event_entity_library"].update(seg_t.get("processed_entities", {}))

            elif action == "OVERLAP_TRANSITION":
                current_event["end_minute"] = seg_t["minute"]
                current_event["captions"].append(seg_t["integrated_caption"])
                current_event["event_entity_library"].update(seg_t.get("processed_entities", {}))
                events.append(current_event)
                current_event = {
                    "start_minute": seg_t["minute"],
                    "end_minute": None,
                    "captions": [seg_t["integrated_caption"]],
                    "summary": new_summary,
                    "event_entity_library": dict(seg_t.get("processed_entities", {})),
                }

            elif action == "HARD_CUT":
                current_event["end_minute"] = segments[i - 1]["minute"]
                events.append(current_event)
                current_event = {
                    "start_minute": seg_t["minute"],
                    "end_minute": None,
                    "captions": [seg_t["integrated_caption"]],
                    "summary": new_summary,
                    "event_entity_library": dict(seg_t.get("processed_entities", {})),
                }
            pbar_inner.update(1)

        current_event["end_minute"] = segments[-1]["minute"]
        events.append(current_event)

        output_data = {
            "video_id": data.get("video_id", video_id),
            "entity_library_global": data.get("entity_library", {}),
            "event_blocks": events,
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        return True

    except Exception as e:
        with tqdm.external_write_mode():
            print(f"Error processing {video_id}: {e}")
        return False
    finally:
        if pbar_inner:
            pbar_inner.close()
        with print_lock:
            available_positions.append(worker_id)
            available_positions.sort()


def main():
    json_files = sorted([
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.json')
    ])

    print(f"Step 3 / Stage 1 Event Segmentation Agent | model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_video, f): f for f in json_files}
        for future in tqdm(as_completed(futures), total=len(json_files), desc="OVERALL PROGRESS", position=0):
            try:
                future.result()
            except Exception as e:
                with tqdm.external_write_mode():
                    print(f"Worker Error: {e}")

    print(f"Finished. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
