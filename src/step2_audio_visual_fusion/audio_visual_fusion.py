import os
import json
import re
import base64
import time
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Required environment variable {name!r} is not set. "
            "Edit this script header or export the variable before running."
        )
    return v


# Endpoint for the LLM used to fuse visual + audio captions (OpenAI-compatible).
API_KEY       = _require_env("STEP2_API_KEY")
VLLM_API_URL  = _require_env("STEP2_API_BASE_URL")
MODEL_NAME    = os.environ.get("STEP2_MODEL_NAME", "gemini-2.5-flash")

# Paths.
AUDIO_DIR          = _require_env("STEP2_AUDIO_DIR")          # pre-extracted audio segments
VIDEO_CAPTION_DIR  = _require_env("STEP2_VIDEO_CAPTION_DIR")  # output of step 1
OUT_DIR            = _require_env("STEP2_OUT_DIR")            # output: per-minute fused captions
TEMP_SEG_DIR       = os.environ.get("STEP2_TEMP_SEG_DIR", "/tmp/traceavbench_step2_segments")

MAX_WORKERS = int(os.environ.get("STEP2_MAX_WORKERS", "36"))
MAX_RETRY   = int(os.environ.get("STEP2_MAX_RETRY", "5"))

AV_FUSION_PROMPT = """
You are an expert video analyst. Your goal is to synthesize a 1-minute audio-visual integrated caption. 
You have two sources: 
1. [Visual Context]: A summary of what is seen.
2. [Audio Track]: The sounds/speech from the same segment.

[Visual Context]
{visual_summary}

[Visual Entities Present (Name & Visual Description)]
{visual_entities}

[Analysis Requirements]
1. Information Preservation: The integrated_caption MUST encapsulate all of the information from the [Visual Context]. 
2. Combined Narrative: Integrate the provided visual description with the actual audio from this segment. 
3. Entity Alignment & Evolution: Match voices/sounds to the visual entities mentioned above if a logical connection exists. 
   - If the audio reveals NEW details about an entity (e.g., their name is spoken, they have a specific voice/accent, or they make a distinct sound), update their description.
   - You do not need to use rigid "Name/Visual/Audio" tags. Just provide a naturally updated, combined descriptive string.
   - If an entity has NO relevant audio associated with it in this minute, simply omit it from the `updated_entities` output.
4. Audio Details: Describe background music, environmental sounds, and specific dialogue/speech context.
5. Final Output: A single, high-density paragraph that captures both what is seen and what is heard, maintaining perfect chronological and identity consistency. 
6. Do not output the timestamps, just the integrated caption.

[Output Format]
Your response must be a valid JSON:
{{
  "minute": {minute},
  "integrated_caption": "A detailed paragraph merging visual events with auditory elements.",
  "updated_entities": {{
    "EntityName": "Updated distinguishing description including newly discovered audio context (if any)"
  }}
}}
"""


def robust_json_parse(raw_output):
    try:
        return json.loads(raw_output)
    except Exception:
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                fixed = re.sub(r",\s*([\]}])", r"\1", json_match.group())
                try:
                    return json.loads(fixed)
                except Exception:
                    return None
    return None


def get_visual_data(audio_id):
    base_id = audio_id.replace("_a", "")
    visual_json_path = os.path.join(VIDEO_CAPTION_DIR, f"{base_id}_captions.json")
    if os.path.exists(visual_json_path):
        with open(visual_json_path, 'r', encoding='utf-8') as f:
            return json.load(f), base_id
    return None, base_id


def encode_audio_to_uri(file_path):
    ext = os.path.splitext(file_path)[1].replace('.', '').lower()
    mime_type = "audio/mpeg" if ext == "mp3" else f"audio/{ext}"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"


def run_av_inference(client, audio_path, visual_summary, visual_entities, minute):
    prompt = AV_FUSION_PROMPT.format(
        visual_summary=visual_summary,
        visual_entities=visual_entities,
        minute=minute,
    )
    audio_uri = encode_audio_to_uri(audio_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_uri, "format": "mp3"},
                    },
                ],
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def process_single_audio(audio_id, worker_id):
    output_file = os.path.join(OUT_DIR, f"{audio_id}_audio_captions.json")
    client = OpenAI(api_key=API_KEY, base_url=VLLM_API_URL)
    seg_folder = os.path.join(TEMP_SEG_DIR, audio_id)

    if not os.path.exists(seg_folder):
        return None

    visual_json_data, base_id = get_visual_data(audio_id)
    if not visual_json_data:
        with open("pipeline_errors.txt", "a") as f:
            f.write(f"No visual JSON: {audio_id}\n")
        return None

    entity_lib = visual_json_data.get("entity_library", {})
    visual_segments = visual_json_data.get("segments", [])
    audio_segments = sorted([f for f in os.listdir(seg_folder) if f.endswith('.mp3')])
    fusion_results = []

    for i, seg_file in enumerate(audio_segments):
        minute_idx = i + 1
        seg_path = os.path.join(seg_folder, seg_file)
        v_seg = next((s for s in visual_segments if s.get("minute") == minute_idx), None)
        v_summary = v_seg.get("caption", "No visual caption.") if v_seg else "No visual context."
        v_active = v_seg.get("active_entities", {}) if v_seg else {}
        v_new = v_seg.get("discovered_new_entities", {}) if v_seg else {}
        v_entities_str = json.dumps({**v_active, **v_new}, indent=2, ensure_ascii=False)

        success = False
        for attempt in range(MAX_RETRY):
            try:
                raw_output = run_av_inference(client, seg_path, v_summary, v_entities_str, minute_idx)
                parsed = robust_json_parse(raw_output)

                if parsed:
                    updated_ents = parsed.get("updated_entities", {})
                    if isinstance(updated_ents, dict):
                        for ent_name, ent_desc in updated_ents.items():
                            if ent_name in entity_lib:
                                entity_lib[ent_name] = ent_desc

                    fusion_results.append({
                        "minute": minute_idx,
                        "integrated_caption": parsed.get("integrated_caption", ""),
                        "visual_original_caption": v_summary,
                        "active_entities": v_active,
                        "audio_updated_entities": updated_ents,
                    })
                    success = True
                    break
            except Exception:
                time.sleep(2 * (attempt + 1))

        if not success:
            fusion_results.append({"minute": minute_idx, "integrated_caption": "GENERATION_FAILED", "status": "error"})

    if fusion_results:
        final_data = {
            "audio_id": audio_id,
            "video_id": base_id,
            "segments": fusion_results,
            "entity_library": entity_lib,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

    return audio_id


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    todo_list = [d for d in os.listdir(TEMP_SEG_DIR) if os.path.isdir(os.path.join(TEMP_SEG_DIR, d))]
    todo_list = [d for d in todo_list if not os.path.exists(os.path.join(OUT_DIR, f"{d}_audio_captions.json"))]

    print(f"Workers: {MAX_WORKERS} | Todo: {len(todo_list)}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_audio, d, i % MAX_WORKERS): d for i, d in enumerate(todo_list)}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Total Progress"):
            pass


if __name__ == "__main__":
    main()
