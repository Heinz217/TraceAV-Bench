from typing import Dict, FrozenSet, List, Tuple

TASK_TYPES_AV: FrozenSet[str] = frozenset(
    {
        "av_retrieval",
        "av_sequencing",
        "av_tracking",
        "av_causal_fwd",
        "av_causal_bwd",
        "av_matching",
        "av_localization",
    }
)
TASK_TYPES_VISUAL_ONLY: FrozenSet[str] = frozenset({"v_spatial", "v_counting"})
TASK_TYPES_AUDIO_ONLY: FrozenSet[str] = frozenset({"a_speech", "a_sound", "a_music"})
TASK_TYPES_HALLUCINATION: FrozenSet[str] = frozenset(
    {"halluc_v2a", "halluc_a2v", "halluc_splicing"}
)
TASK_TYPES_SINGLE_CHOICE_ONLY: FrozenSet[str] = frozenset(
    {
        "v_counting",
        "v_spatial",
        "av_causal_fwd",
        "av_causal_bwd",
        "av_retrieval",
        "a_speech",
        "a_sound",
        "a_music",
        "av_matching",
    }
)

def normalize_trajectory_label(label: str) -> str:
    """Map model output to canonical labels: audio | visual | audio-visual."""
    s = str(label or "").strip().lower().replace("_", "-")
    s = s.replace(" ", "")
    if s in ("audiovisual", "audio-video"):
        s = "audio-visual"
    if s == "video":
        s = "visual"
    return s

def verify_final_trajectory_labels(task_type: str, labels: List[str]) -> Tuple[bool, List[str]]:
    """Validate modality labels on final_trajectory hops for a given task type.

    - AV tasks: at least one hop must contribute audio (label is *audio* or *audio-visual*)
      and at least one hop must contribute visual (*visual* or *audio-visual*). A single
      *audio-visual* hop counts toward both.
    - Visual-only tasks: every hop must be *visual* only (no *audio*, no *audio-visual*).
    - Audio-only tasks: every hop must be *audio* only.
    - Hallucination tasks: no strict check (always passes).
    - Unknown task_type: no strict check.
    """
    reasons: List[str] = []
    normalized = [normalize_trajectory_label(x) for x in labels]
    known_tasks = (
        TASK_TYPES_AV
        | TASK_TYPES_VISUAL_ONLY
        | TASK_TYPES_AUDIO_ONLY
        | TASK_TYPES_HALLUCINATION
    )
    if task_type in TASK_TYPES_HALLUCINATION or task_type not in known_tasks:
        return True, []

    canonical = frozenset({"audio", "visual", "audio-visual"})
    for lab in normalized:
        if not lab:
            reasons.append("empty_label")
        elif lab not in canonical:
            reasons.append(f"unknown_label:{lab}")

    if not normalized:
        reasons.append("no_hops")

    if task_type in TASK_TYPES_VISUAL_ONLY:
        if any(lab and lab != "visual" for lab in normalized):
            reasons.append("visual_only_requires_all_visual")

    if task_type in TASK_TYPES_AUDIO_ONLY:
        if any(lab and lab != "audio" for lab in normalized):
            reasons.append("audio_only_requires_all_audio")

    if task_type in TASK_TYPES_AV:
        valid = [l for l in normalized if l in canonical]
        has_audio = any(l in ("audio", "audio-visual") for l in valid)
        has_visual = any(l in ("visual", "audio-visual") for l in valid)
        if not has_audio:
            reasons.append("av_requires_audio_coverage")
        if not has_visual:
            reasons.append("av_requires_visual_coverage")

    return len(reasons) == 0, sorted(set(reasons))

def get_option_selection_mode(task_type: str) -> str:
    """Return option mode constraint for the given task type.

    - single_choice_only: exactly one correct option.
    - single_or_multiple: model may choose one or more correct options.
    """
    if task_type in TASK_TYPES_SINGLE_CHOICE_ONLY:
        return "single_choice_only"
    return "single_or_multiple"

def get_option_selection_constraint_text(task_type: str, force_multiple: bool = False) -> str:
    if force_multiple:
        return (
            "This is a FORCED MULTI-CHOICE generation pass. You MUST generate a multiple-choice question "
            "with 2-4 correct options. Set question_type='multiple' and provide 2-4 items in correct_options. "
            "Ensure the question requires identifying or selecting multiple valid answers based on the trajectory evidence."
        )
    mode = get_option_selection_mode(task_type)
    if mode == "single_choice_only":
        return (
            "This task requires SINGLE-CHOICE only: exactly one correct option among A/B/C/D. "
            "Set question_type='single' and provide exactly one item in correct_options."
        )
    return (
        "This task may be SINGLE-CHOICE or MULTI-CHOICE. "
        "If one answer is correct, set question_type='single' and one item in correct_options; "
        "if multiple answers are correct, set question_type='multiple' with 2-4 items in correct_options."
    )

TASK_PROMPTS: Dict[str, str] = {
    "av_retrieval": (
        "Key Information Retrieval: construct multi-hop questions that require "
        "locating a specific fact (text, number, speech, sound, or person "
        "identity) whose answer is embedded at a non-obvious position in the video "
        "(beginning, middle, or end). The question must chain at least two events "
        "so that neither event alone is sufficient."
    ),
    "av_sequencing": (
        "Temporal Sequencing: construct multi-hop questions that require the model "
        "to order a set of events or entity-state changes along the video timeline. "
        "Questions may ask for ordering, 'what happened before/after X', or tracking "
        "how an entity's state evolves across events."
    ),
    "av_tracking": (
        "Entity Tracking: construct multi-hop questions that are centric on a specific "
        "entity (person, object, or sound source) and require tracking its identity, "
        "role, relationship, or state across multiple events. Leverage the provided "
        "entity library to ground the question."
    ),
    "av_causal_fwd": (
        "Forward Causal Reasoning: construct multi-hop questions that start from an "
        "earlier cause event and ask what effect or outcome it leads to in a later "
        "event. The causal chain must span at least two events."
    ),
    "av_causal_bwd": (
        "Backward Causal Reasoning: construct multi-hop questions that start from an "
        "observed effect or outcome and ask what earlier cause or prerequisite event "
        "led to it. The causal chain must span at least two events."
    ),
    "av_matching": (
        "Audio-Visual Matching: construct multi-hop questions that require the model "
        "to correctly match an audio cue (speech, sound, music) to its corresponding "
        "visual context, or vice versa, across multiple events. Include both genuine "
        "matches and plausible but incorrect distractors."
    ),
    "av_localization": (
        "Spatio-Temporal Localization: construct multi-hop questions that require the "
        "model to identify the specific minute at which a target event or entity state "
        "occurs. Each event's content is provided as minute-level captions "
        "(minute_captions: [{minute: N, text: '...'}]). The correct answer and all "
        "options MUST be concrete minute integers derived from the trajectory. "
        "The localization must be derived by chaining evidence from at least two events."
    ),
    "v_spatial": (
        "Spatial Reasoning: construct multi-hop questions that require understanding "
        "spatial relationships (position, direction, relative layout) of objects or "
        "persons across multiple events. Visual evidence only."
    ),
    "v_counting": (
        "Conditional Counting: construct multi-hop questions that require counting "
        "actions, state changes, or objects under a specific condition derived from "
        "multiple events (e.g., 'how many times does X happen after Y occurs'). "
        "Visual evidence only."
    ),
    "a_speech": (
        "Speech Content: construct multi-hop questions that require understanding "
        "spoken dialogue or narration content across multiple events. The answer "
        "must be derived solely from what is said (not what is seen)."
    ),
    "a_sound": (
        "Environmental Sound: construct multi-hop questions that require identifying "
        "or reasoning about non-human environmental sounds (e.g., rain, machinery, "
        "crowd noise) across multiple events. Audio evidence only."
    ),
    "a_music": (
        "Background Music: construct multi-hop questions that require reasoning about "
        "background music, ambient sound, or non-foreground audio cues across "
        "multiple events. Audio evidence only."
    ),
    "halluc_v2a": (
        "Visual-to-Audio Hallucination: construct multi-hop questions that present "
        "real visual evidence from the video and ask about an audio detail that does "
        "NOT actually exist in the video. The correct answer must confirm the audio "
        "is absent; distractors should be plausible fabricated audio details."
    ),
    "halluc_a2v": (
        "Audio-to-Visual Hallucination: construct multi-hop questions that present "
        "real audio evidence from the video and ask about a visual detail that does "
        "NOT actually exist in the video. The correct answer must confirm the visual "
        "is absent; distractors should be plausible fabricated visual details."
    ),
    "halluc_splicing": (
        "Hallucinated Splice: construct multi-hop questions that present a fabricated "
        "narrative splicing real fragments from different time points as if they form "
        "a coherent sequence. The correct answer must identify the splice as "
        "impossible or incorrect; distractors should accept the false narrative."
    ),
}

_STEP1_QUALITY_RULE = """\
[Trajectory Quality]
Select only HIGH-QUALITY trajectories. A trajectory is high quality when ALL of the
following hold:
1. STRICT Multi-hop: The intended question MUST be impossible to answer using any
   single event/node in the chain. Each hop must be a necessary piece of the puzzle.
2. No Common Sense / No Well-Known Facts: The answer MUST NOT be inferable through
   general world knowledge, common sense, or widely known facts. The answer must be
   uniquely tied to specific, non-obvious details of THIS video that require watching.
   REJECT trajectories whose answer could be guessed by someone who has never seen the video.
3. Challenging by design: The trajectory must support a question that would genuinely
   challenge an attentive viewer — not just someone who skimmed the video.
4. Strong evidence: every event in the chain contains concrete, specific evidence
   (named entity, exact number, quoted speech, visible action) — avoid vague summaries.
5. Clear entity bridge: adjacent events share at least one named entity whose state,
   role, or identity changes meaningfully between them.
6. Reject weak chains: do NOT include trajectories where the connection between events
   is only thematic, or where the answer is guessable without watching the video."""

_STEP1_QUESTION_DIRECTION_RULE = """\
[Question Direction]
For each trajectory candidate, propose a concrete and UNIQUE question direction.
Across ALL candidates in this response, no two question_direction.focus values may
ask about the same fact, entity, or event — ensure full diversity.
You will receive `option_selection_mode` in the user payload. Ensure each candidate's
question direction is compatible with that mode.
- "focus"          : one sentence describing what the MCQ should ask (e.g. "Ask which person
                     was introduced by name in event 3 after the context in event 1").
                     Must be DISTINCT from every other candidate's focus in this response.
- "answer_hint"    : briefly describe where/how the correct answer is derivable from the
                     trajectory (e.g. "The name is spoken in event 3; event 1 establishes
                     the context that makes event 3 non-obvious").
- "distractor_hint": describe what plausible wrong options should look like. Distractors MUST
                     be grounded in video facts, logically consistent, similar in length to the
                     correct option, and superficially convincing (e.g. "Use other names
                     mentioned in the trajectory that could be confused with the correct one").
- "answer_mode_hint": "single" or "multiple" (must respect option_selection_mode)."""

_STEP1_OUTPUT_SCHEMA = """\
Return ONLY valid JSON:
{
  "trajectory_candidates": [
    {
      "trajectory_id": "string",
      "event_ids": [1, 2, 3],
      "why_multihop": "string",
      "range_type": "short|long",
      "bridge_type": "entity|semantic|temporal|hybrid",
      "question_direction": {
        "focus": "what the question should ask about (1 sentence)",
        "answer_hint": "where/how the answer can be derived from the trajectory",
        "distractor_hint": "what plausible wrong options might look like",
        "answer_mode_hint": "single|multiple"
      }
    }
  ]
}"""

STEP1_SYSTEM_PROMPTS: Dict[str, str] = {
    "av_retrieval": f"""
Task: Select candidate multi-hop trajectories for Key Information Retrieval QA.

[Core Principle]
- Identify events where a specific key fact (on-screen text, number, spoken name,
  person identity) is embedded and can only be fully retrieved by chaining evidence
  from multiple events.
- The target answer may sit at the beginning, middle, or end of the trajectory —
  vary the position across candidates.
- Include BOTH short-range chains (nearby events) and long-range isolated chains.
- Prefer events that contain audio-visual entities with attribute "audio-visual" so
  that both modalities contribute to retrieval.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "av_sequencing": f"""
Task: Select candidate multi-hop trajectories for Temporal Sequencing QA.

[Core Principle]
- Identify events that form a temporal order or entity-state evolution that
  non-trivial sequence that cannot be inferred from any single event.
- Prefer events where an entity's appearance, role, or state visibly or audibly
  changes between events.
- Include BOTH short-range chains and long-range isolated chains.
- Avoid trajectories where the ordering is trivially obvious from event IDs alone.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "av_tracking": f"""
Task: Select candidate multi-hop trajectories for Entity Tracking QA.

[Core Principle]
- Centre each trajectory on a specific entity (person, object, or sound source)
  that appears in at least two non-adjacent events with meaningful state, role,
  or relationship changes.
- Leverage entities with attribute "audio-visual" when both modalities are needed
  to track the entity.
- Include BOTH nearby transitions and far-apart isolated transitions.
- Ensure the question cannot be answered from a single event mention.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "av_causal_fwd": f"""
Task: Select candidate multi-hop trajectories for Forward Causal Reasoning QA.

[Core Principle]
- Identify event pairs or chains where an earlier event is a clear prerequisite
  or cause for something that happens in a later event.
- The causal link must require evidence from both events; neither alone is
  sufficient to answer "what happens as a result of X".
- Include BOTH local causal chains and distant isolated causal chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "av_causal_bwd": f"""
Task: Select candidate multi-hop trajectories for Backward Causal Reasoning QA.

[Core Principle]
- Identify event pairs or chains where a later event shows an effect or outcome
  that can only be explained by tracing back to an earlier cause event.
- The backward causal link must require evidence from both events; neither alone
  is sufficient to answer "what caused X".
- Include BOTH local causal chains and distant isolated causal chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "av_matching": f"""
Task: Select candidate multi-hop trajectories for Audio-Visual Matching QA.

[Core Principle]
- Identify events where a specific audio cue (speech phrase, sound, music) and
  a specific visual element co-occur or are explicitly linked across events.
- Prefer events containing entities with attribute "audio-visual".
- Also identify events where an audio or visual element appears WITHOUT its
  expected counterpart — these support plausible-but-wrong distractor options.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "av_localization": f"""
Task: Select candidate multi-hop trajectories for Spatio-Temporal Localization QA.

[Core Principle]
- Identify trajectories where pinpointing the EXACT MINUTE of a target event or
  entity state requires chaining clues from at least two events.
- Each event will be provided with minute-level captions keyed by absolute minute
  (minute_captions: [{{minute: N, text: '...'}}]). Use these to reason about
  which specific minute the answer falls on.
- Prefer multi-minute events (end_minute > start_minute) so that fine-grained
  minute-level disambiguation is possible.
- The question's correct answer and all distractors MUST be specific minute integers
  drawn from the trajectory's time range — NOT descriptive phrases.
- Include BOTH short-range chains (nearby events) and long-range isolated chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "v_spatial": f"""
Task: Select candidate multi-hop trajectories for Spatial Reasoning QA.

[Core Principle]
- Identify events where spatial relationships (position, direction, relative
  layout) of objects or persons can only be determined by combining visual
  evidence from multiple events.
- Prefer events with rich visual entity descriptions.
- Audio evidence should NOT be required to answer the question.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "v_counting": f"""
Task: Select candidate multi-hop trajectories for Conditional Counting QA.

[Core Principle]
- Identify trajectories where counting actions, state changes, or objects
  requires a condition established in one event and the counting target
  distributed across other events.
- Example pattern: "How many times does X do Y after Z happens?"
- Visual evidence only; audio should not be required.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "a_speech": f"""
Task: Select candidate multi-hop trajectories for Speech Content QA.

[Core Principle]
- Identify events where spoken dialogue or narration across multiple events
  must be combined to answer a question about what was said.
- Prefer events containing entities with attribute "audio-visual" where the
  audio component (speech) is the primary evidence.
- Visual information should NOT be required to answer the question.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "a_sound": f"""
Task: Select candidate multi-hop trajectories for Environmental Sound QA.

[Core Principle]
- Identify events where non-human environmental sounds (rain, machinery, crowd,
  nature sounds) across multiple events must be combined to answer a question.
- The question must be answerable from audio alone without visual evidence.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "a_music": f"""
Task: Select candidate multi-hop trajectories for Background Audio QA.

[Core Principle]
- Identify events where background music, ambient sound, or non-foreground
  audio cues across multiple events must be combined to answer a question.
- The question must be answerable from audio alone without visual evidence.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "halluc_v2a": f"""
Task: Select candidate multi-hop trajectories for Visual-to-Audio Hallucination QA.

[Core Principle]
- Identify events with rich visual content (entities with attribute "visual" or
  "audio-visual") where the visual scene strongly implies an audio detail that
  is NOT actually present in the video.
- The trajectory should provide enough visual evidence to make the hallucinated
  audio seem plausible, while the correct answer is that the audio does not exist.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "halluc_a2v": f"""
Task: Select candidate multi-hop trajectories for Audio-to-Visual Hallucination QA.

[Core Principle]
- Identify events with rich audio content (entities with attribute "audio-visual")
  where the audio strongly implies a visual detail that is NOT actually present
  in the video.
- The trajectory should provide enough audio evidence to make the hallucinated
  visual seem plausible, while the correct answer is that the visual does not exist.
- Include BOTH short-range and long-range chains.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
    "halluc_splicing": f"""
Task: Select candidate multi-hop trajectories for Hallucinated Splice QA.

[Core Principle]
- Identify 2–4 events from DIFFERENT, non-adjacent time points whose real
  fragments could be falsely presented as a single coherent sequence.
- The trajectory should make the splice seem plausible (shared entities or
  similar settings) while the actual timeline makes the splice impossible.
- Prefer events with large temporal gaps between them.

{_STEP1_QUALITY_RULE}

{_STEP1_QUESTION_DIRECTION_RULE}

{_STEP1_OUTPUT_SCHEMA}
""",
}

_STEP3_TIMESTAMP_RULE = """\
[Timestamp Assignment]
For each node in final_trajectory, assign "timestamp_minute" by selecting the single
minute within the event's [start_minute, end_minute] range that best matches the
evidence text. Use the raw_segments list (index 0 = start_minute, index 1 = start_minute+1,
…) to pick the segment whose content is most relevant to the evidence. If the event
spans only one minute, use start_minute directly."""

_STEP3_OUTPUT_SCHEMA = """\
Return ONLY valid JSON:
{
  "final_trajectory": [
    {
      "event_id": 1,
      "evidence": "exact textual evidence from summary or raw_segments",
      "label": "audio|visual|audio-visual",
      "reason": "why this modality label",
      "timestamp_minute": 5
    }
  ],
  "task_specific_key": {
    "key_name": "task-related field name",
    "key_value": "task-related value"
  },
  "question_type": "single|multiple",
  "question": "MCQ question stem",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_options": ["A"],
  "answer_text": "exact text of the correct option"
}"""

_STEP3_LABEL_RULE = """\
[Modality Labeling Rule]
For each evidence step, assign label based on what is ACTUALLY USED to solve that hop:
- "audio"      : only auditory cues (speech, sound, music) are needed
- "video"      : only visual cues (appearance, action, text-on-screen) are needed
- "audio-video": both modalities are jointly required"""

_STEP3_QUALITY_RULE = """\
[Question Quality Priority]
When generating the question, enforce ALL of the following:
- CHALLENGING: The question must be genuinely difficult for an attentive viewer.
  Avoid questions whose answers are obvious after a single viewing or from context alone.
- NO Common Sense / NO Well-Known Facts: Reject any question whose answer can be derived
  from general world knowledge, common sense, or widely known facts without watching the video.
  If a person who has NEVER seen the video could guess the correct answer, the question is invalid.
- STRICT Multi-hop: Removing even ONE evidence step from the trajectory must make the
  question unanswerable. Every hop must be load-bearing.
- Strong grounding: every option must be grounded in trajectory evidence — no invented facts.
- Balanced options: all four options must be plausible on the surface; none should be
  obviously eliminable without careful reasoning."""

_STEP3_DISTRACTOR_RULE = """\
[Distractor Design Rules]
Distractors (wrong options) MUST satisfy ALL of the following:
1. Factually grounded: every distractor must be based on facts, entities, events, numbers,
   or phrases that ACTUALLY APPEAR in the video. Never invent details not present in the trajectory.
2. Logically and commonsensically valid: distractors must not violate basic logic or common sense.
   A wrong option should be wrong because of video-specific evidence, NOT because it is absurd.
3. Superficially plausible: each distractor must look like a reasonable answer to someone who
   has partially watched or misremembered the video — it should require careful reasoning to eliminate.
4. Length parity: all four options (correct + distractors) must be similar in length and level
   of detail. A correct option that is noticeably longer, shorter, or more specific than the
   distractors is NOT acceptable — it gives away the answer.
5. No giveaway language: do NOT use phrases like "none of the above", "all of the above",
   "it is impossible to tell", or any meta-language that breaks the MCQ format.
6. No surface-level tells: options must not differ in tone, grammatical structure, or hedging
   language in ways that make the correct one stand out stylistically."""

_STEP3_OPTION_SELECTION_RULE = """\
[Option Selection Constraint]
You will receive an `option_selection_constraint` field in user input.
- If `single_choice_only`: generate a single-choice question only.
  Set `question_type` to "single" and provide exactly one item in `correct_options`.
- If `single_or_multiple`: you may generate either single-choice or multiple-choice.
  - Single-choice: `question_type="single"` and one item in `correct_options`.
  - Multiple-choice: `question_type="multiple"` and 2-4 items in `correct_options`.
Always keep exactly four options (A/B/C/D)."""

STEP3_SYSTEM_PROMPTS: Dict[str, str] = {
    "av_retrieval": f"""
Task: Generate one multiple-choice Key Information Retrieval (KIR) question.

[Question Design]
- Identify a specific key fact (on-screen text, number, spoken name, person
  identity) that is the retrieval target.
- The question must require chaining at least two evidence steps; the target
  fact must NOT be directly answerable from a single event.
- Vary where the answer sits in the chain (beginning / middle / end).
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors must be plausible in-domain values (e.g., similar numbers, similar
  names) that appear elsewhere in the trajectory or could be confused.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "retrieval_target_type", "key_value": "text|number|speech|person"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "av_sequencing": f"""
Task: Generate one multiple-choice Temporal Sequencing (Temp-Seq) question.

[Question Design]
- Ask the model to order events, identify what happened before/after a given
  event, or describe how an entity's state changed across the trajectory.
- The correct ordering must require evidence from all events in the trajectory.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be plausible alternative orderings or state sequences.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "sequence_length", "key_value": "<number of ordered items>"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "av_tracking": f"""
Task: Generate one multiple-choice Entity Tracking question.

[Question Design]
- Centre the question on a specific entity from the entity_pool.
- Ask about the entity's identity, role, relationship, or state at a specific
  point in the trajectory, requiring cross-event linking.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be other entities or plausible but wrong state descriptions
  drawn from the entity_pool or trajectory context.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "tracked_entity", "key_value": "<entity name>"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "av_causal_fwd": f"""
Task: Generate one multiple-choice Forward Causal Reasoning question.

[Question Design]
- Frame the question as: "Given that [cause from early event], what happens /
  what is the outcome in [later event]?"
- The causal link must be non-trivial and require evidence from both events.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be plausible but causally unrelated or reversed outcomes.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "causal_direction", "key_value": "forward"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "av_causal_bwd": f"""
Task: Generate one multiple-choice Backward Causal Reasoning question.

[Question Design]
- Frame the question as: "Given that [effect/outcome in later event], what was
  the cause / prerequisite in [earlier event]?"
- The backward causal link must be non-trivial and require evidence from both events.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be plausible but causally unrelated or forward-direction answers.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "causal_direction", "key_value": "backward"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "av_matching": f"""
Task: Generate one multiple-choice Audio-Visual Matching question.

[Question Design]
- Ask the model to correctly match an audio cue to its visual context (or vice
  versa) by chaining evidence across events.
- One option is the correct match; the other three are plausible but incorrect
  matches (e.g., audio from a different event, visually similar but wrong scene).
- Produce exactly 4 options (A/B/C/D), one correct.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: all evidence steps for this task type should be labeled "audio-video".

[task_specific_key]
Use: {{"key_name": "match_direction", "key_value": "audio->visual|visual->audio"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "av_localization": f"""
Task: Generate one multiple-choice Spatio-Temporal Localization question.

[Input Format Note]
Each trajectory event is provided with a "minute_captions" field:
  minute_captions: [{{"minute": N, "text": "..."}}]
where N is the absolute minute in the video. Use these minute-annotated captions
to determine exactly which minute the target event or entity state occurs.

[Question Design]
- Ask the model to identify the EXACT MINUTE at which a specific event or entity
  state occurs, requiring evidence from at least two events to narrow it down.
- The question stem must describe the target event/state clearly enough that only
  one minute is correct when the trajectory is followed.
- ALL FOUR options (A/B/C/D) MUST be concrete integer minute values (e.g. 12, 19,
  34, 47) drawn from the trajectory's time range — NOT descriptive phrases.
- The correct option is the minute confirmed by the trajectory evidence chain.
- Distractors should be other plausible minutes from the same trajectory that
  could be confused with the correct one (nearby minutes, minutes of related events).

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "target_minute", "key_value": "<correct minute as integer>"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "v_spatial": f"""
Task: Generate one multiple-choice Spatial Reasoning question.

[Question Design]
- Ask about the spatial relationship (position, direction, relative layout) of
  objects or persons that can only be determined by combining visual evidence
  from multiple events.
- Audio evidence must NOT be needed to answer.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be plausible but spatially incorrect alternatives.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: all evidence steps for this task type should be labeled "video".

[task_specific_key]
Use: {{"key_name": "spatial_relation_type", "key_value": "position|direction|layout|relative"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "v_counting": f"""
Task: Generate one multiple-choice Conditional Counting question.

[Question Design]
- Ask how many times an action occurs, how many objects exist, or how many state
  changes happen, under a condition established in one event and counted across
  other events.
- Example: "After [condition from event A], how many times does [X] occur?"
- Audio evidence must NOT be needed to answer.
- Produce exactly 4 options (A/B/C/D) with distinct integer counts, one correct.
- Distractors should be nearby integers (±1, ±2) or counts from wrong conditions.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: all evidence steps for this task type should be labeled "video".

[task_specific_key]
Use: {{"key_name": "count_target", "key_value": "<what is being counted>"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "a_speech": f"""
Task: Generate one multiple-choice Speech Content question.

[Question Design]
- Ask about what was said (spoken dialogue or narration) across multiple events,
  requiring the model to chain speech evidence from at least two events.
- Visual information must NOT be needed to answer.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be plausible but incorrect speech content from the same
  trajectory or nearby events.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: all evidence steps for this task type should be labeled "audio".

[task_specific_key]
Use: {{"key_name": "speech_type", "key_value": "dialogue|narration|announcement|other"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "a_sound": f"""
Task: Generate one multiple-choice Environmental Sound question.

[Question Design]
- Ask about non-human environmental sounds (rain, machinery, crowd, nature)
  that must be identified or reasoned about across multiple events.
- Visual information must NOT be needed to answer.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be other plausible environmental sounds from the video.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: all evidence steps for this task type should be labeled "audio".

[task_specific_key]
Use: {{"key_name": "sound_category", "key_value": "<type of environmental sound>"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "a_music": f"""
Task: Generate one multiple-choice Background Audio question.

[Question Design]
- Ask about background music, ambient sound, or non-foreground audio cues that
  must be reasoned about across multiple events.
- Visual information must NOT be needed to answer.
- Produce exactly 4 options (A/B/C/D), one correct.
- Distractors should be other plausible background audio descriptions.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: all evidence steps for this task type should be labeled "audio".

[task_specific_key]
Use: {{"key_name": "background_audio_type", "key_value": "music|ambient|other"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "halluc_v2a": f"""
Task: Generate one multiple-choice Visual-to-Audio Hallucination question.

[Question Design]
- Present real visual evidence from the trajectory and ask about an audio detail
  that does NOT actually exist in the video.
- The correct answer must be the option that states the audio is absent or did
  not occur.
- The other three distractors should be fabricated but plausible audio details
  that the visual scene might suggest.
- The question stem should NOT reveal that it is a hallucination test; phrase it
  as a genuine audio inquiry.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: evidence steps are labeled "video" (visual cues drive the hallucination).

[task_specific_key]
Use: {{"key_name": "halluc_type", "key_value": "visual->audio"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "halluc_a2v": f"""
Task: Generate one multiple-choice Audio-to-Visual Hallucination question.

[Question Design]
- Present real audio evidence from the trajectory and ask about a visual detail
  that does NOT actually exist in the video.
- The correct answer must be the option that states the visual is absent or did
  not occur.
- The other three distractors should be fabricated but plausible visual details
  that the audio might suggest.
- The question stem should NOT reveal that it is a hallucination test; phrase it
  as a genuine visual inquiry.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}
Note: evidence steps are labeled "audio" (audio cues drive the hallucination).

[task_specific_key]
Use: {{"key_name": "halluc_type", "key_value": "audio->visual"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
    "halluc_splicing": f"""
Task: Generate one multiple-choice Hallucinated Splice question.

[Question Design]
- Present a fabricated narrative that splices real fragments from different,
  non-adjacent time points as if they form a coherent sequence.
- The correct answer must identify that the described sequence is impossible
  or did not happen as described (e.g., "This sequence never occurred because
  event A happened at minute X and event B at minute Y with no connection").
- The other three distractors should accept the false narrative or propose
  alternative but equally false splices.
- Use the actual timestamps from the trajectory to ground the impossibility.

{_STEP3_QUALITY_RULE}

{_STEP3_LABEL_RULE}

{_STEP3_DISTRACTOR_RULE}

[task_specific_key]
Use: {{"key_name": "halluc_type", "key_value": "splice"}}

{_STEP3_OPTION_SELECTION_RULE}

{_STEP3_TIMESTAMP_RULE}

{_STEP3_OUTPUT_SCHEMA}
""",
}

def get_step1_system_prompt(task_type: str) -> str:
    return STEP1_SYSTEM_PROMPTS.get(task_type, STEP1_SYSTEM_PROMPTS["av_retrieval"])

def get_step3_system_prompt(task_type: str) -> str:
    return STEP3_SYSTEM_PROMPTS.get(task_type, STEP3_SYSTEM_PROMPTS["av_retrieval"])

def get_task_instruction(task_type: str) -> str:
    return TASK_PROMPTS.get(task_type, TASK_PROMPTS["av_retrieval"])

def get_step1_user_payload_schema() -> Dict[str, object]:
    return {
        "trajectory_candidates": [
            {
                "trajectory_id": "string",
                "event_ids": [1, 2, 3],
                "why_multihop": "string",
                "range_type": "short|long",
                "bridge_type": "entity|semantic|temporal|hybrid",
                "question_direction": {
                    "focus": "what the question should ask about (1 sentence)",
                    "answer_hint": "where/how the answer can be derived from the trajectory",
                    "distractor_hint": "what plausible wrong options might look like",
                    "answer_mode_hint": "single|multiple",
                },
            }
        ]
    }

def get_step3_user_payload_schema() -> Dict[str, object]:
    return {
        "final_trajectory": [
            {
                "event_id": 1,
                "evidence": "exact textual evidence from summary or raw_segments",
                "label": "audio|visual|audio-visual",
                "reason": "why this modality label",
                "timestamp_minute": 5,
            }
        ],
        "task_specific_key": {
            "key_name": "task-related field name",
            "key_value": "task-related value",
        },
        "question_type": "single|multiple",
        "question": "MCQ question stem",
        "options": {"A": "option text", "B": "option text", "C": "option text", "D": "option text"},
        "correct_options": ["A"],
        "answer_text": "exact text of the correct option",
    }

def get_step_prompts(task_type: str) -> Tuple[str, str]:
    return get_step1_system_prompt(task_type), get_step3_system_prompt(task_type)
