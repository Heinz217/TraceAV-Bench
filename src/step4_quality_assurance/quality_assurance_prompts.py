from typing import Any, Dict


DIRECT_ANSWER_SYSTEM_PROMPT = """You are a strict multiple-choice solver.
You only see question text and options. You do NOT have access to the video.
You are NOT told whether this question is single-choice or multi-choice.
Infer from wording only. If confidence is insufficient, abstain instead of guessing.
Return valid JSON only.
"""


def build_direct_answer_user_prompt(question: str, options: Dict[str, str]) -> str:
    return f"""Solve the following benchmark question using text-only clues.

Question:
{question}

Options:
A. {options.get("A", "")}
B. {options.get("B", "")}
C. {options.get("C", "")}
D. {options.get("D", "")}

Return a JSON object with this schema:
{{
  "predicted_options": ["A"],
  "confidence": 0.72,
  "reason": "brief reason"
}}

Rules:
- predicted_options must be a list of unique letters from A/B/C/D.
- If uncertain, you may abstain by returning an empty list:
  "predicted_options": []
- confidence must be a number in [0, 1].
- You are not given question_type. Do not assume single-choice by default.
- Do not force a guess when uncertain.
"""

DIRECT_ANSWER_FALLBACK: Dict[str, Any] = {
    "predicted_options": [],
    "confidence": 0.0,
    "reason": "fallback_due_to_api_or_json_error",
}


VERIFICATION_SYSTEM_PROMPT = """You are a strict QA benchmark verifier.
Given the question, options, reference answer, and trajectory evidence, return JSON only.
Evaluate quality and potential flaws conservatively.
"""


def build_verification_user_prompt(
    task_type: str,
    question: str,
    options: Dict[str, str],
    question_type: str,
    correct_options: list,
    answer_text: str,
    trajectory_with_timestamps: list,
) -> str:
    return f"""Verify this benchmark item.

Task type: {task_type}
Question type: {question_type}
Question:
{question}

Options:
A. {options.get("A", "")}
B. {options.get("B", "")}
C. {options.get("C", "")}
D. {options.get("D", "")}

Reference correct_options: {correct_options}
Reference answer_text: {answer_text}

Trajectory evidence:
{trajectory_with_timestamps}

Please evaluate from these angles:
1) Is the multi-hop chain real and correct?
2) Is this pseudo multi-hop (answer solvable by one hop or one local clue)?
3) Option quality:
   - obvious superficial signals (e.g., one option much longer / very different style),
   - are wrong options relevant to the question instead of random noise,
   - are wrong options plausibly confusing and semantically close enough.
4) Answer leakage:
   - can the answer be directly found from wording in the question stem itself?

Return JSON with this schema:
{{
  "is_real_multihop": true,
  "multihop_issue": "none or short text",
  "is_pseudo_multihop": false,
  "pseudo_multihop_reason": "short text",
  "option_quality": {{
    "has_obvious_pattern": false,
    "distractors_relevant": true,
    "distractors_plausible": true,
    "overall_quality": "good",
    "reason": "short text"
  }},
  "answer_leakage_detected": false,
  "answer_leakage_reason": "short text",
  "should_drop": false,
  "drop_reasons": [],
  "severity": "none",
  "summary": "short text"
}}

Rules:
- overall_quality must be one of: very_poor, poor, fair, good, excellent.
- severity must be one of: none, low, medium, high.
- should_drop should be true for major quality flaws.
"""


VERIFICATION_FALLBACK: Dict[str, Any] = {
    "is_real_multihop": True,
    "multihop_issue": "fallback_due_to_api_or_json_error",
    "is_pseudo_multihop": False,
    "pseudo_multihop_reason": "fallback_due_to_api_or_json_error",
    "option_quality": {
        "has_obvious_pattern": False,
        "distractors_relevant": True,
        "distractors_plausible": True,
        "overall_quality": "fair",
        "reason": "fallback_due_to_api_or_json_error",
    },
    "answer_leakage_detected": False,
    "answer_leakage_reason": "fallback_due_to_api_or_json_error",
    "should_drop": False,
    "drop_reasons": ["fallback_due_to_api_or_json_error"],
    "severity": "low",
    "summary": "fallback_due_to_api_or_json_error",
}
