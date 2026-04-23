import json
from .prompts import EVAL_PROMPT, REVISION_PROMPT
from .utils import extract_first_json_object


def evaluate_answer(answer: str, notes: list, llm_call) -> dict:
    prompt = EVAL_PROMPT.format(
        answer=answer,
        notes_json=json.dumps(notes, ensure_ascii=False, indent=2)
    )
    raw = llm_call(prompt, temperature=0.0)

    json_str = extract_first_json_object(raw)
    if not json_str:
        return {
            "correctness": 0,
            "groundedness": 0,
            "completeness": 0,
            "coverage_of_required_fields": 0,
            "source_consistency": 0,
            "comment": "Evaluator did not return valid JSON."
        }

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "correctness": 0,
            "groundedness": 0,
            "completeness": 0,
            "coverage_of_required_fields": 0,
            "source_consistency": 0,
            "comment": "Evaluator JSON parse error."
        }

    required_keys = [
        "correctness",
        "groundedness",
        "completeness",
        "coverage_of_required_fields",
        "source_consistency",
        "comment"
    ]
    for key in required_keys:
        if key not in data:
            data[key] = 0 if key != "comment" else "Missing key."

    return data


def needs_revision(eval_json: dict, threshold: int = 4) -> bool:
    return (
        eval_json["groundedness"] < threshold
        or eval_json["coverage_of_required_fields"] < threshold
        or eval_json["source_consistency"] < threshold
    )


def revise_answer(topic: str, answer: str, notes: list, eval_json: dict, llm_call) -> str:
    prompt = REVISION_PROMPT.format(
        topic=topic,
        answer=answer,
        notes_json=json.dumps(notes, ensure_ascii=False, indent=2),
        eval_json=json.dumps(eval_json, ensure_ascii=False, indent=2)
    )
    return llm_call(prompt, temperature=0.1)