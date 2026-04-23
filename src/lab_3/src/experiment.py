import time
import pandas as pd
from .utils import load_json, ensure_dir, slugify
from .state import AgentState, save_trace
from .evaluator import evaluate_answer
from .metrics import rubric_mean, count_present_sections


def load_topics(path: str) -> list[str]:
    return load_json(path)


def run_experiment(topics, mode_name, runner, llm_call, output_dir="outputs"):
    ensure_dir(f"{output_dir}/traces")
    ensure_dir(f"{output_dir}/results")

    records = []

    for topic in topics:
        start = time.time()
        result = runner(topic, llm_call)
        latency = time.time() - start

        if isinstance(result, AgentState):
            answer = result.final_answer
            notes = result.notes
            n_steps = len(result.history)
            n_sources = len(result.sources)
            stop_reason = result.stop_reason

            trace_path = f"{output_dir}/traces/{mode_name}_{slugify(topic)}.json"
            save_trace(result, trace_path)
        else:
            answer = result
            notes = []
            n_steps = 3
            n_sources = 0
            stop_reason = "baseline_completed"

        eval_json = evaluate_answer(answer, notes, llm_call)

        records.append({
            "topic": topic,
            "mode": mode_name,
            "correctness": eval_json["correctness"],
            "groundedness": eval_json["groundedness"],
            "completeness": eval_json["completeness"],
            "coverage": eval_json["coverage_of_required_fields"],
            "source_consistency": eval_json["source_consistency"],
            "rubric": rubric_mean(eval_json),
            "present_sections": count_present_sections(answer),
            "n_steps": n_steps,
            "n_sources": n_sources,
            "latency": latency,
            "stop_reason": stop_reason,
            "comment": eval_json.get("comment", "")
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{output_dir}/results/{mode_name}_records.csv", index=False, encoding="utf-8")
    return records


def summarize_records(all_records, output_dir="outputs"):
    ensure_dir(f"{output_dir}/results")
    df = pd.DataFrame(all_records)

    summary = df.groupby("mode")[[
        "correctness",
        "groundedness",
        "completeness",
        "coverage",
        "source_consistency",
        "rubric",
        "n_steps",
        "n_sources",
        "latency"
    ]].mean()

    summary.to_csv(f"{output_dir}/results/summary.csv", encoding="utf-8")
    return df, summary