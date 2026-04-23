import json
import time
from .state import AgentState, log_step
from .tools import search_wikipedia, search_openalex, invert_abstract, extract_authors
from .prompts import AGENT_FINAL_PROMPT, QUERY_REWRITE_PROMPT
from .evaluator import evaluate_answer, needs_revision, revise_answer


def can_continue(state: AgentState, max_steps: int) -> bool:
    return state.step_id < max_steps


def rewrite_search_query(topic: str, llm_call) -> str:
    prompt = QUERY_REWRITE_PROMPT.format(topic=topic)
    raw = llm_call(prompt, temperature=0.0)
    return raw.strip().splitlines()[0].strip().strip('"')


def deduplicate_works(works: list[dict]) -> list[dict]:
    seen = set()
    uniq = []
    for work in works:
        work_id = work.get("id")
        if work_id and work_id not in seen:
            uniq.append(work)
            seen.add(work_id)
    return uniq


def prepare_notes(papers: list[dict], top_k: int = 5) -> list[dict]:
    prepared = []

    for paper in papers:
        abstract = invert_abstract(paper.get("abstract_inverted_index"))
        year = paper.get("publication_year")
        authors = extract_authors(paper)

        score = 0
        if abstract:
            score += 2
        if year:
            score += 1

        prepared.append({
            "id": paper.get("id", ""),
            "title": paper.get("display_name", ""),
            "year": year,
            "authors": authors,
            "abstract": abstract[:1500],
            "_score": score
        })

    prepared.sort(key=lambda x: (x["_score"], x["year"] or 0), reverse=True)

    result = []
    for item in prepared[:top_k]:
        item.pop("_score", None)
        result.append(item)

    return result


def run_agent(
    topic: str,
    llm_call,
    max_steps: int = 6,
    top_k: int = 5,
    min_sources: int = 3,
    use_evaluator: bool = False
) -> AgentState:
    state = AgentState(
        topic=topic,
        objective="Подготовить структурированный научно-аналитический обзор темы"
    )

    # 1. Wikipedia context
    if not can_continue(state, max_steps):
        state.status = "finished"
        state.stop_reason = "max_steps_reached"
        return state

    t0 = time.time()
    wiki_context = search_wikipedia(topic)
    latency_ms = (time.time() - t0) * 1000
    log_step(
        state,
        "search_wikipedia",
        {"topic": topic},
        wiki_context[:300] if wiki_context else "empty",
        next_reason="Need general context before scientific search.",
        latency_ms=latency_ms
    )

    # 2. OpenAlex search with original query
    if not can_continue(state, max_steps):
        state.status = "finished"
        state.stop_reason = "max_steps_reached"
        return state

    t0 = time.time()
    papers = search_openalex(topic, per_page=max(top_k + 3, 8))
    latency_ms = (time.time() - t0) * 1000
    state.sources = deduplicate_works(papers)

    log_step(
        state,
        "search_openalex",
        {"query": topic, "per_page": max(top_k + 3, 8)},
        f"found={len(state.sources)}",
        next_reason="If too few sources, rewrite query and retry.",
        latency_ms=latency_ms
    )

    # 3. Retry with rewritten query if too few papers
    if len(state.sources) < min_sources and can_continue(state, max_steps):
        rewritten_query = rewrite_search_query(topic, llm_call)

        t0 = time.time()
        retry_papers = search_openalex(rewritten_query, per_page=max(top_k + 3, 8))
        latency_ms = (time.time() - t0) * 1000

        state.sources = deduplicate_works(state.sources + retry_papers)

        log_step(
            state,
            "search_openalex_retry",
            {"query": rewritten_query, "per_page": max(top_k + 3, 8)},
            f"found_total={len(state.sources)}",
            next_reason="Proceed to extract notes from available sources.",
            latency_ms=latency_ms
        )

    # 4. Extract notes
    if not can_continue(state, max_steps):
        state.status = "finished"
        state.stop_reason = "max_steps_reached"
        return state

    state.notes = prepare_notes(state.sources, top_k=top_k)
    log_step(
        state,
        "extract_notes",
        {"n_sources": len(state.sources), "top_k": top_k},
        f"notes_prepared={len(state.notes)}",
        next_reason="Generate final answer from context and notes."
    )

    # 5. Generate final answer
    if not can_continue(state, max_steps):
        state.status = "finished"
        state.stop_reason = "max_steps_reached"
        return state

    prompt = AGENT_FINAL_PROMPT.format(
        topic=topic,
        wiki_context=wiki_context if wiki_context else "Общий контекст не найден.",
        notes_json=json.dumps(state.notes, ensure_ascii=False, indent=2)
    )

    state.final_answer = llm_call(prompt)
    log_step(
        state,
        "generate_final_answer",
        {"topic": topic, "n_notes": len(state.notes)},
        state.final_answer[:400],
        next_reason="Either finish or evaluate and revise."
    )

    # 6. Evaluator + possible revision
    if use_evaluator and can_continue(state, max_steps):
        eval_json = evaluate_answer(state.final_answer, state.notes, llm_call)
        state.metrics["evaluator_before_revision"] = eval_json

        log_step(
            state,
            "evaluate_final_answer",
            {"topic": topic},
            json.dumps(eval_json, ensure_ascii=False),
            next_reason="Revise answer if evaluator says coverage/groundedness is weak."
        )

        if needs_revision(eval_json) and can_continue(state, max_steps):
            revised_answer = revise_answer(topic, state.final_answer, state.notes, eval_json, llm_call)
            state.final_answer = revised_answer

            log_step(
                state,
                "revise_final_answer",
                {"topic": topic},
                state.final_answer[:400],
                next_reason="Finish after single revision."
            )
            state.stop_reason = "revised_after_evaluation"
        else:
            state.stop_reason = "evaluator_accept"
    else:
        state.stop_reason = "final_answer_generated"

    state.status = "finished"
    return state