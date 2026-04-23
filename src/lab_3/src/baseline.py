from .tools import search_wikipedia
from .prompts import BASELINE_PROMPT


def run_baseline(topic: str, llm_call):
    wiki_context = search_wikipedia(topic)
    prompt = BASELINE_PROMPT.format(
        topic=topic,
        wiki_context=wiki_context if wiki_context else "Контекст по Wikipedia не найден."
    )
    return llm_call(prompt)