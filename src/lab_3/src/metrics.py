REQUIRED_SECTIONS = [
    "## Определение",
    "## Основные подходы",
    "## Ключевые работы",
    "## Применения",
    "## Ограничения",
    "## Использованные источники"
]


def count_present_sections(answer: str) -> int:
    answer_lower = answer.lower()
    return sum(1 for section in REQUIRED_SECTIONS if section.lower() in answer_lower)


def coverage_ratio(answer: str) -> float:
    return count_present_sections(answer) / len(REQUIRED_SECTIONS)


def rubric_mean(eval_json: dict) -> float:
    keys = [
        "correctness",
        "groundedness",
        "completeness",
        "coverage_of_required_fields",
        "source_consistency"
    ]
    values = [eval_json.get(k, 0) for k in keys]
    return sum(values) / len(values)