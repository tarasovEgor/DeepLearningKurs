import requests
from typing import List, Dict


def search_openalex(query: str, per_page: int = 5) -> List[dict]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "select": "id,display_name,publication_year,abstract_inverted_index,authorships"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def invert_abstract(inv_idx: dict) -> str:
    if not inv_idx:
        return ""

    words = []
    for token, positions in inv_idx.items():
        for pos in positions:
            words.append((pos, token))

    words.sort(key=lambda x: x[0])
    return " ".join(token for _, token in words)


def search_wikipedia(query: str) -> str:
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
    response = requests.get(summary_url, timeout=30)

    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "")

    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1
    }
    response = requests.get(search_url, params=params, timeout=30)
    if response.status_code != 200:
        return ""

    results = response.json().get("query", {}).get("search", [])
    if not results:
        return ""

    title = results[0]["title"]
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
    response = requests.get(summary_url, timeout=30)
    if response.status_code != 200:
        return ""

    data = response.json()
    return data.get("extract", "")


def extract_authors(work: Dict) -> list[str]:
    authors = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)
    return authors