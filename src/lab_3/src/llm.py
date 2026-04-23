import os
import logging
import requests
from .utils import load_json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_CFG_PATH = "configs/model_config.json"
logger = logging.getLogger("lab3.llm")


def _extract_openai_content(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty response from API: {data}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        content = "".join(parts)

    return str(content).strip()


def llm_call(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    cfg = load_json(MODEL_CFG_PATH)

    provider = os.getenv("LLM_PROVIDER", cfg.get("provider", "ollama")).lower()
    timeout = cfg.get("timeout", 180)
    temp = cfg["temperature"] if temperature is None else temperature
    max_tokens = cfg.get("max_tokens", cfg.get("num_predict", 200))

    logger.info(
        "llm_call started | provider=%s | timeout=%s | max_tokens=%s | prompt_len=%d",
        provider, timeout, max_tokens, len(prompt)
    )

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", cfg["base_url"]).rstrip("/")
        model_name = model or os.getenv("LLM_MODEL") or cfg["model"]

        logger.info(
            "Ollama request | base_url=%s | model=%s | temperature=%s",
            base_url, model_name, temp
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens
            }
        }

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout
        )

        logger.info("Ollama response status=%s", response.status_code)
        response.raise_for_status()

        data = response.json()
        result = data["response"].strip()
        logger.info("Ollama response parsed | response_len=%d", len(result))
        return result

    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL", cfg["base_url"]).rstrip("/")
        model_name = model or os.getenv("LLM_MODEL") or cfg["model"]

        api_key_env = cfg.get("api_key_env", "POLZA_AI_API_KEY")
        api_key = os.getenv(api_key_env)

        logger.info(
            "OpenAI-compatible request | base_url=%s | model=%s | temperature=%s | api_key_env=%s | api_key_present=%s",
            base_url, model_name, temp, api_key_env, bool(api_key)
        )

        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' is not set")

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        url = f"{base_url}/chat/completions"
        logger.info("POST %s", url)

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        logger.info("OpenAI-compatible response status=%s", response.status_code)

        if not response.ok:
            logger.error("Response text: %s", response.text)

        response.raise_for_status()

        data = response.json()
        result = _extract_openai_content(data)
        logger.info("OpenAI-compatible response parsed | response_len=%d", len(result))
        return result

    raise ValueError(f"Unsupported provider: {provider}")