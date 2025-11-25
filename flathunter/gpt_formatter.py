"""Helpers to format expose data with GPT-powered APIs."""
from __future__ import annotations

import json
from typing import Dict, Optional

import requests

from flathunter.logging import logger


class GPTExposeFormatter:
    """Format expose dictionaries using a GPT-style chat completion API."""

    def __init__(self, config) -> None:
        self._config = config
        self._enabled = bool(config.gpt_enabled())
        self._api_key = config.gpt_api_key()
        self._api_base = config.gpt_api_base()
        self._model = config.gpt_model()
        self._temperature = config.gpt_temperature()
        self._system_prompt = config.gpt_system_prompt()
        self._timeout = config.gpt_timeout_seconds()

    def format(self, expose: Dict, fallback: str) -> str:
        """
        Return a GPT-generated message for the expose or the fallback text on failure.

        :param expose: Listing data that should be summarised.
        :param fallback: Message used if GPT is disabled or errors occur.
        """
        if not self._should_use_gpt():
            return fallback

        try:
            payload = self._build_payload(expose)
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            url = f"{self._api_base.rstrip('/')}/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            if response.status_code != 200:
                logger.warning(
                    "GPT API returned %s: %s",
                    response.status_code,
                    response.text,
                )
                return fallback

            data = response.json()
            message = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not message:
                logger.warning("GPT API response missing content, falling back to default message")
                return fallback
            return message.strip()
        except requests.RequestException:
            logger.warning("Error while calling GPT API, falling back to default message", exc_info=True)
            return fallback

    def _should_use_gpt(self) -> bool:
        return self._enabled and bool(self._api_key) and bool(self._model)

    def _build_payload(self, expose: Dict) -> Dict:
        user_content = (
            "Проанализируй следующую информацию об объявлении и составь краткий, "
            "чёткий текст на русском языке. Сделай акцент на ключевых выгодах, "
            "укажи цену, район/адрес, количество комнат, площадь и добавь ссылку. "
            "Ответ должен занимать не более 6-7 предложений.\n\n"
            f"Данные объявления (JSON):\n{json.dumps(expose, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        return {
            "model": self._model,
            "temperature": self._temperature,
            "messages": messages,
        }

    def enabled(self) -> bool:
        """Expose enabled flag for external checks/tests."""
        return self._should_use_gpt()



