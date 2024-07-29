from dataclasses import dataclass

import requests


@dataclass
class OllamaModel:
    model: str = "phi3:mini"
    base_url: str = "http://0.0.0.0:11434/api/chat"

    def __call__(self, messages: list[dict]) -> str:
        response = requests.post(
            self.base_url,
            json={"model": self.model, "messages": messages, "stream": False},
        )
        response.raise_for_status()

        message = response.json().get("message", {})
        content = message.get("content", "")
        return content
