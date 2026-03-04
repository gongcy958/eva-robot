
import requests


class OllamaLlmClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout_seconds: int = 30,
        retries: int = 3,
    ) -> None:
        self._url = url
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retries = retries

    def generate(self, prompt: str, user_input: str) -> str:
        for i in range(self._retries):
            try:
                response = requests.post(
                    self._url,
                    json={
                        "model": self._model,
                        "prompt": f"{prompt}\nUser: {user_input}",
                        "stream": False,
                    },
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception as exc:
                print(f"[LLM] call failed, retry {i + 1}/{self._retries}: {exc}")

        return "Sorry, I can't respond right now."
