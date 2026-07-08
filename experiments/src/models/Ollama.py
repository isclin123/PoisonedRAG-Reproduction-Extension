import requests

from .Model import Model


class Ollama(Model):
    def __init__(self, config):
        super().__init__(config)
        params = config["params"]
        self.endpoint = params.get("endpoint", "http://127.0.0.1:11434/api/generate")
        self.max_output_tokens = int(params.get("max_output_tokens", 150))
        self.timeout_seconds = float(params.get("timeout_seconds", 120))
        self.num_ctx = int(params.get("num_ctx", 4096))
        self.system_prompt = params.get("system_prompt", "You are a helpful assistant.")
        self.think = params.get("think", None)
        if isinstance(self.think, str):
            self.think = self.think.lower() == "true"

    def query(self, msg):
        payload = {
            "model": self.name,
            "prompt": msg,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if self.think is not None:
            payload["think"] = self.think

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(e)
            return ""
