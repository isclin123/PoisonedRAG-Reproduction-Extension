import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.models import create_model


def make_config():
    return {
        "model_info": {
            "provider": "ollama",
            "name": "qwen3.5:9b",
        },
        "api_key_info": {
            "api_keys": [],
            "api_key_use": 0,
        },
        "params": {
            "temperature": 0.1,
            "seed": 100,
            "gpus": [],
            "max_output_tokens": 32,
            "endpoint": "http://127.0.0.1:11434/api/generate",
            "timeout_seconds": 5,
            "num_ctx": 4096,
            "think": False,
        },
    }


class OllamaModelTests(unittest.TestCase):
    def test_factory_creates_ollama_model_without_api_key(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(make_config(), f)
            config_path = f.name

        model = create_model(config_path)

        self.assertEqual(model.provider, "ollama")
        self.assertEqual(model.name, "qwen3.5:9b")

    @patch("requests.post")
    def test_query_returns_ollama_response_text(self, post):
        from src.models.Ollama import Ollama

        response = Mock()
        response.json.return_value = {"response": "  24  "}
        response.raise_for_status.return_value = None
        post.return_value = response

        model = Ollama(make_config())
        answer = model.query("Question?")

        self.assertEqual(answer, "24")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertEqual(payload["prompt"], "Question?")
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_predict"], 32)


if __name__ == "__main__":
    unittest.main()
