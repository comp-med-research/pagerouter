"""
Provider-agnostic VLM agent interface for the multi-agent ablation.

Supported providers: anthropic, openai, google, moonshot, together.
API keys are read from environment variables — missing keys skip that agent.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

# Canonical model names shared with routing.py
MODELS = [
    "chandra2", "chatgpt_api", "deepseek_ocr_2", "docling_ocr", "dolphin_1_5",
    "dotsocr", "glmocr", "got_ocr2", "hunyuanocr", "mineru_1_2b",
    "monkeyocr_pro_3b", "paddleocrVL_1_5", "rolmocr", "youtu",
]

# cost_per_call = rough estimate per routing call (image + short text); tune from your invoices.
AGENT_REGISTRY: list[dict] = [
    {"name": "claude", "tier": "heavy", "provider": "anthropic", "model": "claude-sonnet-4-6", "cost_per_call": 0.003},
    {"name": "gpt", "tier": "heavy", "provider": "openai", "model": "gpt-5.5", "cost_per_call": 0.010},
    {"name": "gemini", "tier": "heavy", "provider": "google", "model": "gemini-2.5-pro", "cost_per_call": 0.004},
    {"name": "kimi", "tier": "light", "provider": "moonshot", "model": "moonshot-v1-8k-vision", "cost_per_call": 0.001},
    {"name": "qwen", "tier": "light", "provider": "together", "model": "Qwen/Qwen2.5-VL-7B-Instruct", "cost_per_call": 0.0005},
    {"name": "internvl", "tier": "light", "provider": "together", "model": "OpenGVLab/InternVL2-8B", "cost_per_call": 0.0005},
]

_ENV_KEY: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "moonshot":  "MOONSHOT_API_KEY",
    "together":  "TOGETHER_API_KEY",
}

_MY_CHOICE_RE = re.compile(r"my choice\s*[:\-]\s*(\S+)", re.IGNORECASE)


class VLMAgent:
    """Unified interface for calling any VLM with an image + text prompt."""

    def __init__(self, name: str, provider: str, model: str) -> None:
        self.name = name
        self.provider = provider
        self.model = model

        api_key = os.environ.get(_ENV_KEY[provider], "")
        if not api_key:
            raise EnvironmentError(
                f"Agent '{name}' requires {_ENV_KEY[provider]} to be set"
            )
        self._api_key = api_key
        self._client: object | None = None  # lazy-init per provider

    # ── Public API ────────────────────────────────────────────────────────────

    def call(self, image_path: str | Path, prompt: str) -> tuple[str, float]:
        """Call the VLM with an image and prompt.

        Returns
        -------
        (response_text, latency_seconds)
            latency_seconds is wall-clock time from first byte sent to last byte received.
        """
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        dispatch = {
            "anthropic": self._call_anthropic,
            "openai":    self._call_openai,
            "google":    self._call_google,
            "moonshot":  self._call_moonshot,
            "together":  self._call_together,
        }
        t0 = time.perf_counter()
        text = dispatch[self.provider](image_b64, media_type, prompt)
        return text, time.perf_counter() - t0

    def parse_model_choice(self, response: str) -> str | None:
        """Extract a model name from the response text.

        Priority:
          1. "My choice: <name>" pattern
          2. Exact match against known model names (case-insensitive, stripped)
          3. Substring scan for any known model name
        Returns None if no valid model name is found.
        """
        m = _MY_CHOICE_RE.search(response)
        if m:
            candidate = m.group(1).strip(" .,\n").lower()
            exact = next((mod for mod in MODELS if mod == candidate), None)
            if exact:
                return exact

        normalised = response.lower().strip(" .,\n")
        exact = next((mod for mod in MODELS if mod == normalised), None)
        if exact:
            return exact

        return next((mod for mod in MODELS if mod in normalised), None)

    # ── Provider implementations ──────────────────────────────────────────────

    def _call_anthropic(self, image_b64: str, media_type: str, prompt: str) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def _call_openai(self, image_b64: str, media_type: str, prompt: str) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        data_uri = f"data:{media_type};base64,{image_b64}"
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content.strip()

    def _call_google(self, image_b64: str, media_type: str, prompt: str) -> str:
        import google.generativeai as genai

        if self._client is None:
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.model)
        response = self._client.generate_content([
            {"inline_data": {"mime_type": media_type, "data": image_b64}},
            prompt,
        ])
        return response.text.strip()

    def _call_moonshot(self, image_b64: str, media_type: str, prompt: str) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.moonshot.cn/v1",
            )
        data_uri = f"data:{media_type};base64,{image_b64}"
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content.strip()

    def _call_together(self, image_b64: str, media_type: str, prompt: str) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.together.xyz/v1",
            )
        data_uri = f"data:{media_type};base64,{image_b64}"
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content.strip()


def available_agents(requested: list[str] | None = None) -> list[VLMAgent]:
    """Return VLMAgent instances for agents whose API key is available.

    If *requested* is given, only those agent names are considered.
    Agents with missing keys are skipped with a warning.
    """
    import warnings

    agents = []
    registry = AGENT_REGISTRY
    if requested:
        names = set(requested)
        registry = [a for a in AGENT_REGISTRY if a["name"] in names]
        unknown = names - {a["name"] for a in AGENT_REGISTRY}
        if unknown:
            raise ValueError(f"Unknown agent names: {sorted(unknown)}")

    for spec in registry:
        key = os.environ.get(_ENV_KEY[spec["provider"]], "")
        if not key:
            warnings.warn(
                f"Skipping agent '{spec['name']}': {_ENV_KEY[spec['provider']]} not set",
                stacklevel=2,
            )
            continue
        agents.append(VLMAgent(spec["name"], spec["provider"], spec["model"]))

    return agents
