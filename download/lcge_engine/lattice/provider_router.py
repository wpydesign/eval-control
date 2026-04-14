"""
provider_router.py — Multi-provider LLM call routing (v1.4).

Providers (priority order):
    1. Groq          — free tier, 30 req/min
    2. OpenRouter    — free models available
    3. Gemini        — Google AI Studio, 1500/day free
    4. Ollama/local  — CPU inference, no rate limit

Routing rules (MANDATORY):
    - On ANY failure (429, timeout, empty response) -> immediate switch
    - No retry loops on same provider
    - No waiting
    - Track provider_used per call
"""

import json
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("lattice.provider_router")


# ============================================================
# Provider interfaces
# ============================================================

class ProviderResult:
    """Standardized result from any provider."""
    __slots__ = ("content", "token_count", "finish_reason", "latency_ms", "provider")

    def __init__(
        self,
        content: str,
        token_count: int,
        finish_reason: str,
        latency_ms: int,
        provider: str,
    ):
        self.content = content
        self.token_count = token_count
        self.finish_reason = finish_reason
        self.latency_ms = latency_ms
        self.provider = provider

    def is_valid(self) -> bool:
        """Check if the result is usable."""
        if not self.content or len(self.content.strip()) < 3:
            return False
        if self.finish_reason == "error":
            return False
        if self.content.startswith("["):
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "token_count": self.token_count,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
        }


# ============================================================
# Provider 1: Groq
# ============================================================

def _call_groq(prompt: str, temperature: float, top_p: float) -> ProviderResult:
    """Call Groq API (free tier)."""
    start = time.time()
    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return ProviderResult("", 0, "error", 0, "groq")

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=512,
        )
        latency_ms = int((time.time() - start) * 1000)
        content = resp.choices[0].message.content or ""
        token_count = resp.usage.completion_tokens if resp.usage else len(content.split())
        finish = resp.choices[0].finish_reason or "stop"

        return ProviderResult(content, token_count, finish, latency_ms, "groq")

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.debug(f"  Groq failed: {e}")
        return ProviderResult("", 0, "error", latency_ms, "groq")


# ============================================================
# Provider 2: OpenRouter
# ============================================================

def _call_openrouter(prompt: str, temperature: float, top_p: float) -> ProviderResult:
    """Call OpenRouter API (free models)."""
    start = time.time()
    try:
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return ProviderResult("", 0, "error", 0, "openrouter")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        resp = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=512,
        )
        latency_ms = int((time.time() - start) * 1000)
        content = resp.choices[0].message.content or ""
        token_count = resp.usage.completion_tokens if resp.usage else len(content.split())
        finish = resp.choices[0].finish_reason or "stop"

        return ProviderResult(content, token_count, finish, latency_ms, "openrouter")

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.debug(f"  OpenRouter failed: {e}")
        return ProviderResult("", 0, "error", latency_ms, "openrouter")


# ============================================================
# Provider 3: Gemini (Google AI Studio)
# ============================================================

def _call_gemini(prompt: str, temperature: float, top_p: float) -> ProviderResult:
    """Call Gemini API (Google AI Studio free tier)."""
    start = time.time()
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return ProviderResult("", 0, "error", 0, "gemini")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": 512,
            },
        )
        latency_ms = int((time.time() - start) * 1000)
        content = resp.text or ""
        token_count = len(content.split())
        finish = "stop"

        return ProviderResult(content, token_count, finish, latency_ms, "gemini")

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.debug(f"  Gemini failed: {e}")
        return ProviderResult("", 0, "error", latency_ms, "gemini")


# ============================================================
# Provider 4: Ollama / Local (llama-cpp-python)
# ============================================================

_llm_instance = None
_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/"
    "snapshots/9217f5db79a29953eb74d5343926648285ec7e67/"
    "qwen2.5-0.5b-instruct-q2_k.gguf"
)


def _get_local_model():
    global _llm_instance
    if _llm_instance is None:
        from llama_cpp import Llama
        logger.info(f"Loading local model (Ollama fallback)...")
        start = time.time()
        _llm_instance = Llama(
            model_path=_MODEL_PATH,
            n_ctx=512,
            n_threads=4,
            verbose=False,
        )
        logger.info(f"Local model loaded: {time.time()-start:.1f}s")
    return _llm_instance


def _call_ollama(prompt: str, temperature: float, top_p: float) -> ProviderResult:
    """Call local LLM via llama-cpp-python (Ollama-equivalent)."""
    start = time.time()
    try:
        llm = _get_local_model()

        # Convert temperature/top_p for llama-cpp
        resp = llm(
            prompt,
            max_tokens=200,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],
        )
        latency_ms = int((time.time() - start) * 1000)
        content = resp["choices"][0]["text"].strip()
        token_count = resp.get("usage", {}).get("completion_tokens", len(content.split()))
        finish = "stop" if token_count < 200 else "length"

        return ProviderResult(content, token_count, finish, latency_ms, "ollama_local")

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.error(f"  Ollama/local failed: {e}")
        return ProviderResult(f"[LOCAL ERROR] {e}", 0, "error", latency_ms, "ollama_local")


# ============================================================
# Provider 0: z-ai-web-dev-sdk (original bridge)
# ============================================================

def _call_zai_sdk(prompt: str, temperature: float, top_p: float) -> ProviderResult:
    """Call via z-ai-web-dev-sdk Node.js bridge."""
    start = time.time()
    try:
        import subprocess
        import tempfile

        bridge_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "llm_bridge.mjs",
        )
        if not os.path.exists(bridge_script):
            return ProviderResult("", 0, "error", 0, "zai_sdk")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"prompt": prompt, "role": "primary", "temperature": temperature}, f)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["node", bridge_script, temp_path],
                capture_output=True, text=True, timeout=30,
            )
            latency_ms = int((time.time() - start) * 1000)

            if result.returncode != 0:
                return ProviderResult("", 0, "error", latency_ms, "zai_sdk")

            for line in reversed(result.stdout.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    parsed = json.loads(line)
                    content = parsed.get("content", "")
                    if content and not content.startswith("["):
                        return ProviderResult(
                            content,
                            parsed.get("token_count", 0),
                            parsed.get("finish_reason", "stop"),
                            latency_ms,
                            "zai_sdk",
                        )
            return ProviderResult("", 0, "error", latency_ms, "zai_sdk")
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.debug(f"  z-ai-sdk failed: {e}")
        return ProviderResult("", 0, "error", latency_ms, "zai_sdk")


# ============================================================
# Router
# ============================================================

# Ordered provider list (priority)
_PROVIDERS = [
    ("groq", _call_groq),
    ("openrouter", _call_openrouter),
    ("gemini", _call_gemini),
    ("ollama_local", _call_ollama),
]

# z-ai-sdk skipped: known 429
_ACTIVE_PROVIDER_LIST = [
    ("groq", _call_groq),
    ("openrouter", _call_openrouter),
    ("gemini", _call_gemini),
    ("ollama_local", _call_ollama),
]

# Cached active provider list (set after first preflight)
_active_providers = None


def _is_provider_ready(name: str) -> bool:
    """Quick check if a provider is likely usable."""
    if name == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if name == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    if name == "gemini":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if name == "ollama_local":
        return os.path.exists(_MODEL_PATH)
    if name == "zai_sdk":
        bridge = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm_bridge.mjs")
        return os.path.exists(bridge)
    return False


def call_llm_routed(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    try_sdk_first: bool = True,
) -> ProviderResult:
    """
    Call LLM with automatic provider routing.

    After preflight, caches the active provider list to avoid
    calling unavailable providers on every request.

    Tries each provider in order. On ANY failure (empty response,
    error, timeout), immediately switches to next provider.
    No retry loops. No waiting.

    Args:
        prompt: The prompt text.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        try_sdk_first: Try z-ai-web-dev-sdk before Groq.

    Returns:
        ProviderResult with content, metadata, and provider name.
    """
    global _active_providers
    if _active_providers is None:
        _active_providers = [
            (name, fn) for name, fn in _ACTIVE_PROVIDER_LIST
            if _is_provider_ready(name)
        ]
        if not _active_providers:
            _active_providers = _ACTIVE_PROVIDER_LIST  # Fallback: try all

    for provider_name, call_fn in _active_providers:
        result = call_fn(prompt, temperature, top_p)

        if result.is_valid():
            return result

        # Immediate switch on failure
        logger.debug(f"  {provider_name} failed/empty, switching...")

    # All providers failed
    return ProviderResult(
        "[ALL PROVIDERS FAILED]",
        0,
        "error",
        0,
        "none",
    )


def preload_local_model():
    """Preload local model in main thread before threading begins."""
    _get_local_model()


def preflight_check() -> dict:
    """
    Check which providers are available (have API keys or local model).

    Returns:
        dict mapping provider name to availability status.
    """
    status = {}

    # Groq
    status["groq"] = bool(os.environ.get("GROQ_API_KEY"))

    # OpenRouter
    status["openrouter"] = bool(os.environ.get("OPENROUTER_API_KEY"))

    # Gemini
    status["gemini"] = bool(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )

    # Ollama/local
    status["ollama_local"] = os.path.exists(_MODEL_PATH)

    logger.info("Provider availability:")
    for name, available in status.items():
        logger.info(f"  {name}: {'AVAILABLE' if available else 'unavailable'}")

    return status
