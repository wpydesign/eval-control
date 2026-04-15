"""
local_bridge.py — Local LLM bridge for lattice execution.

Runs inference via llama-cpp-python on a local GGUF model.
No API required. No rate limits. Pure CPU inference.

This replaces the z-ai-web-dev-sdk bridge (llm_bridge.mjs) for
situations where the API is unavailable or rate-limited.

Model: Qwen2.5-0.5B-Instruct (Q2_K quantization, ~415MB)
Performance: ~0.7s per call on 4-core CPU
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger("lattice.local_bridge")

# Model path (pre-downloaded)
_DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/"
    "snapshots/9217f5db79a29953eb74d5343926648285ec7e67/"
    "qwen2.5-0.5b-instruct-q2_k.gguf"
)

_llm_instance = None


def get_model(model_path: Optional[str] = None):
    """Lazy-load the model singleton."""
    global _llm_instance
    if _llm_instance is None:
        from llama_cpp import Llama
        path = model_path or _DEFAULT_MODEL_PATH
        logger.info(f"Loading local model: {os.path.basename(path)}")
        start = time.time()
        _llm_instance = Llama(
            model_path=path,
            n_ctx=512,
            n_threads=4,
            verbose=False,
        )
        logger.info(f"Model loaded in {time.time()-start:.1f}s")
    return _llm_instance


def call_local_llm(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
    model_path: Optional[str] = None,
) -> dict:
    """
    Call the local LLM and return structured result.

    Interface matches call_llm() from run_lattice.py.

    Args:
        prompt: The prompt text.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        model_path: Override model path.

    Returns:
        dict with "content", "token_count", "finish_reason", "latency_ms".
    """
    llm = get_model(model_path)
    start = time.time()

    try:
        resp = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        latency_ms = int((time.time() - start) * 1000)
        content = resp["choices"][0]["text"].strip()
        token_count = resp.get("usage", {}).get("completion_tokens", len(content.split()))
        finish_reason = "stop" if token_count < max_tokens else "length"

        return {
            "content": content,
            "token_count": token_count,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        logger.error(f"Local inference error: {e}")
        return {
            "content": f"[LOCAL ERROR] {e}",
            "token_count": 0,
            "finish_reason": "error",
            "latency_ms": latency_ms,
        }
