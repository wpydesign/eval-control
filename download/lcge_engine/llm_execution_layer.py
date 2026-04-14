"""
llm_execution_layer.py — Step 2: LLM Execution Layer

Calls the LLM API for each prompt variant via z-ai-web-dev-sdk Node.js bridge.
Stores: node_id, prompt, response, token_count, refusal_flag

Two-model pairing:
    - Primary: high-capability model (z-ai-web-dev-sdk default)
    - Baseline: same model for MVP (configurable to cheaper model)

Every call is isolated. No conversation history. No state.
"""

import json
import subprocess
import tempfile
import os
from typing import Optional

from .config import PRIMARY_MODEL, BASELINE_MODEL


def _call_llm_via_bridge(prompt: str, model_role: str = "primary") -> dict:
    """
    Call LLM via z-ai-web-dev-sdk Node.js bridge.

    This uses a thin Node.js script to access the z-ai-web-dev-sdk
    which is already installed in the environment.

    Args:
        prompt: The prompt text to send.
        model_role: "primary" or "baseline".

    Returns:
        dict with "content", "token_count", "finish_reason"
    """
    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump({"prompt": prompt, "role": model_role}, f)
        temp_path = f.name

    try:
        bridge_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "llm_bridge.mjs"
        )
        result = subprocess.run(
            ["node", bridge_script, temp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            return {
                "content": f"[LLM CALL FAILED] {error_msg}",
                "token_count": 0,
                "finish_reason": "error",
            }

        # Try to parse JSON output
        output = result.stdout.strip()
        # Find the last JSON object in output (skip any log lines)
        lines = output.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    return {
                        "content": parsed.get("content", ""),
                        "token_count": parsed.get("token_count", 0),
                        "finish_reason": parsed.get("finish_reason", "stop"),
                    }
                except json.JSONDecodeError:
                    continue

        # Fallback: treat entire output as content
        return {
            "content": output,
            "token_count": len(output.split()),
            "finish_reason": "stop",
        }
    except subprocess.TimeoutExpired:
        return {
            "content": "[LLM CALL TIMED OUT]",
            "token_count": 0,
            "finish_reason": "timeout",
        }
    except Exception as e:
        return {
            "content": f"[LLM CALL ERROR] {str(e)}",
            "token_count": 0,
            "finish_reason": "error",
        }
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


class LLMResponse:
    """Stores the complete response data for a single node."""

    def __init__(
        self,
        node_id: str,
        prompt: str,
        model_role: str,
        content: str,
        token_count: int,
        finish_reason: str,
    ):
        self.node_id = node_id
        self.prompt = prompt
        self.model_role = model_role
        self.content = content
        self.token_count = token_count
        self.finish_reason = finish_reason

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "prompt": self.prompt,
            "model_role": self.model_role,
            "response": self.content,
            "token_count": self.token_count,
            "finish_reason": self.finish_reason,
        }

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f"LLMResponse(id={self.node_id}, tokens={self.token_count}, preview='{preview}...')"


def execute_variant(
    prompt_variant,
    model_role: str = "primary",
) -> LLMResponse:
    """
    Execute a single prompt variant against the LLM.

    Args:
        prompt_variant: A PromptVariant object.
        model_role: "primary" or "baseline".

    Returns:
        LLMResponse with the model's output.
    """
    raw = _call_llm_via_bridge(prompt_variant.prompt, model_role)
    return LLMResponse(
        node_id=prompt_variant.node_id,
        prompt=prompt_variant.prompt,
        model_role=model_role,
        content=raw["content"],
        token_count=raw["token_count"],
        finish_reason=raw["finish_reason"],
    )


def execute_all_variants(
    variants: list,
    models: Optional[list] = None,
) -> list[LLMResponse]:
    """
    Execute all prompt variants against specified models.

    Args:
        variants: List of PromptVariant objects.
        models: List of model roles to test (default: ["primary", "baseline"]).

    Returns:
        List of LLMResponse objects (one per variant per model).
    """
    if models is None:
        models = ["primary", "baseline"]

    responses = []
    for variant in variants:
        for model_role in models:
            response = execute_variant(variant, model_role)
            # Tag with strategy info for downstream analysis
            response.strategy = variant.strategy  # type: ignore
            response.variant_index = variant.variant_index  # type: ignore
            response.task = variant.task  # type: ignore
            responses.append(response)

    return responses
