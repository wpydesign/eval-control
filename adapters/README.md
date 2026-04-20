# Adapters

Domain-specific connectors for the core engine.

## What goes here

Adapters translate between external systems and the core interface contract defined in `../INTERFACE.md`.

## Rules

1. **Never modify core files.** Adapters import from core, never the other way around.
2. **Never change the response shape.** Pass the core response through as-is.
3. **Use the canonical interface.** All adapters must produce and consume the JSON format defined in `../INTERFACE.md`.

## Examples

```
adapters/
  openai_eval.py          — evaluate OpenAI outputs
  anthropic_eval.py       — evaluate Anthropic outputs
  custom_llm.py           — generic LLM adapter
  batch_processor.py      — batch evaluation pipeline
  webhook_receiver.py     — receive evaluations via HTTP webhook
```

## Template

```python
"""Adapter template for {your system}."""
import json
from core.survival import SurvivalEngine, SurvivalConfig

class MySystemAdapter:
    def __init__(self, config):
        self.engine = SurvivalEngine(config)

    def evaluate(self, prompt, **kwargs):
        """Evaluate a prompt from {your system}.

        Returns the canonical response format defined in INTERFACE.md.
        Do NOT transform or add fields.
        """
        result = self.engine.evaluate(prompt)
        # Return as-is — the core response IS the canonical format
        return result.to_dict()
```
