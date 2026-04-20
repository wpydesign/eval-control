# Apps

User-facing products built on top of the core engine.

## What goes here

Applications, dashboards, CLI tools, and products that use the core engine to solve specific problems.

## Rules

1. **Never modify core files.** Apps depend on core but never change it.
2. **Use adapters when possible.** Don't bypass the adapter layer.
3. **Own your data.** Each app manages its own configuration, storage, and UI.

## Examples

```
apps/
  deployment_gate/        — pre-deployment risk gate for CI/CD
  quality_dashboard/      — web dashboard for monitoring risk scores
  label_queue/            — human-in-the-loop labeling interface
  comparison_tool/        — A/B model comparison using manifold analysis
```
