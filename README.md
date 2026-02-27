# mcp-graph-eval

Lightweight tooling to evaluate MCP graph QA runs and compare model results.

## Quick Start

```bash
uv sync
cp .env.example .env
```

## Run Evaluation

```bash
uv run python -m src.eval.runner \
  --samples produced_samples.json \
  --limit 20 \
  --output results/hypmol/eval_results.toml
```

## Build Combined Reports

```bash
uv run python -m src.scoring.runner --all
```
