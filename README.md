# heretic-enhanced

Heretic Enhanced LLM Abliteration Pipeline

Inspired by: https://github.com/p-e-w/heretic

## Requirements

- Python 3.10+
- A supported PyTorch installation for the target hardware
- `heretic-llm` 1.4.0 or newer

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Create a configuration file, then edit the model and hardware settings:

```bash
python heretic.py --write-config config.yaml
python heretic.py --config config.yaml --dry-run
python heretic.py --config config.yaml
```

The dry run prints the exact `heretic` command without running the model. Outputs
and logs are written below the configured `output_root`.

For a CPU-only run, remove `device_map: auto` from both the `abliteration` and
`hardware` sections in `config.yaml`. Use environment variables for notification
and Hugging Face credentials; do not commit `config.yaml`.

## Docker

The image expects a runtime-mounted `config.yaml`:

```bash
docker build -t heretic-enhanced .
docker run --gpus all --rm \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/output:/app/output" \
  heretic-enhanced
```

## Features

- YAML configuration with CLI overrides
- Parallel batch processing
- Checkpoint/resume with persisted result state
- Optional `lm-eval` benchmarks
- Optional GGUF, Ollama, and Hugging Face post-processing
- Structured logs and optional email, Slack, or Discord notifications
"""
