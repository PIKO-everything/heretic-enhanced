# heretic-enhanced


Heretic Enhanced LLM Abliteration Pipeline

Inspired by: https://github.com/p-e-w/heretic

Features:
- YAML configuration with CLI overrides
- Parallel batch processing using multiprocessing (logger per worker)
- Automatic VRAM detection & dynamic batch sizing
- Checkpoint/resume with full state persistence
- Benchmark suite: perplexity (WikiText-2), MMLU (5-shot), refusal rate, KL divergence
- Weights & Biases integration (optional)
- Dockerised execution for reproducibility
- Hugging Face upload with auto‑generated model card
- Real‑time progress bars (tqdm) and structured logging
- Email, Slack, Discord, and desktop notifications
"""
