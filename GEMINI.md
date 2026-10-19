# Agentic System - Project Context

## Overview
This repository is an environment for experimenting with small language models (SLMs) and agentic workflows, focusing on domain-specific scientific simulations and text generation.

## Current Setup & Hardware
- **Base Model**: `HuggingFaceTB/SmolLM2-360M`
- **Framework**: PyTorch (`torch`), Hugging Face `transformers`, `accelerate`
- **Dependency Management**: Poetry (`pyproject.toml`)
- **Python Version**: `>=3.12`
- **Hardware**: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)

## Key Files
- `main.py`: Inference test script running `SmolLM2-360M` in `bfloat16`. Includes domain-specific prompt tests related to particle Monte Carlo simulation for Deep Reactive Ion Etching (DRIE / Bosch process) and Ion Energy Distribution Functions (IEDFs).
- `pyproject.toml`: Poetry project definition and dependencies.

## Future Plans & Goals
- **Domain Fine-Tuning**:
  - Implement fine-tuning on domain-specific research papers (e.g., plasma physics, DRIE/Bosch process, Monte Carlo etch rate calculations).
  - Add training libraries (`peft`, `trl`, `datasets`).
  - Support LoRA/QLoRA adapter training and/or causal language modeling for domain adaptation.
  - Evaluate base vs. fine-tuned model outputs on DRIE simulation prompts.
