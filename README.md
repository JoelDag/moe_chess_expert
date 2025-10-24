# Chess Expert MoE (Standalone)

Chess Expert MoE is a focused spin-out of the adapter work that combines mixture-of-experts routing with chess-aware features.  It can be used to fine-tune LLaMA-style causal language models so that a dedicated “chess expert” branch is available during training and inference.

This standalone repository contains everything needed to:

- Generate chess datasets with Stockfish or synthetic evaluations.
- Apply the Chess Expert adapter to a base model.
- Run lightweight smoke tests and lm-evaluation-harness checks.

The layout mirrors the original research prototype but removes cross-repo dependencies so you can showcase the project, extend it, or run bespoke experiments.

## Features

- **MoE-LPR adapter** built on top of PEFT’s Mixture-of-Experts tuner with stage-specific auxiliary losses.
- **Chess expert FFN** that fuses language representations with centipawn / NNUE-style signals.
- **Context-aware collator** that injects routing overrides, Stockfish evaluations, and dummy fallbacks.
- **Harness-integrated trainer** with optional lm-evaluation-harness logging.
- **Dataset tooling** for extracting positions from PGN files and labelling them via Stockfish.

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```
   (Use `pip install -r requirements.txt` if you prefer not to install the package.)

2. **Set the source path (if not installing as a package)**
   ```bash
   export PYTHONPATH="$(pwd)/src"
   ```

3. **Run the smoke-test training loop**
   ```bash
   WANDB_DISABLED=true \
   python src/train_chess_expert.py --config configs/train_chess_expert_test.yaml
   ```
   The default config performs a single optimizer step against the bundled `data/chess_stockfish.jsonl` dataset and uses the public `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint.

## Preparing Chess Data

Use the dataset builder when you need fresh Stockfish labels:

```bash
python data_prep/chess/build_stockfish_dataset.py \
  --pgn data/chess_sample.pgn \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output data/chess_stockfish.jsonl \
  --max-positions 50
```

- Supplying `--engine-path` enables real Stockfish evaluations.
- Omitting it falls back to random centipawn values for offline testing.

## Project Layout

- `configs/` – ready-to-run experiment configs (TinyLlama smoke test included).
- `data/` – sample PGN and JSONL used by quick-start scripts.
- `data_prep/` – dataset generation utilities with Stockfish integration.
- `docs/` – status notes and future work trackers.
- `src/approaches/` – MoE-LPR base implementation and chess expert extension.
- `src/core/` – shared abstractions (logging, adapter plumbing, aux-loss wrapper).
- `src/train_chess_expert.py` – primary training entry point.
- `src/evaluation.py` – Trainer subclass that can trigger lm-evaluation-harness.

## Next Steps

- Plug in a GPU-backed environment and increase `max_steps` for longer runs.
- Enable lm-eval tasks via the `HarnessTrainer` and a task list in your config.
- Replace synthetic chess signals with NNUE-derived features once available.

> **Tip:** keep `WANDB_DISABLED=true` for offline experiments; remove it to log to Weights & Biases when credentials are configured.
