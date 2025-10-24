# Chess MoE Migration Status

## Current Implementation
- Chess expert now uses a MOE-LPR style layer (`ChessMLP`) that clones the base FFN experts and appends a dedicated chess expert network.
- Context/routing override plumbing flows through `ChessExpertModel`, enabling per-token chess signals during forward passes.
- Training script `src/train_chess_expert.py` supports local datasets, optional Stockfish evaluation, and CPU-only runs with wandb disabled.
- Sample config `configs/train_chess_expert_test.yaml` points to the public TinyLlama checkpoint and a dummy JSONL dataset for smoke tests.
- `data_prep/chess/build_stockfish_dataset.py` provides a utility to generate Stockfish-labelled datasets compatible with the new collator.
- Script now accepts a missing `--engine-path`; when omitted, synthetic centipawn values are generated, allowing fully offline dataset creation (see `data/chess_stockfish.jsonl`).

## Recent Validation
- Smoke training command: `python src/train_chess_expert.py --config configs/train_chess_expert_test.yaml` (use `WANDB_DISABLED=true` and a local dataset/model). Current config limits to `max_steps=1` to avoid CPU timeouts.
- Generate offline chess-position data (no Stockfish needed): `python data_prep/chess/build_stockfish_dataset.py --pgn data/chess_sample.pgn --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output data/chess_stockfish.jsonl --max-positions 50 --pad-left`
- Verified a forward pass on `data/chess_stockfish.jsonl` with the ChessDataCollator (see manual snippet in session history) – the loss and aux loss are finite, confirming that the chess expert is selected with real signals.
- Full `trainer.train()` on CPU currently times out; reducing `max_steps` works for small validation but saving checkpoints still hits the safetensors shared-weight warning. Use `model.model.save_pretrained(..., safe_serialization=False)` if you need to persist weights midrun.

## Quick-Start Checklist for Next Session
1. (Optional) Regenerate or extend the chess dataset:
   ```bash
   python data_prep/chess/build_stockfish_dataset.py \
     --pgn <new_games.pgn> \
     --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --output data/chess_stockfish.jsonl \
     --max-positions 200 --pad-left
   ```
2. Run a manual forward sanity check (uses the collator directly): see `docs/snippets/chess_forward_check.py` once added, or reuse the inline snippet from 2024-10-15 session.
3. For a 1-step training smoke test:
   ```bash
   HF_DATASETS_CACHE=./.cache/hf_datasets \
   WANDB_DISABLED=true WANDB_MODE=offline \
   python src/train_chess_expert.py --config configs/train_chess_expert_test.yaml
   ```
   (Expect completion after a single optimizer step; checkpoint saving remains disabled.)
4. Investigate safetensors save behaviour before enabling multi-step runs (`save_model(..., safe_serialization=False)` is the workaround).
- Saving checkpoints via safetensors requires caution: cloned FFN weights and chess expert modules share memory; use `model.save_pretrained(..., safe_serialization=False)` if necessary.

## Remaining Work
1. Generate a real Stockfish-labelled dataset and verify routing behaviour with genuine chess context.
2. Replace the chess expert MLP with an NNUE-backed module once bindings are available.
3. Add unit tests covering routing overrides, chess logit biasing, and context shape handling.
4. Investigate safetensors serialization for shared-weight experts or enforce full copies during expert construction.

## Next Steps Toward Full Training
1. **Curate Rich Chess Data:** Scale up `data/chess_stockfish.jsonl` (thousands of positions), ideally with true Stockfish or NNUE evaluations and optional policy hints; ensure the dataset mixes well with natural-language corpora.
2. **Stabilize Checkpointing:** Decide on serialization (`safe_serialization=False` or duplicate expert weights) so multi-step runs can save/restart reliably.
3. **Router Analytics & Tests:** Log chess expert activation frequency, add unit tests for overrides/bias, and monitor aux loss to prevent router collapse.
4. **Training Curriculum:** Design a schedule that interleaves general language batches with chess batches, tune `chess_routing_bias`, and validate losses on both domains.
5. **Evaluation Harness:** Establish baseline metrics (language perplexity, Stockfish-eval error, move prediction accuracy) for consistent post-training analysis.

## How Far From Dual-Domain Training?
- **Infrastructure:** Context plumbing, MoE layer, and dataset ingestion are ready for mixed-domain finetunes.
- **Outstanding Pieces:** Need large-scale chess data, checkpoint solution, and monitoring/tests before investing in long runs.
- **Expectation:** Once the above steps land, you can launch natural-language + chess MoE training with minor config changes; current blockers are data quality/quantity and reliability tooling rather than core architecture.

## Notes
- Large local assets (`TinyLlama/`, `.cache/`, `data/`) are git-ignored.
- To re-run smoke tests, ensure the local model path in the config is valid and that the desired dataset exists on disk.
