# Chess Expert MoE – Migration Plan

This document tracks the transition from the original LoRA-based “Chess Expert” prototype to a proper heterogeneous Mixture-of-Experts built on the MOE-LPR codepath. The new design keeps full-capacity FFNs for standard experts, then adds a specialised chess module that can evolve into an embedded NNUE.

---

## Why Abandon the LoRA Adapter Path?
- **Architectural freedom:** LoRA layers force experts to look like rank-updated FFNs. Copying whole FFNs (as MOE-LPR already does) lets the chess expert be any network that maps chess signals → hidden dimension.
- **Shared routing semantics:** Reusing MOE-LPR’s router logic means auxiliary losses, logging, and gating behaviour stay consistent with the existing MoE stack.
- **Future NNUE drop-in:** Once NNUE bindings exist, the chess expert slot can directly invoke them without worrying about base FFN structure.

---

## Implementation Plan

1. **Fork MOE-LPR Layer**
   - Clone `moelpr.layer.MLP` into `chess_expert.layer` (e.g., `ChessMoELprLayer`).
   - Maintain full FFN cloning for standard experts; reserve one slot for the chess specialist.

2. **Chess Expert Module**
   - Implement `ChessExpertFFN` that:
     - Consumes hidden states plus chess context (`chess_eval`, future NNUE features).
     - Projects chess signals via a small MLP, optionally merges with hidden states.
     - Returns `hidden_dim` vectors to satisfy the MoE contract.

3. **Context Plumbing**
   - Thread `context` / `routing_override` through the new layer (mirroring the adapter prototype).
   - Add optional routing bias that boosts the chess expert logit when `mask_is_chess` is true.

4. **Config & Adapter**
   - Create `ChessMoELprConfig` (extend existing MOE-LPR config with chess flags).
   - Add `ChessMoELprAdapter` that swaps the FFN blocks for the new layer in the base model.
   - Register the adapter in `approaches.registry`.

5. **Data & Collator**
   - Reuse the Stockfish dataset tooling (`data_prep/chess/build_stockfish_dataset.py`).
   - Update the collator to emit context compatible with the new layer (same API as prototypes).

6. **Validation**
   - Implement unit tests for routing override, chess expert gating bias, and context shapes.
   - Run the smoke-training script against the JSONL Stockfish dataset and ensure no regressions in aux loss or training stability.

---

## Migration Checklist

- [ ] Remove LoRA-specific chess modules once the MOE-LPR variant is functional.
- [ ] Update documentation and configs to point at the new adapter (`use_method: chess_moelpr` or similar).
- [ ] Deprecate dummy-context paths after real chess data is verified.
- [ ] Benchmark routing distribution before/after migration to confirm the chess expert fires as expected.
- [ ] Plan NNUE integration after the heterogeneous expert proves stable (dedicated module + bindings).

---

**Short-term goal:** land steps 1–4 so that a chess-aware expert can train end-to-end using precomputed Stockfish signals.  
**Mid-term goal:** replace the chess expert’s MLP with a tiny NNUE and evaluate routing behaviour without tool calls.
