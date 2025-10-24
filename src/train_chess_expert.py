import argparse
import os
import yaml
import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency for experiments
    wandb = None

from typing import Callable, Optional

from datasets import load_dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from approaches.registry import get_selected_method
from approaches.chess_expert.stockfish import StockfishConfig, StockfishEvaluator
from core.wrappers import AuxLossWrapper
from core.logger import get_logger
from evaluation import HarnessTrainer

log = get_logger("chess_expert_trainer")


def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_chess_expert_args(parser):
    """Adds arguments specific to the Chess Expert MoE-LPR method."""
    parser.add_argument("--chess_num_experts", type=int, default=2, help="Number of standard FFN experts (including base).")
    parser.add_argument("--chess_top_k", type=int, default=1, help="Number of experts selected per token.")
    parser.add_argument("--chess_aux_loss_coef", type=float, default=0.01, help="Load-balancing loss weight for stage 1.")
    parser.add_argument("--chess_lpr_loss_coef", type=float, default=None, help="LPR loss weight for stage 2.")
    parser.add_argument("--chess_layers_to_transform", type=str, default=None, help="Comma-separated transformer layer ids to convert.")
    parser.add_argument("--chess_stage", type=int, default=1, help="Training stage (1=balance, 2=LPR).")
    parser.add_argument("--chess_debug_mode", type=bool, default=False, help="Enable verbose routing logs.")
    parser.add_argument("--chess_include_expert", type=bool, default=True, help="Whether to append the chess expert.")
    parser.add_argument("--chess_feature_size", type=int, default=1, help="Context feature size passed to the chess expert.")
    parser.add_argument("--chess_hidden_dim", type=int, default=128, help="Hidden projection size inside the chess expert MLP.")
    parser.add_argument("--chess_dropout", type=float, default=0.1, help="Dropout probability applied in the chess expert.")
    parser.add_argument("--chess_routing_bias", type=float, default=0.0, help="Bias added to chess expert logit when mask_is_chess is true.")
    parser.add_argument("--chess_eval_field", type=str, default="chess_eval")
    parser.add_argument("--chess_mask_field", type=str, default="chess_mask")
    parser.add_argument("--chess_fen_field", type=str, default="fen")
    parser.add_argument("--disable_dummy_context", action="store_true")
    parser.add_argument("--context_dummy_prob", type=float, default=0.1)
    parser.add_argument("--context_dummy_std", type=float, default=0.5)
    parser.add_argument("--stockfish_path", type=str, default=None)
    parser.add_argument("--stockfish_depth", type=int, default=12)
    parser.add_argument("--stockfish_nodes", type=int, default=None)
    parser.add_argument("--stockfish_movetime", type=float, default=None)
    parser.add_argument("--stockfish_threads", type=int, default=1)
    parser.add_argument("--stockfish_hash", type=int, default=256)
    parser.add_argument("--stockfish_cp_clamp", type=float, default=1000.0)
    return parser


class ChessDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that prepares a batch for language modeling and injects chess context.

    The collator looks for per-example fields matching ``context_field`` and ``mask_field``; when
    present they should be sequences aligned with ``input_ids``.  If no explicit context is provided
    it can optionally call an evaluation provider (e.g., a Stockfish wrapper) with a FEN string.
    As a final fallback it can generate random dummy context for smoke testing.
    """

    def __init__(
        self,
        tokenizer,
        mlm: bool = False,
        chess_expert_index: Optional[int] = None,
        *,
        context_field: str = "chess_eval",
        mask_field: str = "chess_mask",
        fen_field: Optional[str] = "fen",
        eval_provider: Optional[Callable[[str], float]] = None,
        use_dummy_if_missing: bool = True,
        dummy_activation_prob: float = 0.1,
        dummy_eval_std: float = 0.5,
    ):
        super().__init__(tokenizer, mlm)
        self.chess_expert_index = chess_expert_index
        self.context_field = context_field
        self.mask_field = mask_field
        self.fen_field = fen_field
        self.eval_provider = eval_provider
        self.use_dummy_if_missing = use_dummy_if_missing
        self.dummy_activation_prob = dummy_activation_prob
        self.dummy_eval_std = dummy_eval_std
        self._warned_dummy_fallback = False

    def _maybe_log_dummy_warning(self):
        if not self._warned_dummy_fallback and self.use_dummy_if_missing:
            log.warning(
                "ChessDataCollator: no chess context found in batch examples; "
                "falling back to random dummy signals. Provide 'chess_eval'/'chess_mask' "
                "fields or an eval_provider to use real data."
            )
            self._warned_dummy_fallback = True

    def __call__(self, examples):
        preserved = []
        for example in examples:
            preserved.append({
                "eval": example.pop(self.context_field, None),
                "mask": example.pop(self.mask_field, None),
                "fen": example.pop(self.fen_field, None) if self.fen_field else None,
            })

        batch = super().__call__(examples)

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        chess_eval = torch.zeros(batch_size, seq_length, 1, dtype=torch.float32, device=device)
        chess_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)

        for idx, extra in enumerate(preserved):
            valid_length = seq_length
            if attention_mask is not None:
                valid_length = int(attention_mask[idx].sum().item())

            eval_values = extra.get("eval")
            mask_values = extra.get("mask")

            if eval_values is not None:
                eval_tensor = torch.as_tensor(eval_values, dtype=torch.float32, device=device).view(-1)
                num_tokens = min(valid_length, eval_tensor.numel())
                if num_tokens > 0:
                    chess_eval[idx, :num_tokens, 0] = eval_tensor[:num_tokens]
                    if mask_values is not None:
                        mask_tensor = torch.as_tensor(mask_values, dtype=torch.bool, device=device).view(-1)
                        chess_mask[idx, :num_tokens] = mask_tensor[:num_tokens]
                    else:
                        chess_mask[idx, :num_tokens] = True
                continue

            fen_payload = extra.get("fen") if self.fen_field else None
            if self.eval_provider is not None and fen_payload is not None:
                if isinstance(fen_payload, str):
                    try:
                        eval_value = float(self.eval_provider(fen_payload))
                    except Exception as exc:
                        log.warning(f"ChessDataCollator: evaluation provider failed for FEN {fen_payload}: {exc}")
                        eval_value = 0.0
                    chess_eval[idx, :valid_length, 0] = eval_value
                    chess_mask[idx, :valid_length] = True
                    continue
                if isinstance(fen_payload, (list, tuple)):
                    eval_buffer = []
                    for fen in fen_payload[:valid_length]:
                        try:
                            eval_buffer.append(float(self.eval_provider(fen)))
                        except Exception as exc:
                            log.warning(f"ChessDataCollator: evaluation provider failed for FEN {fen}: {exc}")
                            eval_buffer.append(0.0)
                    if eval_buffer:
                        eval_tensor = torch.tensor(eval_buffer, dtype=torch.float32, device=device)
                        num_tokens = min(valid_length, eval_tensor.numel())
                        chess_eval[idx, :num_tokens, 0] = eval_tensor[:num_tokens]
                        chess_mask[idx, :num_tokens] = True
                    continue

            if self.use_dummy_if_missing:
                self._maybe_log_dummy_warning()
                if valid_length > 0:
                    random_mask = torch.rand(valid_length, device=device) < self.dummy_activation_prob
                    if random_mask.any():
                        random_vals = torch.randn(random_mask.sum().item(), device=device) * self.dummy_eval_std
                        chess_eval[idx, :valid_length, 0][random_mask] = random_vals
                        chess_mask[idx, :valid_length][random_mask] = True

        flat_eval = chess_eval.view(-1, 1)
        flat_mask = chess_mask.view(-1)

        routing_override = torch.full((flat_mask.shape[0],), -1, dtype=torch.long, device=device)
        if self.chess_expert_index is not None:
            routed_tokens = int(flat_mask.sum().item())
            if routed_tokens > 0:
                routing_override[flat_mask] = self.chess_expert_index
                log.debug(
                    "ChessDataCollator: forcing chess expert index %s for %d tokens",
                    self.chess_expert_index,
                    routed_tokens,
                )

        batch["context"] = {
            "chess_eval": flat_eval,
            "mask_is_chess": flat_mask,
        }
        batch["routing_override"] = routing_override
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--use_method", default="chess_expert", help="The MoE method to use.")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q_proj,v_proj")

    # Add chess expert specific arguments
    parser = get_chess_expert_args(parser)

    # General training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False)
    parser.add_argument("--logging_first_step", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="moe-study-chess-expert")

    args = parser.parse_args()

    # Load YAML config and merge with CLI args
    cfg = load_yaml_config(args.config)
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    
    # Convert cfg dictionary back to argparse Namespace for compatibility
    args = argparse.Namespace(**cfg)

    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}
    if wandb is not None and not wandb_disabled:
        wandb.init(project=args.wandb_project, config=args)
    else:
        log.info("Skipping wandb.init (WANDB disabled or package unavailable).")

    # Load model and tokenizer
    adapter = get_selected_method(args.use_method)
    model, tokenizer, _ = adapter.apply(
        {"model_name_or_path": args.model_name_or_path},
        {},
        args,
    )
    model.to("cpu")

    # Wrap model for auxiliary loss calculation
    aux_coef = args.chess_aux_loss_coef if args.chess_aux_loss_coef is not None else 0.0
    model = AuxLossWrapper(model, coef=aux_coef)

    # Load dataset (supports HF hub names or local JSON/JSONL files)
    data_source = args.dataset
    if isinstance(data_source, str) and data_source.endswith((".jsonl", ".json")):
        raw_dataset = load_dataset("json", data_files={"train": data_source})["train"]
    else:
        raw_dataset = load_dataset(data_source, split="train")
    dataset = raw_dataset.train_test_split(test_size=0.1)

    fen_field = args.chess_fen_field
    if isinstance(fen_field, str) and fen_field.lower() in {"", "none"}:
        fen_field = None

    context_fields = {args.chess_eval_field, args.chess_mask_field}
    if fen_field:
        context_fields.add(fen_field)
    context_fields = {field for field in context_fields if field}

    if args.max_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))

    if "input_ids" not in dataset["train"].column_names:
        def tokenize_fn(examples):
            tokenized = tokenizer(examples["text"], truncation=True, max_length=args.cutoff_len)
            for field in context_fields:
                if field in examples:
                    tokenized[field] = examples[field]
            return tokenized

        columns_to_remove = [
            col for col in dataset["train"].column_names if col not in context_fields.union({"text"})
        ]
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)
    else:
        if "text" in dataset["train"].column_names:
            dataset = dataset.remove_columns(["text"])

    extraneous_cols = {"text", "fen", "san", "cp", "move_index"}
    drop_cols = [col for col in dataset["train"].column_names if col in extraneous_cols]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)

    stockfish_evaluator: Optional[StockfishEvaluator] = None
    eval_provider: Optional[Callable[[str], float]] = None
    stockfish_path = getattr(args, "stockfish_path", None)
    if stockfish_path:
        stockfish_cfg = StockfishConfig(
            path=stockfish_path,
            depth=getattr(args, "stockfish_depth", 12),
            nodes=getattr(args, "stockfish_nodes", None),
            movetime=getattr(args, "stockfish_movetime", None),
            threads=getattr(args, "stockfish_threads", 1),
            hash_size=getattr(args, "stockfish_hash", 256),
            cp_clamp=getattr(args, "stockfish_cp_clamp", 1000.0),
        )
        try:
            stockfish_evaluator = StockfishEvaluator(stockfish_cfg)
            eval_provider = stockfish_evaluator
            log.info("Initialized Stockfish evaluator from %s", stockfish_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Stockfish engine at {stockfish_path}: {exc}") from exc

    report_to = "wandb" if wandb is not None and not wandb_disabled else "none"

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=float(args.learning_rate),
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=getattr(args, "fp16", False),
        bf16=getattr(args, "bf16", False),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        logging_first_step=getattr(args, "logging_first_step", False),
        report_to=report_to,
        remove_unused_columns=False,
    )

    # Determine the chess expert index for the data collator (last expert when included).
    chess_expert_idx = args.chess_num_experts if args.chess_include_expert else None

    data_collator = ChessDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        chess_expert_index=chess_expert_idx,
        context_field=args.chess_eval_field,
        mask_field=args.chess_mask_field,
        fen_field=fen_field,
        eval_provider=eval_provider,
        use_dummy_if_missing=not args.disable_dummy_context,
        dummy_activation_prob=args.context_dummy_prob,
        dummy_eval_std=args.context_dummy_std,
    )

    trainer = HarnessTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        do_harness_eval=False, # Disable for this test
    )

    log.info("Starting training with ChessExpert test script...")
    trainer.train()
    log.info("Training finished.")

    model.model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    model.model.config.save_pretrained(args.output_dir)

    if stockfish_evaluator is not None:
        stockfish_evaluator.close()


if __name__ == "__main__":
    main()
