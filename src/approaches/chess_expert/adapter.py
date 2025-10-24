import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from approaches.chess_expert.config import ChessMoeConfig
from approaches.chess_expert.model import ChessExpertModel
from core.generic_adapter import MethodAdapter


class ChessExpertAdapter(MethodAdapter):
    """Adapter wrapper for the Chess Expert MoE-LPR method."""

    name = "chess_expert"

    def apply(self, model_args, data_args, training_args):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args["model_name_or_path"],
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args["model_name_or_path"],
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        layers_to_transform = []
        raw_layers = getattr(training_args, "chess_layers_to_transform", None)
        if raw_layers:
            try:
                layers_to_transform = [int(layer.strip()) for layer in str(raw_layers).split(",") if layer.strip() != ""]
            except ValueError:
                layers_to_transform = []

        config = ChessMoeConfig(
            num_experts=getattr(training_args, "chess_num_experts", 2),
            topk=getattr(training_args, "chess_top_k", 1),
            aux_loss_coef=getattr(training_args, "chess_aux_loss_coef", None),
            lpr_loss_coef=getattr(training_args, "chess_lpr_loss_coef", None),
            layers_to_transform=layers_to_transform,
            stage=getattr(training_args, "chess_stage", 1),
            moelpr_debug_mode=getattr(training_args, "chess_debug_mode", False),
            include_chess_expert=getattr(training_args, "chess_include_expert", True),
            chess_feature_size=getattr(training_args, "chess_feature_size", 1),
            chess_hidden_dim=getattr(training_args, "chess_hidden_dim", 128),
            chess_dropout=getattr(training_args, "chess_dropout", 0.1),
            chess_routing_bias=getattr(training_args, "chess_routing_bias", 0.0),
        )

        model = ChessExpertModel(base_model, config, adapter_name="default")

        return model, tokenizer, {}

    def aux_loss(self, model):
        if hasattr(model, "get_aux_loss") and callable(model.get_aux_loss):
            return model.get_aux_loss()
        warnings.warn("Model has no get_aux_loss method, returning 0.0 tensor.")
        return torch.tensor(0.0, device=next(model.parameters()).device)
