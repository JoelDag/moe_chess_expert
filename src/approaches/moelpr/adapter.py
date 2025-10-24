import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.generic_adapter import MethodAdapter
from approaches.moelpr.model import MoeModel
from approaches.moelpr.config import MoeConfig


class MoelprAdapter(MethodAdapter):
    """Adapter wrapper for MoE-LPR"""

    name = "moelpr"

    def apply(self, model_args, data_args, training_args):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args["model_name_or_path"],
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args["model_name_or_path"],
            padding_side="left",
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        if getattr(training_args, "moelpr_layers_to_transform", None):
            try:
                layers = [int(x) for x in str(training_args.moelpr_layers_to_transform).split(",") if x != ""]
            except ValueError:
                layers = []
        else:
            layers = []

        moelpr_config = MoeConfig(
            num_experts=training_args.moelpr_num_experts,
            topk=training_args.moelpr_top_k,
            aux_loss_coef=training_args.moelpr_aux_loss_coef,
            lpr_loss_coef=training_args.moelpr_lpr_loss_coef,
            layers_to_transform=layers,
            stage=getattr(training_args, "moelpr_stage", 1),
            moelpr_debug_mode=getattr(training_args, "moelpr_debug_mode", False)
        )
        model = MoeModel(base_model, moelpr_config, adapter_name="default")

        # Ensure AuxLossWrapper can call a getter without error
        if not hasattr(model, "get_aux_loss"):
            def _zero_aux():
                return torch.tensor(0.0, device=next(model.parameters()).device)
            model.get_aux_loss = _zero_aux

        return model, tokenizer, {}

    def aux_loss(self, model):
        if hasattr(model, "get_aux_loss") and callable(model.get_aux_loss):
            return model.get_aux_loss()
        warnings.warn("Model has no get_aux_loss, using 0.0 tensor.")
        return torch.tensor(0.0, device=next(model.parameters()).device)
