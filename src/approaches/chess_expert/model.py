from typing import Any

import torch

from approaches.moelpr.model import MoeModel
from .config import ChessMoeConfig
from .layer import ChessMLP, load_balancing_loss_func, lpr_loss_func


class ChessExpertModel(MoeModel):
    """
    MoE-LPR model wrapper that installs a specialised chess expert per transformer block.
    """

    def __init__(self, model, config: ChessMoeConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_new_module(self, moe_config: ChessMoeConfig, adapter_name: str, target, **kwargs: Any):
        return ChessMLP(
            target,
            adapter_name,
            num_experts=moe_config.num_experts,
            init_moe_weights=moe_config.init_moe_weights,
            topk=moe_config.topk,
            aux_loss_coef=moe_config.aux_loss_coef,
            lpr_loss_coef=moe_config.lpr_loss_coef,
            moelpr_debug_mode=moe_config.moelpr_debug_mode,
            include_chess_expert=moe_config.include_chess_expert,
            chess_feature_size=moe_config.chess_feature_size,
            chess_hidden_dim=moe_config.chess_hidden_dim,
            chess_dropout=moe_config.chess_dropout,
            chess_routing_bias=moe_config.chess_routing_bias,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        context = kwargs.pop("context", None)
        routing_override = kwargs.pop("routing_override", None)

        lang_mask = kwargs.get("lang_mask", None)
        attention_mask = kwargs.get("attention_mask", None)

        for module in self.model.modules():
            if isinstance(module, ChessMLP):
                module.lang_mask = lang_mask
                module.attention_mask = attention_mask
                adapter = module.active_adapter[0]
                device = module.moe_router_embedding[adapter].weight.device
                module.set_context(context, routing_override, device=device)

        try:
            return self.model(*args, **kwargs)
        finally:
            for module in self.model.modules():
                if isinstance(module, ChessMLP):
                    module.clear_context()

    def get_aux_loss(self):
        device = next(self.model.parameters()).device
        total_aux = torch.tensor(0.0, device=device)

        active_adapter = self.active_adapter
        if isinstance(active_adapter, list):
            active_adapter = active_adapter[0]
        config: ChessMoeConfig = self.peft_config[active_adapter]

        if not self.training:
            return total_aux

        router_logits_per_layer = []
        attention_mask = None
        lang_mask = None

        for module in self.model.modules():
            if isinstance(module, ChessMLP) and module.latest_router_logits is not None:
                router_logits_per_layer.append(module.latest_router_logits)
                if attention_mask is None:
                    attention_mask = module.attention_mask
                if lang_mask is None:
                    lang_mask = module.lang_mask

        if not router_logits_per_layer:
            return total_aux

        total_experts = config.num_experts + (1 if config.include_chess_expert else 0)

        if config.stage == 1 and config.aux_loss_coef is not None:
            balance_terms = []
            for logits in router_logits_per_layer:
                balance = load_balancing_loss_func(
                    (logits,),
                    total_experts,
                    config.topk,
                    attention_mask,
                )
                balance_terms.append(torch.as_tensor(balance, device=device, dtype=torch.float32))
            balance_loss = torch.mean(torch.stack(balance_terms))
            total_aux = total_aux + config.aux_loss_coef * balance_loss

        elif config.stage == 2 and config.lpr_loss_coef is not None:
            if lang_mask is None:
                return total_aux
            lpr = lpr_loss_func(router_logits_per_layer, lang_mask=lang_mask)
            lpr = torch.as_tensor(lpr, device=device, dtype=torch.float32)
            total_aux = total_aux + config.lpr_loss_coef * lpr

        return total_aux
