from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from approaches.moelpr.layer import (
    MoeLayer,
    lpr_loss_func,
    load_balancing_loss_func,
)
from core.logger import get_logger

log = get_logger("chess_moelpr")


class ChessExpertFFN(nn.Module):
    """
    Lightweight projection network that maps chess-specific features into the model hidden space.
    """

    def __init__(self, hidden_dim: int, feature_dim: int, hidden_proj: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        inner_dim = max(hidden_proj, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor, chess_features: torch.Tensor) -> torch.Tensor:
        if chess_features.dim() == 1:
            chess_features = chess_features.unsqueeze(-1)
        if chess_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Chess feature dimension mismatch: expected {self.feature_dim}, got {chess_features.shape[-1]}"
            )
        fused = torch.cat([hidden_states, chess_features], dim=-1)
        return self.mlp(fused)


class ChessMoeLayer(MoeLayer):
    """
    Extension of the base MoeLayer that installs an additional chess expert and stores per-forward context.
    """

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.chess_expert_index = {}
        self.standard_expert_count = {}
        self.include_chess = {}
        self.chess_feature_size = {}
        self.chess_hidden_dim = {}
        self.chess_dropout = {}
        self.chess_routing_bias = {}

        # Context state (set/reset each forward)
        self.context_eval: Optional[torch.Tensor] = None
        self.context_mask: Optional[torch.Tensor] = None
        self.routing_override: Optional[torch.Tensor] = None

    def update_layer(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        *,
        num_experts: int,
        init_moe_weights: bool,
        include_chess_expert: bool,
        chess_feature_size: int,
        chess_hidden_dim: int,
        chess_dropout: float,
        chess_routing_bias: float,
    ) -> None:
        total_experts = num_experts + (1 if include_chess_expert else 0)
        super().update_layer(base_layer, adapter_name, total_experts, init_moe_weights)

        self.standard_expert_count[adapter_name] = num_experts
        self.include_chess[adapter_name] = include_chess_expert
        self.chess_feature_size[adapter_name] = chess_feature_size
        self.chess_hidden_dim[adapter_name] = chess_hidden_dim
        self.chess_dropout[adapter_name] = chess_dropout
        self.chess_routing_bias[adapter_name] = chess_routing_bias

        if include_chess_expert:
            chess_module = ChessExpertFFN(
                hidden_dim=self.in_features,
                feature_dim=chess_feature_size,
                hidden_proj=chess_hidden_dim,
                dropout=chess_dropout,
            )
            # Replace the last cloned expert with the chess expert
            self.moe_experts[adapter_name][-1] = chess_module
            self.chess_expert_index[adapter_name] = total_experts - 1

            # Align device/dtype with base layer weights
            weight_ref = self._get_weight_reference(base_layer)
            chess_module.to(device=weight_ref.device, dtype=weight_ref.dtype)
        else:
            self.chess_expert_index[adapter_name] = None

        # Ensure router embedding matches dtype/device of base weights
        self._move_modules_to_base_device(base_layer, adapter_name)
        self.set_adapter(self.active_adapters)

    def _get_weight_reference(self, base_layer: nn.Module) -> torch.Tensor:
        if hasattr(base_layer, "gate_proj"):
            return base_layer.gate_proj.weight
        if hasattr(base_layer, "fc1"):
            return base_layer.fc1.weight
        raise ValueError("Unsupported base layer structure for ChessMoeLayer.")

    def _move_modules_to_base_device(self, base_layer: nn.Module, adapter_name: str) -> None:
        weight_ref = self._get_weight_reference(base_layer)
        modules_to_move = [
            self.moe_router_embedding[adapter_name],
            self.moe_experts[adapter_name],
        ]
        for module in modules_to_move:
            if module is None:
                continue
            if isinstance(module, nn.ModuleList):
                for sub in module:
                    sub.to(device=weight_ref.device, dtype=weight_ref.dtype)
            else:
                module.to(device=weight_ref.device, dtype=weight_ref.dtype)

    def set_context(
        self,
        context: Optional[dict],
        routing_override: Optional[torch.Tensor],
        *,
        device: Optional[torch.device],
    ) -> None:
        self.context_eval = None
        self.context_mask = None
        self.routing_override = None

        if routing_override is not None:
            if not isinstance(routing_override, torch.Tensor):
                routing_override = torch.tensor(routing_override, dtype=torch.long, device=device)
            else:
                routing_override = routing_override.to(device=device, dtype=torch.long)
            self.routing_override = routing_override.view(-1)

        if not context:
            return

        eval_tensor = context.get("chess_eval")
        mask_tensor = context.get("mask_is_chess")

        if eval_tensor is not None:
            if not isinstance(eval_tensor, torch.Tensor):
                eval_tensor = torch.tensor(eval_tensor, dtype=torch.float32, device=device)
            else:
                eval_tensor = eval_tensor.to(device=device, dtype=torch.float32)
            if eval_tensor.dim() == 1:
                eval_tensor = eval_tensor.unsqueeze(-1)
            self.context_eval = eval_tensor.view(-1, eval_tensor.shape[-1])

        if mask_tensor is not None:
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor, dtype=torch.bool, device=device)
            else:
                mask_tensor = mask_tensor.to(device=device, dtype=torch.bool)
            self.context_mask = mask_tensor.view(-1)

        if self.context_eval is not None and self.context_mask is not None:
            min_len = min(self.context_eval.shape[0], self.context_mask.shape[0])
            self.context_eval = self.context_eval[:min_len]
            self.context_mask = self.context_mask[:min_len]
        elif self.context_eval is not None and self.context_mask is None:
            self.context_mask = torch.zeros(
                self.context_eval.shape[0], dtype=torch.bool, device=self.context_eval.device
            )
        elif self.context_eval is None and self.context_mask is not None:
            self.context_eval = torch.zeros(
                self.context_mask.shape[0], 1, dtype=torch.float32, device=self.context_mask.device
            )

    def clear_context(self) -> None:
        self.context_eval = None
        self.context_mask = None
        self.routing_override = None

    def _gather_chess_features(
        self,
        adapter_name: str,
        indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feature_dim = self.chess_feature_size.get(adapter_name, 1)
        if self.context_eval is None or indices.numel() == 0:
            return torch.zeros((indices.numel(), feature_dim), device=device, dtype=dtype)

        if indices.max().item() >= self.context_eval.shape[0]:
            max_valid = self.context_eval.shape[0] - 1
            indices = indices.clamp_max(max_valid)

        gathered = self.context_eval[indices].to(device=device, dtype=dtype)
        if gathered.shape[-1] != feature_dim:
            if gathered.shape[-1] > feature_dim:
                gathered = gathered[:, :feature_dim]
            else:
                pad_width = feature_dim - gathered.shape[-1]
                gathered = torch.nn.functional.pad(gathered, (0, pad_width))

        if self.context_mask is not None:
            mask = self.context_mask[indices].to(device=device)
            gathered = gathered * mask.unsqueeze(-1).to(dtype=dtype)
        return gathered


class ChessMLP(nn.Module, ChessMoeLayer):
    """
    Chess-aware variant of the MoE-LPR MLP layer. Incorporates an additional chess expert and consumes routing context.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        *,
        num_experts: int,
        init_moe_weights: bool,
        topk: int,
        aux_loss_coef: Optional[float],
        lpr_loss_coef: Optional[float],
        moelpr_debug_mode: bool,
        include_chess_expert: bool,
        chess_feature_size: int,
        chess_hidden_dim: int,
        chess_dropout: float,
        chess_routing_bias: float,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        ChessMoeLayer.__init__(self, base_layer)

        self.aux_loss_coef = aux_loss_coef
        self.topk = topk
        self.lpr_loss_coef = lpr_loss_coef
        self.debug_mode = moelpr_debug_mode
        self._active_adapter = adapter_name

        self.lang_mask = None
        self.attention_mask = None
        self.latest_router_logits = None

        self.update_layer(
            base_layer,
            adapter_name,
            num_experts=num_experts,
            init_moe_weights=init_moe_weights,
            include_chess_expert=include_chess_expert,
            chess_feature_size=chess_feature_size,
            chess_hidden_dim=chess_hidden_dim,
            chess_dropout=chess_dropout,
            chess_routing_bias=chess_routing_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        adapter = self.active_adapter[0]
        router = self.moe_router_embedding[adapter]

        result, router_logits = self.topk_route(x, router, adapter)
        self.latest_router_logits = router_logits
        return result.to(previous_dtype)

    def topk_route(self, hidden_states, router, adapter=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = router(hidden_states)

        total_experts = self.num_experts[adapter]
        chess_index = self.chess_expert_index.get(adapter, None)

        if (
            chess_index is not None
            and self.context_mask is not None
            and self.chess_routing_bias.get(adapter, 0.0) != 0.0
        ):
            router_logits[:, chess_index] = router_logits[:, chess_index] + (
                self.chess_routing_bias[adapter] * self.context_mask.float()
            )

        if self.routing_override is not None and self.routing_override.numel() == router_logits.shape[0]:
            forced_mask = self.routing_override >= 0
            if forced_mask.any():
                override_targets = self.routing_override[forced_mask].clamp(0, total_experts - 1)
                router_logits[forced_mask] = -1e9
                router_logits[forced_mask, override_targets] = 1e9

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        if self.topk != 1:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        if self.debug_mode:
            num_inputs = router_logits.shape[0]
            expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=total_experts)
            log.info(f"Expert counts: {expert_counts.tolist()}")

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, total_experts).permute(2, 1, 0)
        experts = [self.base_layer] + [module for module in self.moe_experts[adapter]]

        for expert_idx in range(total_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            if chess_index is not None and expert_idx == chess_index:
                chess_module = experts[expert_idx]
                if not isinstance(chess_module, ChessExpertFFN):
                    continue
                chess_features = self._gather_chess_features(
                    adapter, top_x, current_state.device, current_state.dtype
                )
                current_hidden_states = chess_module(current_state, chess_features)
            else:
                expert_layer = experts[expert_idx]
                current_hidden_states = expert_layer(current_state)

            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


__all__ = [
    "ChessExpertFFN",
    "ChessMoeLayer",
    "ChessMLP",
    "lpr_loss_func",
    "load_balancing_loss_func",
]
