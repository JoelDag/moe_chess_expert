import logging
import os


def get_logger(name: str = "chess_expert"):
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger()


def log_adapter_architecture(model):
    """
    Log a concise summary of the active Chess Expert / MoE adapters inside ``model``.
    """
    try:
        from approaches.chess_expert.layer import ChessMLP
        from approaches.moelpr.layer import MLP
    except ImportError:
        log.warning("Adapter modules are unavailable; skipping adapter architecture summary.")
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info("Adapter model: %s trainable / %s total parameters", f"{trainable_params:,d}", f"{total_params:,d}")

    for name, module in model.named_modules():
        if isinstance(module, ChessMLP):
            adapter = getattr(module, "active_adapter", None)
            if isinstance(adapter, (list, tuple)):
                adapter = adapter[0]
            if adapter is None:
                adapter = getattr(module, "_active_adapter", "default")
            standard = module.standard_expert_count.get(adapter, module.num_experts.get(adapter, 0))
            chess_bias = module.chess_routing_bias.get(adapter, 0.0)
            chess_index = module.chess_expert_index.get(adapter, None)
            log.info(
                " - %s (ChessMLP): %d base experts + chess expert=%s, top_k=%s, routing_bias=%.3f",
                name,
                standard,
                chess_index if chess_index is not None else "disabled",
                module.topk,
                chess_bias,
            )
        elif isinstance(module, MLP):
            adapter = getattr(module, "active_adapter", None)
            if isinstance(adapter, (list, tuple)):
                adapter = adapter[0]
            if adapter is None:
                adapter = getattr(module, "_active_adapter", "default")
            log.info(
                " - %s (MoE MLP): %d experts, top_k=%s",
                name,
                module.num_experts.get(adapter, 0),
                module.topk,
            )
