from approaches.chess_expert.adapter import ChessExpertAdapter

_REGISTRY = {
    "chess_expert": ChessExpertAdapter(),
}


def get_selected_method(name: str):
    """Return the registered adapter matching ``name`` if available."""
    return _REGISTRY.get(name.lower())


__all__ = ["get_selected_method", "ChessExpertAdapter"]
