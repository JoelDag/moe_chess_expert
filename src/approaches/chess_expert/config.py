from dataclasses import dataclass, field

from approaches.moelpr.config import MoeConfig


@dataclass
class ChessMoeConfig(MoeConfig):
    """
    MoE-LPR configuration extended with chess-expert specific knobs.
    """

    include_chess_expert: bool = field(
        default=True,
        metadata={"help": "If true, append a specialised chess expert to each transformed FFN."},
    )
    chess_feature_size: int = field(
        default=1,
        metadata={
            "help": "Dimensionality of the chess feature vector provided in the context (e.g. centipawn eval, NNUE)."
        },
    )
    chess_hidden_dim: int = field(
        default=128,
        metadata={"help": "Hidden width of the chess expert projection MLP."},
    )
    chess_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout applied inside the chess expert MLP."},
    )
    chess_routing_bias: float = field(
        default=0.0,
        metadata={
            "help": "Optional additive bias applied to the chess expert router logit when mask_is_chess is true."
        },
    )
