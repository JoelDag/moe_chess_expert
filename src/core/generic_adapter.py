import torch.nn as nn

from typing import Tuple
from dataclasses import dataclass
from typing import Iterable
from transformers import PreTrainedModel, PreTrainedTokenizerBase

@dataclass
class AdapterConfig:
    """
    Generic adapter configuration which holds the tunable hyperparameters of each approach
    """
    r: int
    alpha: int
    dropout: float
    target_modules: Iterable[str]

class MethodAdapter:
    """Interface that each adapter method must implement."""
    name: str = "BASE"

    def apply(
        self,
        model_args,
        data_args,
        training_args,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Apply the adapter method to the base model and return the adapted model and tokenizer."""
        raise NotImplementedError

    def aux_loss(self, model: nn.Module):
        """Return method-specific auxiliary loss if any."""
        raise NotImplementedError
