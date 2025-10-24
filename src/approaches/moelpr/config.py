# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

if not hasattr(PeftType, "MOE"):
    PeftType.MOE = "MOE"


@dataclass
class MoeConfig(PeftConfig):
    target_modules: Optional[Union[List[str], str]] = field(default=None)
    init_moe_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Moe layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    num_experts: int = field(
        default=2,
        metadata={
            "help": (
                "The total number of experts for moe fine-tuning. If set to N, then N-1 new experts are added."
            ),
        },
    )
    topk: int = field(
        default=1,
        metadata={
            "help": (
                "How much experts are selected for each token."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "Upcycling to MoE layer for which layers."
        },
    )
    save_all_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Updates and save all the parameters of MoE."
            ),
        },
    )
    aux_loss_coef: float = field(
        default=None,
        metadata={
            "help": (
                "The weight of the load balancing loss. Only will be used if set."
            ),
        },
    )
    lpr_loss_coef: float = field(
        default=None,
        metadata={
            "help": (
                "The weight of the lpr loss. Only will be used if set."
            ),
        },
    )
    stage: int = field(
        default=1,
        metadata={
            "help": "Training stage for MoE-LPR. Stage 1 is for training new experts, Stage 2 is for router review."
        }
    )
    moelpr_debug_mode: bool = field(default=False, metadata={"help": "Enable debug mode for MoE-LPR."})

    def __post_init__(self):
        self.peft_type = PeftType.MOE