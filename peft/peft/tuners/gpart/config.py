# Copyright 2024-present the HuggingFace Inc. team.
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
from typing import Dict, List, Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class GPartConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GPartModel`].

    Paper: Global Partition Fine-Tuning (GPart)

    Args:
        d (`int`):
            The number of groups for parameter partitioning. This is the key hyperparameter that controls the
            parameter budget - the number of trainable parameters will be exactly d.
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        gpart_dropout (`float`):
            The dropout probability for GPart layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for GPart. Can be 'none', 'all' or 'gpart_only'. If 'all' or 'gpart_only', the corresponding
            biases will be updated during training. Be aware that this means that, even when disabling the adapters,
            the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from GPart layers to be set as trainable and saved in the final checkpoint.
        init_bound (`float`):
            The bound for initializing theta_d. When 0 (default), theta_d is initialized as a zeros vector.
            When non-zero, theta_d is initialized with a uniform distribution between -init_bound and init_bound.
        proj_seed (`int`):
            Random seed for initializing the parameter-to-group assignments (used for "random" strategy).
        grouping_strategy (`str`):
            The strategy for assigning parameters to groups. Options are:
            - "random": Random partition using proj_seed (default, original GPart behavior)
            - "signed_magnitude": Deterministic partition by sign and magnitude of pretrained weights
        layers_to_transform (`Union[List[int],int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        isometric (`bool`):
            If True (default), the partition matrix P satisfies P^T P = I_d:
            each column is normalized by 1/sqrt(group_size), making the map from
            theta_d to weight space an isometric embedding.
            If False, no normalization is applied (P^T P = diag(n_1,...,n_d)),
            which is simpler but does not preserve Euclidean distances.
    """

    d: int = field(
        default=1024,
        metadata={"help": "The number of groups for parameter partitioning."},
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with GPart."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            )
        },
    )
    gpart_dropout: float = field(default=0.0, metadata={"help": "GPart dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: str = field(
        default="none",
        metadata={
            "help": (
                "Bias type for GPart. Can be 'none', 'all' or 'gpart_only'. "
                "If 'none' (default), biases are excluded from the GPart partition and remain frozen at their "
                "pretrained values — only weights are updated via theta_d. "
                "If 'all' or 'gpart_only', biases are included in the flattened parameter vector alongside "
                "weights and updated through the same theta_d mechanism. "
                "Be aware that when bias is 'all' or 'gpart_only', the model will not produce the same output "
                "as the base model would have without adaptation, even when disabling the adapters."
            )
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from GPart layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_bound: float = field(
        default=0.0,
        metadata={
            "help": (
                "The bound for initializing theta_d. When 0 (default), theta_d is initialized as a zeros vector."
                " When non-zero, theta_d is initialized with a uniform distribution between -init_bound and init_bound."
            ),
        },
    )
    proj_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for initializing the parameter-to-group assignments."
        },
    )
    grouping_strategy: Literal["random", "signed_magnitude"] = field(
        default="random",
        metadata={
            "help": (
                "The strategy for assigning parameters to groups. Options are: "
                "'random' (default, original GPart behavior) or 'signed_magnitude' (deterministic partition by sign and magnitude)."
            )
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )
    isometric: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), the partition matrix P satisfies P^T P = I_d: "
                "each column is normalized by 1/sqrt(group_size), making the map from "
                "theta_d to weight space an isometric embedding. "
                "If False, no normalization is applied (P^T P = diag(n_1,...,n_d)), "
                "which is simpler but does not preserve Euclidean distances."
            )
        },
    )
    block_granularity: Literal["global", "module_type"] = field(
        default="global",
        metadata={
            "help": (
                "How parameters are grouped into blocks before GPart assignment. "
                "'global' (default): all adapted parameters share one partition (original GPart behavior). "
                "'module_type': one partition per module family (e.g., q_proj, k_proj, v_proj, etc.), "
                "reducing cross-module gradient interference."
            )
        },
    )
    block_budget_rule: Literal["proportional", "uniform", "manual"] = field(
        default="proportional",
        metadata={
            "help": (
                "How the total d budget is split across blocks. "
                "'proportional' (default): d_b ≈ d * N_b / N with largest-remainder rounding. "
                "'uniform': equal d for each block. "
                "'manual': use block_d_map for explicit allocation."
            )
        },
    )
    block_d_map: Optional[Dict[str, int]] = field(
        default=None,
        metadata={
            "help": (
                "Explicit manual allocation of d per block when block_budget_rule='manual'. "
                "Keys are block names (e.g., 'q_proj', 'v_proj', 'other_linear'), values are d_b. "
                "Must sum to d. Ignored when block_budget_rule is not 'manual'."
            )
        },
    )
    block_seed_mode: Literal["shared", "offset"] = field(
        default="shared",
        metadata={
            "help": (
                "Whether all blocks reuse one seed or derive block-local seeds. "
                "'shared' (default): all blocks use the same proj_seed. "
                "'offset': each block derives a unique seed from proj_seed and block name."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.GPART
        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )
