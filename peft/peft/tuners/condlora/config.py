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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class CondLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CondLoraModel`].

    CondLoRA (Conditional LoRA) differs from standard LoRA in that instead of learning fixed low-rank matrices A and B,
    it learns small linear networks that take the *weight matrix* as input and produce task-conditioned low-rank
    factors. These networks are shared across all layers targeting the same module type (e.g. all "q_proj" layers share
    the same projection networks). Additionally, CondLoRA supports optional input conditioning via the ``use_x``
    parameter.

    Paper: https://arxiv.org/abs/2403.14946

    Args:
        r (`int`):
            CondLoRA attention dimension (rank). Must be positive.
        target_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to apply CondLoRA to. If this is specified, only modules with the specified names
            will be replaced. When passing a string, a regex match will be performed. When passing a list, either an
            exact match or suffix match is performed. If not specified, modules will be chosen according to the model
            architecture. Note: CondLoRA requires square weight matrices (in_features == out_features), e.g. attention
            projection layers.
        lora_alpha (`int`):
            The alpha parameter for CondLoRA scaling. The effective scaling is ``lora_alpha / r``.
        lora_dropout (`float`):
            The dropout probability for CondLoRA layers.
        fan_in_fan_out (`bool`):
            Set this to ``True`` if the layer to replace stores weight like ``(fan_in, fan_out)`` (e.g. GPT-2's
            Conv1D).
        bias (`str`):
            Bias type for CondLoRA. Can be ``'none'``, ``'all'`` or ``'lora_only'``. If ``'all'`` or
            ``'lora_only'``, the corresponding biases will be updated during training.
        modules_to_save (`Optional[list[str]]`):
            List of modules apart from CondLoRA layers to be set as trainable and saved in the final checkpoint.
        init_lora_weights (`bool`):
            Whether to initialize the shared linear projection weights with the default initialization (Kaiming uniform
            for A, zeros for B). Setting to ``False`` skips initialization entirely (for debugging).
        layers_to_transform (`Optional[Union[list[int], int]]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[list[str], str]]`):
            The layer pattern name, used only if ``layers_to_transform`` is not ``None``. This should target the
            ``nn.ModuleList`` of the model (often called ``'layers'`` or ``'h'``).
        use_x (`Literal["none", "type1", "type2"]`):
            Whether and how to condition the low-rank update on the input activations ``x``.

            - ``"none"``: Standard CondLoRA without input conditioning.
            - ``"type1"``: Adds a residual ``lora_x_scaling * linear_x(x)`` to the intermediate activation before
              multiplying by lora_B. The ``linear_x`` network is shared across all targeted layers.
            - ``"type2"``: Computes a BOS-token–conditioned outer product and adds it to the output. Specifically,
              ``linear_x`` maps the BOS hidden state to a (in_features × in_features) matrix.

        lora_x_scaling (`float`):
            Scaling factor for the input-conditioned term when ``use_x != "none"``. Must be in ``(0.0, 1.0]`` when
            ``use_x != "none"``, and exactly ``0.0`` when ``use_x == "none"``.
    """

    r: int = field(default=8, metadata={"help": "CondLoRA attention dimension (rank)."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with CondLoRA. "
                "For example, ['q_proj', 'v_proj']. Note: CondLoRA is designed for square weight matrices "
                "(in_features == out_features)."
            )
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "CondLoRA alpha scaling factor."})
    lora_dropout: float = field(default=0.0, metadata={"help": "CondLoRA dropout probability."})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if the target layer stores weight like (fan_in, fan_out), e.g. Conv1D."},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none",
        metadata={"help": "Bias type. Can be 'none', 'all' or 'lora_only'."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from CondLoRA layers to be set as trainable and saved in the final checkpoint. "
                "For example, in Sequence Classification tasks, the final 'classifier' layer should be here."
            )
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the shared projection weights. When True, uses Kaiming uniform for the A "
                "projection and zeros for the B projection, matching the standard LoRA default."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indices to transform. If a list of ints is passed, PEFT will transform only the layers "
                "at those indices. If a single integer is passed, only that layer is transformed."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name used when layers_to_transform is set. Targets the nn.ModuleList of the model."
            )
        },
    )
    use_x: Literal["none", "type1", "type2"] = field(
        default="none",
        metadata={
            "help": (
                "Input conditioning mode. 'none': no conditioning. 'type1': adds lora_x_scaling * linear_x(x) as "
                "residual to intermediate activation. 'type2': BOS-conditioned outer product added to output."
            )
        },
    )
    lora_x_scaling: float = field(
        default=0.0,
        metadata={
            "help": (
                "Scaling factor for the input-conditioned term (use_x != 'none'). Must be in (0, 1] when "
                "use_x != 'none' and exactly 0.0 otherwise."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CONDLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # Validate use_x / lora_x_scaling combination
        if self.use_x != "none":
            if not (0.0 < self.lora_x_scaling <= 1.0):
                raise ValueError(
                    f"When use_x != 'none', lora_x_scaling must be in (0.0, 1.0], got {self.lora_x_scaling}."
                )
        else:
            if self.lora_x_scaling != 0.0:
                raise ValueError(
                    f"When use_x == 'none', lora_x_scaling must be 0.0, got {self.lora_x_scaling}. "
                    "Set lora_x_scaling=0.0 or choose a use_x mode."
                )
        if self.r <= 0:
            raise ValueError(f"r must be a positive integer, got {self.r}.")
