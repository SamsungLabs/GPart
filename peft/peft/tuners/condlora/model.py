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

import math
import warnings
from typing import Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)

from .config import CondLoraConfig
from .layer import CondLoraLayer, CondLoraLinear


class CondLoraModel(BaseTuner):
    """
    CondLoRA (Conditional LoRA) model wrapping a pretrained model.

    Unlike standard LoRA which learns fixed low-rank matrices A and B, CondLoRA learns small linear networks
    (``cond_lora_A`` and ``cond_lora_B``) that take the *weight matrix* of each targeted layer as input and produce
    task-conditioned low-rank factors. Crucially, these projection networks are **shared across all layers targeting
    the same module name** (e.g. all ``q_proj`` layers in a transformer share the same pair of projection networks).

    The shared networks are stored on the ``CondLoraModel`` itself as ``self.cond_lora_shared``, a nested
    ``nn.ModuleDict`` with structure::

        self.cond_lora_shared[adapter_name][target_name]["A"]  ->  nn.Linear(in_features, r)
        self.cond_lora_shared[adapter_name][target_name]["B"]  ->  nn.Linear(in_features, r)
        self.cond_lora_shared[adapter_name]["__x__"]           ->  nn.Linear(...) or missing

    References to these shared networks are passed to each ``CondLoraLinear`` layer so that gradients are computed
    only once per network per forward pass (since they are the same Python objects).

    Paper: https://arxiv.org/abs/2403.14946

    Args:
        model (`nn.Module`):
            The model to be adapted.
        config (`CondLoraConfig`):
            The CondLoRA configuration.
        adapter_name (`str`):
            The name of the adapter (defaults to ``"default"``).

    Example::

        from transformers import AutoModelForCausalLM
        from peft import get_peft_model, CondLoraConfig, TaskType

        config = CondLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model = get_peft_model(model, config)
    """

    prefix: str = "cond_lora_"
    tuner_layer_cls = CondLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(self, model: nn.Module, config: CondLoraConfig, adapter_name: str) -> None:
        # Storage for the shared projection networks, keyed by adapter_name → target_module_name → "A"/"B"
        # Call super().__init__() first to properly initialize nn.Module
        super().__init__(model, config, adapter_name)
        # Now we can safely initialize cond_lora_shared
        if not hasattr(self, "cond_lora_shared"):
            self.cond_lora_shared = nn.ModuleDict({})

    # ------------------------------------------------------------------
    # BaseTuner required overrides
    # ------------------------------------------------------------------

    def _prepare_adapter_config(self, peft_config: CondLoraConfig, model_config: dict) -> CondLoraConfig:
        if peft_config.target_modules is None:
            if model_config.get("model_type") not in self.target_module_mapping:
                raise ValueError(
                    "Please specify `target_modules` in `peft_config`. "
                    "CondLoRA is designed for square weight matrices (in_features == out_features), "
                    "e.g. attention projection layers."
                )
            peft_config.target_modules = set(self.target_module_mapping[model_config["model_type"]])
        return peft_config

    def _create_and_replace(
        self,
        peft_config: CondLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **kwargs,
    ) -> None:
        if current_key is None:
            raise ValueError("current_key must not be None.")

        # Determine the short "module type" name for network sharing.
        # e.g. "model.layers.0.self_attn.q_proj" → "q_proj"
        module_type = current_key.split(".")[-1]

        if isinstance(target, CondLoraLayer):
            # Already a CondLoRA layer (e.g. adding a second adapter)
            cond_lora_A, cond_lora_B, cond_lora_x = self._get_or_create_shared_networks(
                adapter_name, module_type, peft_config, target.in_features, target.out_features
            )
            target.update_layer(
                adapter_name=adapter_name,
                r=peft_config.r,
                lora_alpha=peft_config.lora_alpha,
                lora_dropout=peft_config.lora_dropout,
                init_lora_weights=peft_config.init_lora_weights,
                use_x=peft_config.use_x,
                lora_x_scaling=peft_config.lora_x_scaling,
                cond_lora_A=cond_lora_A,
                cond_lora_B=cond_lora_B,
                cond_lora_x=cond_lora_x,
            )
        else:
            kwargs_create = {
                "r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
                "fan_in_fan_out": peft_config.fan_in_fan_out,
                "init_lora_weights": peft_config.init_lora_weights,
                "use_x": peft_config.use_x,
                "lora_x_scaling": peft_config.lora_x_scaling,
            }
            new_module = self._create_new_module(
                peft_config, adapter_name, target, module_type, **kwargs_create
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(
        self,
        peft_config: CondLoraConfig,
        adapter_name: str,
        target: nn.Module,
        module_type: str,
        **kwargs,
    ) -> CondLoraLinear:
        """
        Create a new CondLoraLinear wrapping the target layer.

        Also registers the shared projection networks (if not already created) in ``self.cond_lora_shared``.
        """
        if isinstance(target, BaseTunerLayer):
            target_base = target.get_base_layer()
        else:
            target_base = target

        fan_in_fan_out = kwargs.get("fan_in_fan_out", False)
        is_target_conv_1d_layer = False

        if isinstance(target_base, nn.Linear):
            in_features, out_features = target_base.in_features, target_base.out_features
            if fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False.",
                    UserWarning,
                )
                kwargs["fan_in_fan_out"] = peft_config.fan_in_fan_out = False
        elif isinstance(target_base, Conv1D):
            shape = target_base.weight.ds_shape if hasattr(target_base.weight, "ds_shape") else target_base.weight.shape
            in_features, out_features = shape  # Conv1D stores (in_features, out_features)
            is_target_conv_1d_layer = True
            if not fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True.",
                    UserWarning,
                )
                kwargs["fan_in_fan_out"] = peft_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. CondLoRA currently supports "
                "`torch.nn.Linear` and `transformers.pytorch_utils.Conv1D`."
            )

        cond_lora_A, cond_lora_B, cond_lora_x = self._get_or_create_shared_networks(
            adapter_name, module_type, peft_config, in_features, out_features
        )

        new_module = CondLoraLinear(
            base_layer=target,
            adapter_name=adapter_name,
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            fan_in_fan_out=kwargs.get("fan_in_fan_out", False),
            init_lora_weights=peft_config.init_lora_weights,
            use_x=peft_config.use_x,
            lora_x_scaling=peft_config.lora_x_scaling,
            cond_lora_A=cond_lora_A,
            cond_lora_B=cond_lora_B,
            cond_lora_x=cond_lora_x,
            is_target_conv_1d_layer=is_target_conv_1d_layer,
        )
        return new_module

    def _get_or_create_shared_networks(
        self,
        adapter_name: str,
        module_type: str,
        peft_config: CondLoraConfig,
        in_features: int,
        out_features: int,
    ) -> tuple[nn.Linear, nn.Linear, Optional[nn.Linear]]:
        """
        Return (or lazily create) the shared ``cond_lora_A``, ``cond_lora_B`` and optionally ``cond_lora_x``
        projection networks for a given adapter and module type.

        All layers of the same module type and adapter share the same network objects.
        """
        # Ensure cond_lora_shared exists (needed during initialization)
        if not hasattr(self, "cond_lora_shared"):
            self.cond_lora_shared = nn.ModuleDict({})
        
        if adapter_name not in self.cond_lora_shared:
            self.cond_lora_shared[adapter_name] = nn.ModuleDict()

        adapter_shared = self.cond_lora_shared[adapter_name]

        if module_type not in adapter_shared:
            adapter_shared[module_type] = nn.ModuleDict()
            # cond_lora_A: in_features → r  (applied row-wise to weight matrix)
            net_A = nn.Linear(in_features=in_features, out_features=peft_config.r, bias=False)
            # cond_lora_B: in_features → r
            net_B = nn.Linear(in_features=in_features, out_features=peft_config.r, bias=False)

            if peft_config.init_lora_weights:
                nn.init.kaiming_uniform_(net_A.weight, a=math.sqrt(5))
                nn.init.zeros_(net_B.weight)

            adapter_shared[module_type]["A"] = net_A
            adapter_shared[module_type]["B"] = net_B

        cond_lora_A = adapter_shared[module_type]["A"]
        cond_lora_B = adapter_shared[module_type]["B"]

        # Handle the shared input-conditioning network (shared across ALL module types)
        cond_lora_x: Optional[nn.Linear] = None
        if peft_config.use_x != "none":
            x_key = "__x__"
            if x_key not in adapter_shared:
                if peft_config.use_x == "type1":
                    # Maps input features → r (same shape as cond_lora_A)
                    net_x = nn.Linear(in_features=in_features, out_features=peft_config.r, bias=False)
                elif peft_config.use_x == "type2":
                    # Maps scalar (1) → in_features for BOS-token outer product
                    net_x = nn.Linear(in_features=1, out_features=in_features, bias=False)
                else:
                    raise ValueError(f"Unknown use_x mode: {peft_config.use_x!r}")

                nn.init.zeros_(net_x.weight)
                adapter_shared[x_key] = net_x

            cond_lora_x = adapter_shared[x_key]

        return cond_lora_A, cond_lora_B, cond_lora_x

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Freeze all base model parameters; keep only CondLoRA parameters trainable.

        CondLoRA's shared projection networks (``self.cond_lora_shared``) use the prefix ``cond_lora_`` in their
        parameter names but are stored on *this* (CondLoraModel) module, not inside ``model``. The default
        BaseTuner logic only scans ``model.named_parameters()``, so we must explicitly ensure the shared networks
        are marked trainable here as well.
        """
        # First run the standard logic (freeze base model, unfreeze adapter params in model)
        super()._mark_only_adapters_as_trainable(model)

        # Also ensure the shared networks on self are trainable (if they exist)
        if hasattr(self, "cond_lora_shared"):
            for p in self.cond_lora_shared.parameters():
                p.requires_grad = True

        # Handle bias modes
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            elif bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, CondLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Bias mode '{bias}' is not supported.")

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False) -> dict:
        from dataclasses import asdict
        from enum import Enum

        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def enable_adapter_layers(self) -> None:
        """Enable all CondLoRA adapter layers."""
        self._enable_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all CondLoRA adapter layers."""
        self._enable_adapter_layers(enabled=False)

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> nn.Module:
        """
        Merge the CondLoRA adapter(s) into the base model weights and return the unloaded base model.

        Note: because the CondLoRA delta weight depends on the current base weights, merging captures the adapter's
        effect at the current weight values.
        """
        return self._unload_and_optionally_merge(
            merge=True, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> nn.Module:
        """Return the base model by removing all CondLoRA modules without merging."""
        return self._unload_and_optionally_merge(merge=False)
