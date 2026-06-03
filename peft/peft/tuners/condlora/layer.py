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

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class CondLoraLayer(BaseTunerLayer):
    """
    Base mixin class for CondLoRA layers.

    CondLoRA replaces the static lora_A / lora_B matrices of standard LoRA with two small linear networks
    (``cond_lora_A`` and ``cond_lora_B``) that operate on the *weight matrix* of the target layer to produce
    task-conditioned, weight-dependent low-rank factors.

    Importantly, these projection networks are **shared** across all target layers of the same module type (e.g. all
    ``q_proj`` layers in a transformer share the same ``cond_lora_A`` and ``cond_lora_B``). The sharing is managed at
    the model level (``CondLoraModel``) and references to the shared networks are stored in the per-adapter dicts
    below.

    The forward pass computes:
        lora_A = cond_lora_A(W).T          # (r, out_features)
        lora_B = cond_lora_B(W)            # (out_features, r)
        delta_y = lora_B(lora_A(dropout(x))) * scaling
    """

    # Names of ModuleDicts that contain adapter parameters (used by BaseTunerLayer.get_base_layer, etc.)
    adapter_layer_names: tuple[str, ...] = ("cond_lora_A", "cond_lora_B", "cond_lora_x")
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_x_scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r: dict[str, int] = {}
        self.lora_alpha: dict[str, int] = {}
        self.scaling: dict[str, float] = {}
        self.lora_x_scaling: dict[str, float] = {}
        self.use_x: dict[str, str] = {}
        self.lora_dropout = nn.ModuleDict({})
        # Shared projection networks stored per adapter name. Each value is an nn.Linear.
        # Note: Multiple CondLoraLinear layers that share the same cond_lora_A/B will
        # hold references to the *same* nn.Linear objects so gradients accumulate correctly.
        self.cond_lora_A = nn.ModuleDict({})
        self.cond_lora_B = nn.ModuleDict({})
        self.cond_lora_x = nn.ModuleDict({})  # optional input-conditioning network
        self._disable_adapters = False
        self.merged_adapters: list[str] = []
        self.kwargs = kwargs

        base = self.get_base_layer()
        if isinstance(base, nn.Linear):
            self.in_features = base.in_features
            self.out_features = base.out_features
        else:
            # Conv1D stores weights as (in_features, out_features)
            shape = base.weight.ds_shape if hasattr(base.weight, "ds_shape") else base.weight.shape
            self.in_features, self.out_features = shape

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
        use_x: str,
        lora_x_scaling: float,
        cond_lora_A: nn.Linear,
        cond_lora_B: nn.Linear,
        cond_lora_x: Optional[nn.Linear],
    ) -> None:
        """
        Register all adapter-specific state for a given adapter name.

        The ``cond_lora_A``, ``cond_lora_B``, and ``cond_lora_x`` networks are provided by the model (they may be
        shared with other layers). This method stores references to them so that gradients flow correctly and the
        networks are included in the module's parameter set.
        """
        if r <= 0:
            raise ValueError(f"r must be a positive integer, got {r}.")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
        self.lora_x_scaling[adapter_name] = lora_x_scaling
        self.use_x[adapter_name] = use_x

        # Dropout
        lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer

        # Store references to the shared linear networks
        self.cond_lora_A[adapter_name] = cond_lora_A
        self.cond_lora_B[adapter_name] = cond_lora_B
        if cond_lora_x is not None:
            self.cond_lora_x[adapter_name] = cond_lora_x

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the weight delta W_delta = lora_B @ lora_A produced by applying the shared projections to the
        base weight matrix. Used for merge/unmerge operations.

        Note: CondLoRA is designed for square weight matrices (in_features == out_features). The delta weight has
        shape (out_features, out_features) == (out_features, in_features) when square.
        """
        weight = self.get_base_layer().weight
        # linear_for_lora_A: (in_features → r), so applied to weight (out, in) → (out, r)
        lora_A = self.cond_lora_A[adapter_name](weight)  # (out_features, r)
        lora_B = self.cond_lora_B[adapter_name](weight)  # (out_features, r)
        # delta = lora_B @ lora_A.T = (out_features, r) @ (r, out_features) = (out_features, out_features)
        delta = lora_B @ lora_A.T
        return transpose(delta, self.fan_in_fan_out)


class CondLoraLinear(nn.Module, CondLoraLayer):
    """
    CondLoRA adapter applied to a ``torch.nn.Linear`` (or ``Conv1D``) layer.

    The base layer is wrapped (not subclassed) following the modern PEFT convention, so the base weights are
    accessible via ``self.get_base_layer()``.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        fan_in_fan_out: bool,
        init_lora_weights: bool,
        use_x: str,
        lora_x_scaling: float,
        cond_lora_A: nn.Linear,
        cond_lora_B: nn.Linear,
        cond_lora_x: Optional[nn.Linear] = None,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        CondLoraLayer.__init__(self, base_layer=base_layer, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_x=use_x,
            lora_x_scaling=lora_x_scaling,
            cond_lora_A=cond_lora_A,
            cond_lora_B=cond_lora_B,
            cond_lora_x=cond_lora_x,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the CondLoRA adapter weights into the base layer weights.

        Because CondLoRA's effective A/B matrices are derived from the current weight (not fixed), merging permanently
        bakes in the current delta weight.

        Args:
            safe_merge (`bool`):
                If True, check for NaN values before merging.
            adapter_names (`list[str]`, *optional*):
                Adapters to merge. Defaults to all active adapters.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.cond_lora_A:
                continue
            if active_adapter in self.merged_adapters:
                warnings.warn(f"Adapter '{active_adapter}' is already merged. Skipping.")
                continue

            base_layer = self.get_base_layer()
            delta_weight = self.get_delta_weight(active_adapter) * self.scaling[active_adapter]

            if safe_merge and not torch.isfinite(delta_weight).all():
                raise ValueError(
                    f"NaN or Inf values found in delta_weight for adapter '{active_adapter}'. "
                    "Use safe_merge=False to skip this check."
                )

            base_layer.weight.data += delta_weight
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Un-merge all merged adapters from the base layer weights.
        """
        if not self.merged_adapters:
            warnings.warn("Nothing to unmerge — no adapters are currently merged.")
            return

        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.cond_lora_A:
                continue
            base_layer = self.get_base_layer()
            delta_weight = self.get_delta_weight(active_adapter) * self.scaling[active_adapter]
            base_layer.weight.data -= delta_weight

    def _linear(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the base linear transformation."""
        return F.linear(x, transpose(self.get_base_layer().weight, self.fan_in_fan_out), bias=self.get_base_layer().bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self._disable_adapters:
            # Adapters disabled — unmerge first if merged, then run base
            if self.merged_adapters:
                self.unmerge()
            result = self._linear(x)
        elif self.merged_adapters:
            # Already merged into base weights
            result = self._linear(x)
        else:
            # Active CondLoRA forward pass
            result = self._linear(x)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.cond_lora_A:
                    continue

                cond_lora_A = self.cond_lora_A[active_adapter]
                cond_lora_B = self.cond_lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                lora_x_scaling = self.lora_x_scaling[active_adapter]
                use_x = self.use_x[active_adapter]

                weight = self.get_base_layer().weight  # (out_features, in_features)

                # Derive per-forward low-rank factors from the current weight
                # cond_lora_A: nn.Linear(in_features, r), weight shape (r, in_features)
                # applied to weight (out_features, in_features) → (out_features, r)
                lora_A = cond_lora_A(weight).T  # (r, out_features)  [= (r, in_features) for square weights]
                lora_B = cond_lora_B(weight)     # (out_features, r)

                x_cast = x.to(cond_lora_A.weight.dtype)
                x_dropped = dropout(x_cast)

                if use_x == "type1":
                    # type1: add input-conditioned residual to intermediate activation
                    cond_lora_x = self.cond_lora_x[active_adapter]
                    x_ = lora_x_scaling * cond_lora_x(x_dropped)  # (batch, seq, r)
                    lora_out = F.linear(x_dropped, lora_A)          # (batch, seq, r)
                    lora_out = lora_out + x_
                    lora_out = F.linear(lora_out, lora_B)           # (batch, seq, out_features)
                    result = result + lora_out * scaling

                elif use_x == "type2":
                    # type2: BOS-token outer product conditioning
                    cond_lora_x = self.cond_lora_x[active_adapter]
                    # Take BOS (first token) hidden states → shape (batch, in_features)
                    bos_hidden = x_dropped[:, 0, :]  # (batch, in_features)
                    # Unsqueeze to (batch, in_features, 1) then apply linear (1 → in_features)
                    lora_x = cond_lora_x(bos_hidden.unsqueeze(-1))  # (batch, in_features, in_features)
                    # Outer-product conditioning: x @ lora_x * scale
                    x_ = torch.bmm(x_dropped, lora_x) * lora_x_scaling * scaling  # (batch, seq, in_features)

                    lora_out = F.linear(x_dropped, lora_A)   # (batch, seq, r)
                    lora_out = F.linear(lora_out, lora_B)    # (batch, seq, out_features)
                    result = result + lora_out * scaling + x_

                else:
                    # use_x == "none"
                    lora_out = F.linear(x_dropped, lora_A)   # (batch, seq, r)
                    lora_out = F.linear(lora_out, lora_B)    # (batch, seq, out_features)
                    result = result + lora_out * scaling

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "cond_lora." + rep
