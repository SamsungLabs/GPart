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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class GPartLayer(BaseTunerLayer):
    adapter_layer_names = ()
    other_param_names = ("gpart_indices", "gpart_scales", "gpart_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.gpart_dropout = nn.ModuleDict({})

        # Per-parameter group assignments — LongTensor, shape (param_count,)
        # These are non-persistent because they can be recomputed from the seed.
        self.gpart_indices = BufferDict({}, persistent=False)

        # Per-parameter scaling factors — shape (param_count,)
        # Isometric mode: 1/sqrt(group_size); non-isometric: 1.0
        # These are non-persistent because they can be recomputed from the seed.
        self.gpart_scales = BufferDict({}, persistent=False)

        # Per-adapter flag: whether to include bias in the GPart partition.
        # True when bias config is "all" or "gpart_only"; False when "none".
        self._gpart_update_bias: dict[str, bool] = {}

        # Per-adapter block name (which block this layer belongs to).
        self._gpart_block_name: dict[str, str] = {}

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape
                if hasattr(base_layer.weight, "ds_shape")
                else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        gpart_theta_blocks,
        block_name: str,
        d: int,
        gpart_dropout: float = 0.0,
        bias_config: str = "none",
    ):
        if d <= 0:
            raise ValueError(f"`d` {d} should be a positive integer value")

        dropout_layer = (
            nn.Dropout(p=gpart_dropout) if gpart_dropout > 0.0 else nn.Identity()
        )
        self.gpart_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        # Store the model-level nested ParameterDict as a plain Python
        # attribute (_gpart_theta_blocks_ref), NOT as a registered submodule.
        # If registered, PyTorch would register it on every wrapped layer,
        # causing the theta parameters to appear N times in state_dict —
        # corrupting checkpoints.
        # Use object.__setattr__ to bypass nn.Module's __setattr__ which would
        # register nn.ModuleDict as a submodule.
        object.__setattr__(self, "_gpart_theta_blocks_ref", gpart_theta_blocks)

        # Store which block this layer belongs to.
        self._gpart_block_name[adapter_name] = block_name

        # Store whether biases should be included in the GPart partition.
        # "none" → biases excluded (frozen); "all"/"gpart_only" → biases included.
        self._gpart_update_bias[adapter_name] = bias_config != "none"

        self.reset_gpart_parameters(adapter_name, d)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _get_theta(self, adapter_name: str) -> nn.Parameter:
        """Fetch the block-local theta vector for this adapter and layer's block."""
        block_name = self._gpart_block_name[adapter_name]
        return self._gpart_theta_blocks_ref[adapter_name][block_name]

    def reset_gpart_parameters(self, adapter_name: str, d: int):
        """
        Sets placeholder zero-indices and unit scales for this layer.
        The real globally-consistent assignments are pushed by GPartModel
        via update_scales / direct gpart_indices assignment after all layers
        are injected.
        """
        param_count = self.in_features * self.out_features
        if (
            self._gpart_update_bias.get(adapter_name, False)
            and hasattr(self.base_layer, "bias")
            and self.base_layer.bias is not None
        ):
            param_count += self.out_features

        self.gpart_indices[adapter_name] = torch.zeros(param_count, dtype=torch.long)
        self.gpart_scales[adapter_name] = torch.ones(param_count, dtype=torch.float)

    def update_scales(self, adapter_name: str, gpart_scales: torch.Tensor):
        """Push per-parameter scaling factors computed globally by GPartModel."""
        # Access theta via _gpart_theta_blocks_ref
        block_name = self._gpart_block_name.get(adapter_name)
        if block_name is not None and adapter_name in self._gpart_theta_blocks_ref:
            if block_name in self._gpart_theta_blocks_ref[adapter_name]:
                base_layer = self.get_base_layer()
                target_device = base_layer.weight.device
                target_dtype = base_layer.weight.dtype
                self.gpart_scales[adapter_name] = gpart_scales.to(
                    device=target_device, dtype=target_dtype
                )


class Linear(nn.Linear, GPartLayer):
    """GPart adapter applied to a dense (nn.Linear or Conv1D) layer."""

    def __init__(
        self,
        base_layer,
        gpart_theta_blocks,
        adapter_name: str,
        block_name: str,
        d: int,
        gpart_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        bias_config: str = "none",
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        GPartLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            gpart_theta_blocks,
            block_name,
            d,
            gpart_dropout,
            bias_config=bias_config,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_delta_flat(self, adapter: str, cast_to_fp32: bool) -> torch.Tensor:
        """
        Returns delta_flat where:
          delta_flat[i] = theta[group(i)] * scale(i)
        covering all parameters of this layer (weight first, then bias).
        """
        theta = self._get_theta(adapter)
        device = self.gpart_indices[adapter].device
        indices = self.gpart_indices[adapter].to(device)
        scales = self.gpart_scales[adapter].to(device)
        theta = theta.to(device)

        if cast_to_fp32:
            theta = theta.float()
            scales = scales.float()

        delta_flat = theta[indices] * scales
        return delta_flat

    # ------------------------------------------------------------------
    # Merge / Unmerge
    # ------------------------------------------------------------------

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the weight delta for merging.
        Also caches the bias delta in self._last_bias_delta for merge/unmerge.
        """
        device = self.gpart_indices[adapter].device
        dtype = self._get_theta(adapter).dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        delta_flat = self._compute_delta_flat(adapter, cast_to_fp32)
        weight_shape = self.get_base_layer().weight.shape
        delta_weight = delta_flat[: weight_shape.numel()].view(weight_shape)

        # extract and cache bias delta instead of discarding it
        self._last_bias_delta = None
        base_layer = self.get_base_layer()
        if (
            self._gpart_update_bias.get(adapter, False)
            and hasattr(base_layer, "bias")
            and base_layer.bias is not None
        ):
            bias_start = weight_shape.numel()
            bias_end = bias_start + base_layer.bias.numel()
            bias_delta = delta_flat[bias_start:bias_end].view(base_layer.bias.shape)
            if cast_to_fp32:
                bias_delta = bias_delta.to(dtype)
            self._last_bias_delta = bias_delta

        return transpose(delta_weight, self.fan_in_fan_out)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.gpart_indices.keys():
                continue

            base_layer = self.get_base_layer()
            delta_w = self.get_delta_weight(active_adapter)  # sets _last_bias_delta

            if safe_merge:
                orig_weights = base_layer.weight.data.clone()
                orig_weights += delta_w
                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. "
                        f"The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = orig_weights
            else:
                base_layer.weight.data += delta_w.to(base_layer.weight.device)

            # apply bias delta during merge
            if self._last_bias_delta is not None:
                base_layer.bias.data += self._last_bias_delta.to(base_layer.bias.device)

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.gpart_indices.keys():
                continue
            base_layer = self.get_base_layer()
            delta_w = self.get_delta_weight(active_adapter)  # sets _last_bias_delta
            base_layer.weight.data -= delta_w.to(base_layer.weight.device)

            # subtract bias delta during unmerge
            if self._last_bias_delta is not None:
                base_layer.bias.data -= self._last_bias_delta.to(base_layer.bias.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        if self.merged:
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        valid_adapters = [
            adapter
            for adapter in self.active_adapters
            if adapter in self.gpart_indices.keys()
        ]

        if not valid_adapters:
            return self.base_layer(x, *args, **kwargs).to(previous_dtype)

        base = self.get_base_layer()
        x_in = x.to(base.weight.dtype)

        eff_weight = base.weight
        eff_bias = base.bias

        for active_adapter in valid_adapters:
            theta = self._get_theta(active_adapter).to(base.weight.device)
            indices = self.gpart_indices[active_adapter].to(base.weight.device)
            scales = self.gpart_scales[active_adapter].to(
                device=base.weight.device, dtype=base.weight.dtype
            )
            delta_flat = theta[indices] * scales

            w_numel = base.weight.numel()
            delta_w = delta_flat[:w_numel].view_as(base.weight)
            delta_w = self.gpart_dropout[active_adapter](delta_w)
            eff_weight = eff_weight + delta_w

            if (
                self._gpart_update_bias.get(active_adapter, False)
                and base.bias is not None
            ):
                delta_b = delta_flat[w_numel : w_numel + base.bias.numel()].view_as(
                    base.bias
                )
                eff_bias = delta_b if eff_bias is None else (eff_bias + delta_b)

        out = F.linear(x_in, eff_weight, eff_bias)
        return out.to(previous_dtype)
