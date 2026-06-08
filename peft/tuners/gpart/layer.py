# Copyright 2026 Samsung
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
    other_param_names = ("gpart_indices", "gpart_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.gpart_dropout = nn.ModuleDict({})

        self.gpart_indices = BufferDict({}, persistent=False)
        self._gpart_update_bias: dict[str, bool] = {}

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
        else:
            raise TypeError(f"Unsupported base layer type: {type(base_layer)!r}")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        gpart_theta_d,
        gpart_global_scales,
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

        object.__setattr__(self, "_gpart_theta_d_ref", gpart_theta_d)
        object.__setattr__(self, "_gpart_global_scales_ref", gpart_global_scales)

        self._gpart_update_bias[adapter_name] = bias_config != "none"

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)


class Linear(nn.Linear, GPartLayer):
    """GPart adapter applied to a dense (nn.Linear or Conv1D) layer."""

    def __init__(
        self,
        base_layer,
        gpart_theta_d,
        gpart_global_scales,
        adapter_name: str,
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
            gpart_theta_d,
            gpart_global_scales,
            d,
            gpart_dropout,
            bias_config=bias_config,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def _compute_delta_flat(self, adapter: str, cast_to_fp32: bool) -> torch.Tensor:
        theta_d = self._gpart_theta_d_ref[adapter]
        indices = self.gpart_indices[adapter]
        device = theta_d.device
        if indices.device != device:
            indices = indices.to(device)
        scales = self._gpart_global_scales_ref[adapter]
        if scales.device != device:
            scales = scales.to(device)

        if cast_to_fp32:
            theta_d = theta_d.float()
            scales = scales.float()

        return theta_d[indices] * scales[indices]

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.gpart_indices[adapter].device
        dtype = self._gpart_theta_d_ref[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        delta_flat = self._compute_delta_flat(adapter, cast_to_fp32)
        weight_shape = self.get_base_layer().weight.shape
        delta_weight = delta_flat[: weight_shape.numel()].view(weight_shape)

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
            delta_w = self.get_delta_weight(active_adapter)

            merged_weight = base_layer.weight.detach().clone()
            merged_weight.add_(
                delta_w.to(device=merged_weight.device, dtype=merged_weight.dtype)
            )

            merged_bias = None
            if self._last_bias_delta is not None:
                merged_bias = base_layer.bias.detach().clone()
                merged_bias.add_(
                    self._last_bias_delta.to(
                        device=merged_bias.device, dtype=merged_bias.dtype
                    )
                )

            if safe_merge:
                if not torch.isfinite(merged_weight).all():
                    raise ValueError(
                        f"NaNs/Infs detected in merged weights. "
                        f"The adapter {active_adapter} seems to be broken"
                    )
                if merged_bias is not None and not torch.isfinite(merged_bias).all():
                    raise ValueError(
                        f"NaNs/Infs detected in merged bias. "
                        f"The adapter {active_adapter} seems to be broken"
                    )

            with torch.no_grad():
                base_layer.weight.copy_(merged_weight)
                if merged_bias is not None:
                    base_layer.bias.copy_(merged_bias)

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
            delta_w = self.get_delta_weight(active_adapter)

            with torch.no_grad():
                base_layer.weight.sub_(
                    delta_w.to(
                        device=base_layer.weight.device, dtype=base_layer.weight.dtype
                    )
                )
                if self._last_bias_delta is not None:
                    base_layer.bias.sub_(
                        self._last_bias_delta.to(
                            device=base_layer.bias.device, dtype=base_layer.bias.dtype
                        )
                    )

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

        for active_adapter in self.active_adapters:
            if active_adapter not in self.gpart_indices.keys():
                continue

            theta = self._gpart_theta_d_ref[active_adapter].to(base.weight.device)
            indices = self.gpart_indices[active_adapter]
            if indices.device != base.weight.device:
                indices = indices.to(base.weight.device)
            scales = self._gpart_global_scales_ref[active_adapter].to(
                device=base.weight.device, dtype=base.weight.dtype
            )
            delta_flat = theta[indices] * scales[indices]

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
