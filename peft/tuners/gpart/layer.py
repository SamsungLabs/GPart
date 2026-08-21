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
from torch.autograd.function import once_differentiable
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict
from .fastfood import fastfood_project_slice, fastfood_project_transpose_slice
from .grouping import generate_implicit_group_ids


def _capture_dropout_rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "cpu":
        return torch.get_rng_state()
    raise RuntimeError(f"GPart dropout recomputation is unsupported on {device.type}")


def _dropout_mask_like(
    reference: torch.Tensor,
    dropout_p: float,
    rng_state: torch.Tensor | None = None,
) -> torch.Tensor:
    def make_mask() -> torch.Tensor:
        return F.dropout(
            torch.ones_like(reference),
            p=dropout_p,
            training=True,
        )

    if rng_state is None:
        return make_mask()

    if reference.device.type == "cuda":
        device_index = reference.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]):
            torch.cuda.set_rng_state(rng_state, reference.device)
            return make_mask()

    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state)
        return make_mask()


class _ImplicitGPartLinearFunction(torch.autograd.Function):
    """GPart contribution with assignment and update recomputation."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        theta: torch.Tensor,
        scales: torch.Tensor,
        weight_rows: int,
        weight_cols: int,
        bias_numel: int,
        start_offset: int,
        d: int,
        proj_seed: int,
        fan_in_fan_out: bool,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        weight_rows = int(weight_rows)
        weight_cols = int(weight_cols)
        bias_numel = int(bias_numel)
        start_offset = int(start_offset)
        d = int(d)
        proj_seed = int(proj_seed)
        fan_in_fan_out = bool(fan_in_fan_out)
        dropout_p = float(dropout_p)
        training = bool(training)

        weight_numel = weight_rows * weight_cols
        group_ids = generate_implicit_group_ids(
            start_offset=start_offset,
            numel=weight_numel + bias_numel,
            d=d,
            proj_seed=proj_seed,
            device=x.device,
        )
        delta_flat = (theta * scales).index_select(0, group_ids)
        delta_weight = delta_flat[:weight_numel].view(weight_rows, weight_cols)

        dropout_rng_state = None
        if training and dropout_p > 0.0:
            dropout_rng_state = _capture_dropout_rng_state(x.device)
            delta_weight = delta_weight * _dropout_mask_like(
                delta_weight,
                dropout_p,
            )

        linear_weight = delta_weight.transpose(0, 1) if fan_in_fan_out else delta_weight
        delta_bias = None
        if bias_numel:
            delta_bias = delta_flat[weight_numel:].view(bias_numel)

        result = F.linear(x, linear_weight, delta_bias)

        ctx.save_for_backward(x, theta, scales)
        ctx.weight_rows = weight_rows
        ctx.weight_cols = weight_cols
        ctx.bias_numel = bias_numel
        ctx.start_offset = start_offset
        ctx.d = d
        ctx.proj_seed = proj_seed
        ctx.fan_in_fan_out = fan_in_fan_out
        ctx.dropout_p = dropout_p
        ctx.training = training
        ctx.dropout_rng_state = dropout_rng_state
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        x, theta, scales = ctx.saved_tensors
        weight_numel = ctx.weight_rows * ctx.weight_cols
        group_ids = generate_implicit_group_ids(
            start_offset=ctx.start_offset,
            numel=weight_numel + ctx.bias_numel,
            d=ctx.d,
            proj_seed=ctx.proj_seed,
            device=x.device,
        )
        delta_flat = (theta * scales).index_select(0, group_ids)
        delta_weight = delta_flat[:weight_numel].view(
            ctx.weight_rows,
            ctx.weight_cols,
        )

        dropout_mask = None
        if ctx.training and ctx.dropout_p > 0.0:
            dropout_mask = _dropout_mask_like(
                delta_weight,
                ctx.dropout_p,
                ctx.dropout_rng_state,
            )
            delta_weight = delta_weight * dropout_mask

        linear_weight = (
            delta_weight.transpose(0, 1) if ctx.fan_in_fan_out else delta_weight
        )
        out_features, in_features = linear_weight.shape
        grad_output_2d = grad_output.reshape(-1, out_features).to(linear_weight.dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d.matmul(linear_weight).view_as(x)

        grad_theta = None
        if ctx.needs_input_grad[1]:
            x_2d = x.reshape(-1, in_features)
            grad_linear_weight = grad_output_2d.transpose(0, 1).matmul(x_2d)
            grad_weight = (
                grad_linear_weight.transpose(0, 1)
                if ctx.fan_in_fan_out
                else grad_linear_weight
            )
            if dropout_mask is not None:
                grad_weight = grad_weight * dropout_mask

            grad_parts = [grad_weight.reshape(-1)]
            if ctx.bias_numel:
                grad_parts.append(grad_output_2d.sum(dim=0))
            grad_delta_flat = torch.cat(grad_parts)

            grad_source = grad_delta_flat * scales.index_select(0, group_ids)
            grad_theta = torch.zeros_like(theta)
            grad_theta.scatter_add_(0, group_ids, grad_source)

        return (
            grad_x,
            grad_theta,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FastfoodGPartLinearFunction(torch.autograd.Function):
    """Layer-streamed Fastfood contribution with transform recomputation."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        theta: torch.Tensor,
        signs: torch.Tensor,
        gaussian: torch.Tensor,
        permutation: torch.Tensor,
        directional_gains: torch.Tensor,
        total_params: int,
        start_offset: int,
        weight_rows: int,
        weight_cols: int,
        bias_numel: int,
        isometric: bool,
        fan_in_fan_out: bool,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        total_params = int(total_params)
        start_offset = int(start_offset)
        weight_rows = int(weight_rows)
        weight_cols = int(weight_cols)
        bias_numel = int(bias_numel)
        isometric = bool(isometric)
        fan_in_fan_out = bool(fan_in_fan_out)
        dropout_p = float(dropout_p)
        training = bool(training)

        weight_numel = weight_rows * weight_cols
        delta_flat = fastfood_project_slice(
            theta,
            signs,
            gaussian,
            permutation,
            directional_gains=directional_gains,
            total_params=total_params,
            start_offset=start_offset,
            numel=weight_numel + bias_numel,
            isometric=isometric,
        )
        delta_weight = delta_flat[:weight_numel].view(weight_rows, weight_cols)

        dropout_rng_state = None
        if training and dropout_p > 0.0:
            dropout_rng_state = _capture_dropout_rng_state(x.device)
            delta_weight = delta_weight * _dropout_mask_like(delta_weight, dropout_p)

        linear_weight = delta_weight.transpose(0, 1) if fan_in_fan_out else delta_weight
        delta_bias = None
        if bias_numel:
            delta_bias = delta_flat[weight_numel:].view(bias_numel)
        result = F.linear(x, linear_weight, delta_bias)

        ctx.save_for_backward(x, theta)
        # Fixed projection factors already live in non-persistent model
        # buffers. Keep references without registering them as saved autograd
        # tensors, which avoids treating the full global state as per-layer
        # activation storage.
        ctx.signs = signs
        ctx.gaussian = gaussian
        ctx.permutation = permutation
        ctx.directional_gains = directional_gains
        ctx.total_params = total_params
        ctx.start_offset = start_offset
        ctx.weight_rows = weight_rows
        ctx.weight_cols = weight_cols
        ctx.bias_numel = bias_numel
        ctx.isometric = isometric
        ctx.fan_in_fan_out = fan_in_fan_out
        ctx.dropout_p = dropout_p
        ctx.training = training
        ctx.dropout_rng_state = dropout_rng_state
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        x, theta = ctx.saved_tensors
        weight_numel = ctx.weight_rows * ctx.weight_cols
        delta_flat = fastfood_project_slice(
            theta,
            ctx.signs,
            ctx.gaussian,
            ctx.permutation,
            directional_gains=ctx.directional_gains,
            total_params=ctx.total_params,
            start_offset=ctx.start_offset,
            numel=weight_numel + ctx.bias_numel,
            isometric=ctx.isometric,
        )
        delta_weight = delta_flat[:weight_numel].view(
            ctx.weight_rows,
            ctx.weight_cols,
        )

        dropout_mask = None
        if ctx.training and ctx.dropout_p > 0.0:
            dropout_mask = _dropout_mask_like(
                delta_weight,
                ctx.dropout_p,
                ctx.dropout_rng_state,
            )
            delta_weight = delta_weight * dropout_mask

        linear_weight = (
            delta_weight.transpose(0, 1)
            if ctx.fan_in_fan_out
            else delta_weight
        )
        out_features, in_features = linear_weight.shape
        grad_output_2d = grad_output.reshape(-1, out_features).to(linear_weight.dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d.matmul(linear_weight).view_as(x)

        grad_theta = None
        if ctx.needs_input_grad[1]:
            x_2d = x.reshape(-1, in_features)
            grad_linear_weight = grad_output_2d.transpose(0, 1).matmul(x_2d)
            grad_weight = (
                grad_linear_weight.transpose(0, 1)
                if ctx.fan_in_fan_out
                else grad_linear_weight
            )
            if dropout_mask is not None:
                grad_weight = grad_weight * dropout_mask

            grad_parts = [grad_weight.reshape(-1)]
            if ctx.bias_numel:
                grad_parts.append(grad_output_2d.sum(dim=0))
            grad_delta_flat = torch.cat(grad_parts)
            grad_theta = fastfood_project_transpose_slice(
                grad_delta_flat,
                ctx.signs,
                ctx.gaussian,
                ctx.permutation,
                directional_gains=ctx.directional_gains,
                theta_numel=theta.numel(),
                total_params=ctx.total_params,
                start_offset=ctx.start_offset,
                isometric=ctx.isometric,
            )

        return (
            grad_x,
            grad_theta,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class GPartLayer(BaseTunerLayer):
    adapter_layer_names = ()
    other_param_names = (
        "gpart_indices",
        "gpart_dropout",
        "_gpart_update_bias",
        "_gpart_assignment_backend",
        "_gpart_projection_type",
        "_gpart_isometric",
        "_gpart_total_params",
        "_gpart_d",
        "_gpart_proj_seed",
        "_gpart_param_offset",
        "_gpart_block_id",
        "_gpart_theta_start",
        "_gpart_theta_end",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.gpart_dropout = nn.ModuleDict({})

        self.gpart_indices = BufferDict({}, persistent=False)
        self._gpart_update_bias: dict[str, bool] = {}
        self._gpart_assignment_backend: dict[str, str] = {}
        self._gpart_projection_type: dict[str, str] = {}
        self._gpart_isometric: dict[str, bool] = {}
        self._gpart_total_params: dict[str, int] = {}
        self._gpart_d: dict[str, int] = {}
        self._gpart_proj_seed: dict[str, int] = {}
        self._gpart_param_offset: dict[str, int] = {}
        self._gpart_block_id: dict[str, str] = {}
        self._gpart_theta_start: dict[str, int] = {}
        self._gpart_theta_end: dict[str, int] = {}

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
        gpart_fastfood_signs,
        gpart_fastfood_gaussian,
        gpart_fastfood_permutation,
        d: int,
        gpart_dropout: float = 0.0,
        bias_config: str = "none",
        assignment_backend: str = "materialized",
        proj_seed: int = 42,
        projection_type: str = "partition",
        isometric: bool = True,
    ):
        if d <= 0:
            raise ValueError(f"`d` {d} should be a positive integer value")

        dropout_layer = (
            nn.Dropout(p=gpart_dropout) if gpart_dropout > 0.0 else nn.Identity()
        )
        self.gpart_dropout.update(nn.ModuleDict({adapter_name: dropout_layer}))

        object.__setattr__(self, "_gpart_theta_d_ref", gpart_theta_d)
        object.__setattr__(self, "_gpart_global_scales_ref", gpart_global_scales)
        object.__setattr__(self, "_gpart_fastfood_signs_ref", gpart_fastfood_signs)
        object.__setattr__(self, "_gpart_fastfood_gaussian_ref", gpart_fastfood_gaussian)
        object.__setattr__(
            self, "_gpart_fastfood_permutation_ref", gpart_fastfood_permutation
        )

        self._gpart_update_bias[adapter_name] = bias_config != "none"
        self._gpart_assignment_backend[adapter_name] = assignment_backend
        self._gpart_projection_type[adapter_name] = projection_type
        self._gpart_isometric[adapter_name] = bool(isometric)
        self._gpart_d[adapter_name] = d
        self._gpart_proj_seed[adapter_name] = int(proj_seed)
        self._gpart_theta_start[adapter_name] = 0
        self._gpart_theta_end[adapter_name] = d

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)


class Linear(nn.Linear, GPartLayer):
    """GPart adapter applied to a dense (nn.Linear or Conv1D) layer."""

    def __init__(
        self,
        base_layer,
        gpart_theta_d,
        gpart_global_scales,
        gpart_fastfood_signs,
        gpart_fastfood_gaussian,
        gpart_fastfood_permutation,
        adapter_name: str,
        d: int,
        gpart_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        bias_config: str = "none",
        assignment_backend: str = "materialized",
        proj_seed: int = 42,
        projection_type: str = "partition",
        isometric: bool = True,
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
            gpart_fastfood_signs,
            gpart_fastfood_gaussian,
            gpart_fastfood_permutation,
            d,
            gpart_dropout,
            bias_config=bias_config,
            assignment_backend=assignment_backend,
            proj_seed=proj_seed,
            projection_type=projection_type,
            isometric=isometric,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def _get_group_ids(
        self,
        adapter: str,
        device: torch.device,
    ) -> torch.Tensor:
        if self._gpart_assignment_backend[adapter] == "stateless":
            base = self.get_base_layer()
            bias_numel = (
                base.bias.numel()
                if self._gpart_update_bias.get(adapter, False) and base.bias is not None
                else 0
            )
            return generate_implicit_group_ids(
                start_offset=self._gpart_param_offset[adapter],
                numel=base.weight.numel() + bias_numel,
                d=self._gpart_d[adapter],
                proj_seed=self._gpart_proj_seed[adapter],
                device=device,
            )

        indices = self.gpart_indices[adapter]
        return indices if indices.device == device else indices.to(device)

    def _get_theta_and_scales(
        self,
        adapter: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._gpart_theta_start[adapter]
        end = self._gpart_theta_end[adapter]
        theta = self._gpart_theta_d_ref[adapter][start:end]
        scales = self._gpart_global_scales_ref[adapter]
        if self._gpart_projection_type[adapter] == "partition":
            scales = scales[start:end]
        return theta, scales

    def _compute_delta_flat(self, adapter: str, cast_to_fp32: bool) -> torch.Tensor:
        theta_d, scales = self._get_theta_and_scales(adapter)
        device = theta_d.device
        if cast_to_fp32:
            theta_d = theta_d.float()

        base = self.get_base_layer()
        bias_numel = (
            base.bias.numel()
            if self._gpart_update_bias.get(adapter, False) and base.bias is not None
            else 0
        )
        if self._gpart_projection_type[adapter] == "fastfood":
            return fastfood_project_slice(
                theta_d,
                self._gpart_fastfood_signs_ref[adapter],
                self._gpart_fastfood_gaussian_ref[adapter],
                self._gpart_fastfood_permutation_ref[adapter],
                directional_gains=self._gpart_global_scales_ref[adapter],
                total_params=self._gpart_total_params[adapter],
                start_offset=self._gpart_param_offset[adapter],
                numel=base.weight.numel() + bias_numel,
                isometric=self._gpart_isometric[adapter],
            )

        indices = self._get_group_ids(adapter, device)
        if scales.device != device:
            scales = scales.to(device)

        if cast_to_fp32:
            scales = scales.float()

        return (theta_d * scales)[indices]

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        theta_d, _ = self._get_theta_and_scales(adapter)
        dtype = theta_d.dtype
        cast_to_fp32 = theta_d.device.type == "cpu" and dtype == torch.float16

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
            if active_adapter not in self.gpart_dropout:
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
            if active_adapter not in self.gpart_dropout:
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
            adapter for adapter in self.active_adapters if adapter in self.gpart_dropout
        ]
        result = self.base_layer(x, *args, **kwargs)
        if not valid_adapters:
            return result.to(previous_dtype)

        base = self.get_base_layer()
        target_dtype = base.weight.dtype
        target_device = base.weight.device
        x_in = x.to(device=target_device, dtype=target_dtype)

        for active_adapter in valid_adapters:
            theta, scales = self._get_theta_and_scales(active_adapter)
            theta = theta.to(
                device=target_device,
                dtype=target_dtype,
            )
            scales = scales.to(
                device=target_device,
                dtype=target_dtype,
            )
            update_bias = (
                self._gpart_update_bias.get(active_adapter, False)
                and base.bias is not None
            )
            bias_numel = base.bias.numel() if update_bias else 0

            dropout = self.gpart_dropout[active_adapter]
            dropout_p = dropout.p if isinstance(dropout, nn.Dropout) else 0.0
            if self._gpart_projection_type[active_adapter] == "fastfood":
                contribution = _FastfoodGPartLinearFunction.apply(
                    x_in,
                    theta,
                    self._gpart_fastfood_signs_ref[active_adapter],
                    self._gpart_fastfood_gaussian_ref[active_adapter],
                    self._gpart_fastfood_permutation_ref[active_adapter],
                    scales,
                    self._gpart_total_params[active_adapter],
                    self._gpart_param_offset[active_adapter],
                    base.weight.shape[0],
                    base.weight.shape[1],
                    bias_numel,
                    self._gpart_isometric[active_adapter],
                    self.fan_in_fan_out,
                    dropout_p,
                    dropout.training,
                )
            elif (
                self._gpart_assignment_backend[active_adapter]
                == "stateless"
            ):
                contribution = _ImplicitGPartLinearFunction.apply(
                    x_in,
                    theta,
                    scales,
                    base.weight.shape[0],
                    base.weight.shape[1],
                    bias_numel,
                    self._gpart_param_offset[active_adapter],
                    self._gpart_d[active_adapter],
                    self._gpart_proj_seed[active_adapter],
                    self.fan_in_fan_out,
                    dropout_p,
                    dropout.training,
                )
            else:
                indices = self.gpart_indices[active_adapter]
                if indices.device != target_device:
                    indices = indices.to(target_device)
                delta_flat = (theta * scales)[indices]

                weight_numel = base.weight.numel()
                delta_weight = delta_flat[:weight_numel].view_as(base.weight)
                delta_weight = self.gpart_dropout[active_adapter](delta_weight)
                linear_weight = transpose(delta_weight, self.fan_in_fan_out)

                delta_bias = None
                if bias_numel:
                    delta_bias = delta_flat[weight_numel:].view_as(base.bias)
                contribution = F.linear(x_in, linear_weight, delta_bias)

            result = result + contribution.to(result.dtype)

        return result.to(previous_dtype)
