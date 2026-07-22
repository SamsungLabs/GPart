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

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import GPartConfig
from .grouping import (
    generate_implicit_group_ids,
    generate_random_assignment,
    generate_signed_magnitude_assignment,
)
from .layer import GPartLayer, Linear


class GPartModel(BaseTuner):
    """
    Creates GPart model from a pretrained transformers model.

    GPart (Global Partition Fine-Tuning) works by:
    1. Flattening all N trainable model parameters into a single vector.
    2. Randomly assigning each parameter to one of d groups using a seeded RNG.
    3. Learning a single vector theta_d ∈ R^d (one scalar per group).
    4. At each forward pass, adding a delta to each parameter.
    5. Only theta_d has requires_grad=True; all base model parameters are frozen.
    """

    prefix: str = "gpart_"
    tuner_layer_cls = GPartLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __init__(
        self, model, config, adapter_name, low_cpu_mem_usage: bool = False
    ) -> None:
        super().__init__(
            model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self._assign_global_indices_and_scales(config[adapter_name], adapter_name)
        self._initialized_adapters.add(adapter_name)

    def _pre_injection_hook(
        self, model: nn.Module, config: GPartConfig, adapter_name: str
    ) -> None:
        if not hasattr(self, "gpart_theta_d"):
            self.gpart_theta_d = nn.ParameterDict({})
        if not hasattr(self, "gpart_global_scales"):
            self.gpart_global_scales = BufferDict({}, persistent=False)
        if not hasattr(self, "_initialized_adapters"):
            self._initialized_adapters: set[str] = set()
        if not hasattr(self, "_gpart_param_offset"):
            self._gpart_param_offset: dict[str, int] = {}
        if not hasattr(self, "_gpart_layers"):
            self._gpart_layers: dict[str, list[GPartLayer]] = {}
        if not hasattr(self, "_constructor_adapter_name"):
            self._constructor_adapter_name = adapter_name

        if adapter_name not in self.gpart_theta_d:
            self._init_gpart_theta_d(config, adapter_name)
        self._gpart_layers.setdefault(adapter_name, [])
        self._gpart_param_offset.setdefault(adapter_name, 0)

    def _post_injection_hook(
        self, model: nn.Module, config: GPartConfig, adapter_name: str
    ) -> None:
        if adapter_name == self._constructor_adapter_name:
            return
        self._assign_global_indices_and_scales(config[adapter_name], adapter_name)
        self._initialized_adapters.add(adapter_name)

    def _create_and_replace(
        self,
        gpart_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        self._gpart_param_offset.setdefault(adapter_name, 0)
        self._gpart_layers.setdefault(adapter_name, [])

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": gpart_config.fan_in_fan_out,
            "bias": bias,
        }

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                gpart_theta_d=self.gpart_theta_d,
                gpart_global_scales=self.gpart_global_scales,
                d=gpart_config.d,
                gpart_dropout=gpart_config.gpart_dropout,
                bias_config=gpart_config.bias,
                assignment_backend=gpart_config.assignment_backend,
                proj_seed=gpart_config.proj_seed,
            )
            injected_layer = target
        else:
            new_module = self._create_new_module(
                gpart_config=gpart_config,
                gpart_theta_d=self.gpart_theta_d,
                gpart_global_scales=self.gpart_global_scales,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )

            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
            injected_layer = new_module

        base = injected_layer.get_base_layer()
        param_count = base.weight.numel()
        if (
            gpart_config.bias != "none"
            and hasattr(base, "bias")
            and base.bias is not None
        ):
            param_count += base.bias.numel()

        if not hasattr(injected_layer, "_gpart_param_offset"):
            injected_layer._gpart_param_offset = {}
        injected_layer._gpart_param_offset[adapter_name] = self._gpart_param_offset[
            adapter_name
        ]
        self._gpart_param_offset[adapter_name] += param_count
        self._gpart_layers.setdefault(adapter_name, []).append(injected_layer)

    @staticmethod
    def _create_new_module(
        gpart_config, gpart_theta_d, gpart_global_scales, adapter_name, target, **kwargs
    ):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is "
                    "`torch.nn.Linear`. Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = gpart_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is "
                    "`Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = gpart_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the "
                "following modules are supported: `torch.nn.Linear`, "
                "`transformers.pytorch_utils.Conv1D`."
            )

        return Linear(
            base_layer=target,
            gpart_theta_d=gpart_theta_d,
            gpart_global_scales=gpart_global_scales,
            adapter_name=adapter_name,
            d=gpart_config.d,
            gpart_dropout=gpart_config.gpart_dropout,
            bias_config=gpart_config.bias,
            assignment_backend=gpart_config.assignment_backend,
            proj_seed=gpart_config.proj_seed,
            **kwargs,
        )

    def _init_gpart_theta_d(self, config: GPartConfig, adapter_name: str) -> None:
        gpart_theta_d = torch.zeros(config.d)
        if config.init_bound != 0.0:
            torch.nn.init.uniform_(gpart_theta_d, -config.init_bound, config.init_bound)
        self.gpart_theta_d[adapter_name] = nn.Parameter(gpart_theta_d)

    def generate_assignments(
        self,
        total_params: int,
        d: int,
        proj_seed: int,
        strategy: str = "random",
        params_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if strategy == "random":
            return generate_random_assignment(total_params, d, proj_seed)
        if strategy == "signed_magnitude":
            if params_values is None:
                raise ValueError(
                    "params_values must be provided for signed_magnitude strategy"
                )
            if params_values.numel() != total_params:
                raise ValueError(
                    f"params_values numel ({params_values.numel()}) must match total_params ({total_params})"
                )
            return generate_signed_magnitude_assignment(params_values, d)
        raise ValueError(f"Unknown grouping strategy: {strategy}")

    def _assign_global_indices_and_scales(
        self, gpart_config: GPartConfig, adapter_name: str
    ) -> None:
        d = gpart_config.d
        proj_seed = gpart_config.proj_seed
        isometric = gpart_config.isometric
        strategy = gpart_config.grouping_strategy
        include_bias = gpart_config.bias != "none"
        assignment_backend = gpart_config.assignment_backend

        gpart_layers = list(self._gpart_layers.get(adapter_name, []))
        gpart_layers.sort(key=lambda m: m._gpart_param_offset[adapter_name])

        if not gpart_layers:
            return

        total_params = sum(
            layer.get_base_layer().weight.numel()
            + (
                layer.get_base_layer().bias.numel()
                if include_bias
                and hasattr(layer.get_base_layer(), "bias")
                and layer.get_base_layer().bias is not None
                else 0
            )
            for layer in gpart_layers
        )

        if strategy == "random":
            if assignment_backend == "implicit_stateless_v1":
                group_counts = self._count_implicit_groups_streaming(
                    gpart_layers=gpart_layers,
                    adapter_name=adapter_name,
                    d=d,
                    proj_seed=proj_seed,
                    include_bias=include_bias,
                )
            else:
                group_counts = self._assign_random_indices_streaming(
                    gpart_layers=gpart_layers,
                    adapter_name=adapter_name,
                    d=d,
                    proj_seed=proj_seed,
                    include_bias=include_bias,
                )
        else:
            params_values = None
            if strategy == "signed_magnitude":
                params_values = self._collect_param_values(
                    gpart_layers, include_bias=include_bias
                )

            all_indices = self.generate_assignments(
                total_params,
                d,
                proj_seed,
                strategy=strategy,
                params_values=params_values,
            )

            for layer in gpart_layers:
                base = layer.get_base_layer()
                w_count = base.weight.numel()
                b_count = (
                    base.bias.numel()
                    if include_bias and hasattr(base, "bias") and base.bias is not None
                    else 0
                )
                layer_params = w_count + b_count
                offset = layer._gpart_param_offset[adapter_name]
                layer_indices = all_indices[offset : offset + layer_params].clone()
                layer.gpart_indices[adapter_name] = layer_indices.to(
                    device=base.weight.device
                )

            group_counts = torch.bincount(all_indices, minlength=d)

        group_counts = torch.clamp(group_counts, min=1)
        if isometric:
            scales = 1.0 / torch.sqrt(group_counts.float())
        else:
            scales = torch.ones(d, dtype=torch.float32)

        target_device = self.gpart_theta_d[adapter_name].device
        self.gpart_global_scales[adapter_name] = scales.to(target_device)

    def _count_implicit_groups_streaming(
        self,
        gpart_layers: list[GPartLayer],
        adapter_name: str,
        d: int,
        proj_seed: int,
        include_bias: bool,
        chunk_size: int = 1_048_576,
    ) -> torch.Tensor:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        group_counts = torch.zeros(d, dtype=torch.int64, device="cpu")
        expected_offset = 0

        for layer in gpart_layers:
            offset = layer._gpart_param_offset[adapter_name]
            if offset != expected_offset:
                raise ValueError(
                    "GPart canonical parameter offsets are not contiguous: "
                    f"expected {expected_offset}, got {offset}"
                )

            base = layer.get_base_layer()
            layer_params = base.weight.numel() + (
                base.bias.numel()
                if include_bias and hasattr(base, "bias") and base.bias is not None
                else 0
            )
            if adapter_name in layer.gpart_indices:
                del layer.gpart_indices[adapter_name]

            for local_start in range(0, layer_params, chunk_size):
                chunk_numel = min(chunk_size, layer_params - local_start)
                group_ids = generate_implicit_group_ids(
                    start_offset=offset + local_start,
                    numel=chunk_numel,
                    d=d,
                    proj_seed=proj_seed,
                    device=torch.device("cpu"),
                )
                group_counts += torch.bincount(group_ids, minlength=d)

            expected_offset += layer_params

        return group_counts

    def _assign_random_indices_streaming(
        self,
        gpart_layers: list[GPartLayer],
        adapter_name: str,
        d: int,
        proj_seed: int,
        include_bias: bool,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(proj_seed)
        group_counts = torch.zeros(d, dtype=torch.int32)

        for layer in gpart_layers:
            base = layer.get_base_layer()
            layer_params = base.weight.numel() + (
                base.bias.numel()
                if include_bias and hasattr(base, "bias") and base.bias is not None
                else 0
            )
            layer_indices_cpu = torch.randint(
                low=0,
                high=d,
                size=(layer_params,),
                generator=generator,
                dtype=torch.int32,
            )
            group_counts += torch.bincount(layer_indices_cpu, minlength=d)
            layer.gpart_indices[adapter_name] = layer_indices_cpu.to(base.weight.device)

        return group_counts

    def _collect_param_values(
        self, gpart_layers: list[GPartLayer], include_bias: bool = True
    ) -> torch.Tensor:
        parts = []
        for layer in gpart_layers:
            base = layer.get_base_layer()
            parts.append(base.weight.detach().reshape(-1).float().cpu())
            if include_bias and hasattr(base, "bias") and base.bias is not None:
                parts.append(base.bias.detach().reshape(-1).float().cpu())
        return torch.cat(parts, dim=0)

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        theta_d_params = sum(
            param.numel()
            for name, param in self.named_parameters()
            if "gpart_theta_d" in name
        )
        buffer_count = sum(
            buf.numel()
            for name, buf in self.named_buffers()
            if "gpart_indices" in name or "gpart_global_scales" in name
        )
        return theta_d_params, buffer_count

    def print_savable_parameters(self) -> None:
        gpart_params, buffer_count = self.get_nb_savable_parameters()
        print(
            f"GPart params to-be-saved (float32-equivalent): {gpart_params:,d} "
            f"|| total runtime adapter footprint: {(gpart_params + buffer_count):,d}"
        )
