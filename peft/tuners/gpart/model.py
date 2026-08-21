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
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .._buffer_dict import BufferDict
from .config import GPartConfig
from .fastfood import (
    generate_fastfood_directional_gains,
    generate_fastfood_state,
)
from .grouping import (
    generate_implicit_group_ids,
    generate_random_assignment,
    generate_signed_magnitude_assignment,
)
from .layer import GPartLayer, Linear


@dataclass
class _GPartBlockManifest:
    block_id: str
    block_ordinal: int
    theta_start: int
    theta_end: int
    d_block: int
    total_params: int
    proj_seed: int
    layers: list[GPartLayer]


class GPartModel(BaseTuner):
    """
    Creates GPart model from a pretrained transformers model.

    GPart learns one shared intrinsic vector and projects it into the flattened
    adapted parameter space. The default ``partition`` projection retains the
    original sparse normalized grouping; ``fastfood`` supplies the structured
    dense random-projection baseline used for intrinsic-dimension comparisons.
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

    def inject_adapter(
        self,
        model: nn.Module,
        adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # BaseTuner invokes _pre_injection_hook only during construction, not
        # when PeftModel.add_adapter calls inject_adapter later.
        if not hasattr(self, "gpart_theta_d") or adapter_name not in self.gpart_theta_d:
            self._pre_injection_hook(model, self.peft_config[adapter_name], adapter_name)
        super().inject_adapter(
            model,
            adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )
        config = self.peft_config[adapter_name]
        if config.partition_scope == "transformer_block":
            self._assign_transformer_block_indices_and_scales(
                config, adapter_name
            )
        else:
            self._assign_global_indices_and_scales(config, adapter_name)
        self._initialized_adapters.add(adapter_name)

    def _pre_injection_hook(
        self, model: nn.Module, config: GPartConfig, adapter_name: str
    ) -> None:
        if not hasattr(self, "gpart_theta_d"):
            self.gpart_theta_d = nn.ParameterDict({})
        if not hasattr(self, "gpart_global_scales"):
            self.gpart_global_scales = BufferDict({}, persistent=False)
        if not hasattr(self, "gpart_fastfood_signs"):
            self.gpart_fastfood_signs = BufferDict({}, persistent=False)
        if not hasattr(self, "gpart_fastfood_gaussian"):
            self.gpart_fastfood_gaussian = BufferDict({}, persistent=False)
        if not hasattr(self, "gpart_fastfood_permutation"):
            self.gpart_fastfood_permutation = BufferDict({}, persistent=False)
        if not hasattr(self, "_initialized_adapters"):
            self._initialized_adapters: set[str] = set()
        if not hasattr(self, "_gpart_param_offset"):
            self._gpart_param_offset: dict[str, int] = {}
        if not hasattr(self, "_gpart_layers"):
            self._gpart_layers: dict[str, list[GPartLayer]] = {}
        if not hasattr(self, "_gpart_block_param_offsets"):
            self._gpart_block_param_offsets: dict[str, dict[str, int]] = {}
        if not hasattr(self, "_gpart_block_manifests"):
            self._gpart_block_manifests: dict[
                str, list[_GPartBlockManifest]
            ] = {}
        if adapter_name not in self.gpart_theta_d:
            self._init_gpart_theta_d(config, adapter_name)
        self._gpart_layers.setdefault(adapter_name, [])
        self._gpart_param_offset.setdefault(adapter_name, 0)
        self._gpart_block_param_offsets.setdefault(adapter_name, {})
        self._gpart_block_manifests.setdefault(adapter_name, [])

    @staticmethod
    def _natural_path_key(path: str) -> tuple[tuple[int, int | str], ...]:
        return tuple(
            (0, int(component)) if component.isdigit() else (1, component)
            for component in path.split(".")
        )

    @staticmethod
    def _resolve_transformer_block_id(
        current_key: str,
        layers_pattern,
    ) -> str | None:
        if layers_pattern is None:
            containers = {"layers", "layer", "h", "block", "blocks"}
        elif isinstance(layers_pattern, str):
            containers = {layers_pattern}
        else:
            containers = set(layers_pattern)

        components = current_key.split(".")
        for index, component in enumerate(components[:-1]):
            if component in containers and components[index + 1].isdigit():
                return ".".join(components[: index + 2])
        return None

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
        self._gpart_block_param_offsets.setdefault(adapter_name, {})

        block_id = None
        if gpart_config.partition_scope == "transformer_block":
            block_id = self._resolve_transformer_block_id(
                current_key,
                gpart_config.layers_pattern,
            )
            if block_id is None:
                raise ValueError(
                    "Could not resolve a numbered transformer block for targeted "
                    f"GPart module {current_key!r}. Narrow target_modules or set "
                    "layers_pattern to the transformer block container name."
                )

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
                gpart_fastfood_signs=self.gpart_fastfood_signs,
                gpart_fastfood_gaussian=self.gpart_fastfood_gaussian,
                gpart_fastfood_permutation=self.gpart_fastfood_permutation,
                d=gpart_config.d,
                gpart_dropout=gpart_config.gpart_dropout,
                bias_config=gpart_config.bias,
                assignment_backend=gpart_config.assignment_backend,
                proj_seed=gpart_config.proj_seed,
                projection_type=gpart_config.projection_type,
                isometric=gpart_config.isometric,
            )
            injected_layer = target
        else:
            new_module = self._create_new_module(
                gpart_config=gpart_config,
                gpart_theta_d=self.gpart_theta_d,
                gpart_global_scales=self.gpart_global_scales,
                gpart_fastfood_signs=self.gpart_fastfood_signs,
                gpart_fastfood_gaussian=self.gpart_fastfood_gaussian,
                gpart_fastfood_permutation=self.gpart_fastfood_permutation,
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

        if gpart_config.partition_scope == "transformer_block":
            block_offsets = self._gpart_block_param_offsets[adapter_name]
            local_offset = block_offsets.get(block_id, 0)
            injected_layer._gpart_param_offset[adapter_name] = local_offset
            injected_layer._gpart_block_id[adapter_name] = block_id
            block_offsets[block_id] = local_offset + param_count
        else:
            injected_layer._gpart_param_offset[adapter_name] = self._gpart_param_offset[
                adapter_name
            ]
            self._gpart_param_offset[adapter_name] += param_count
        self._gpart_layers.setdefault(adapter_name, []).append(injected_layer)

    @staticmethod
    def _create_new_module(
        gpart_config,
        gpart_theta_d,
        gpart_global_scales,
        gpart_fastfood_signs,
        gpart_fastfood_gaussian,
        gpart_fastfood_permutation,
        adapter_name,
        target,
        **kwargs,
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
            gpart_fastfood_signs=gpart_fastfood_signs,
            gpart_fastfood_gaussian=gpart_fastfood_gaussian,
            gpart_fastfood_permutation=gpart_fastfood_permutation,
            adapter_name=adapter_name,
            d=gpart_config.d,
            gpart_dropout=gpart_config.gpart_dropout,
            bias_config=gpart_config.bias,
            assignment_backend=gpart_config.assignment_backend,
            proj_seed=gpart_config.proj_seed,
            projection_type=gpart_config.projection_type,
            isometric=gpart_config.isometric,
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
        projection_type = gpart_config.projection_type
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
        if projection_type == "fastfood" and d > total_params:
            raise ValueError(
                f"Fastfood intrinsic dimension d={d} exceeds the adapted "
                f"dimension D={total_params}"
            )
        for layer in gpart_layers:
            layer._gpart_total_params[adapter_name] = total_params

        if projection_type == "fastfood":
            signs, gaussian, permutation = generate_fastfood_state(
                total_params=total_params,
                d=d,
                proj_seed=proj_seed,
            )
            target_device = self.gpart_theta_d[adapter_name].device
            self.gpart_fastfood_signs[adapter_name] = signs.to(target_device)
            self.gpart_fastfood_gaussian[adapter_name] = gaussian.to(target_device)
            self.gpart_fastfood_permutation[adapter_name] = permutation.to(target_device)
            if isometric:
                directional_gains = torch.ones(1, dtype=torch.float32)
            else:
                directional_gains = generate_fastfood_directional_gains(
                    d=d,
                    proj_seed=proj_seed,
                )
            self.gpart_global_scales[adapter_name] = directional_gains.to(
                target_device
            )
            for layer in gpart_layers:
                if adapter_name in layer.gpart_indices:
                    del layer.gpart_indices[adapter_name]
            return

        for state in (
            self.gpart_fastfood_signs,
            self.gpart_fastfood_gaussian,
            self.gpart_fastfood_permutation,
        ):
            if adapter_name in state:
                del state[adapter_name]

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

    def _assign_transformer_block_indices_and_scales(
        self,
        gpart_config: GPartConfig,
        adapter_name: str,
    ) -> None:
        include_bias = gpart_config.bias != "none"
        grouped_layers: dict[str, list[GPartLayer]] = {}
        for layer in self._gpart_layers.get(adapter_name, []):
            block_id = layer._gpart_block_id[adapter_name]
            grouped_layers.setdefault(block_id, []).append(layer)

        if not grouped_layers:
            return

        block_ids = sorted(grouped_layers, key=self._natural_path_key)
        block_count = len(block_ids)
        if gpart_config.d < block_count:
            raise ValueError(
                "Transformer-block GPart requires at least one coordinate per "
                f"active block, but d={gpart_config.d} and blocks={block_count}"
            )

        base_width, remainder = divmod(gpart_config.d, block_count)
        widths = [
            base_width + (1 if ordinal < remainder else 0)
            for ordinal in range(block_count)
        ]
        scales = torch.empty(gpart_config.d, dtype=torch.float32)
        manifests: list[_GPartBlockManifest] = []
        theta_start = 0

        for ordinal, (block_id, d_block) in enumerate(zip(block_ids, widths)):
            layers = grouped_layers[block_id]
            layers.sort(key=lambda layer: layer._gpart_param_offset[adapter_name])
            total_params = sum(
                layer.get_base_layer().weight.numel()
                + (
                    layer.get_base_layer().bias.numel()
                    if include_bias
                    and getattr(layer.get_base_layer(), "bias", None) is not None
                    else 0
                )
                for layer in layers
            )
            if d_block > total_params:
                raise ValueError(
                    "Invalid transformer-block GPart allocation: "
                    f"d={gpart_config.d}, blocks={block_count}, block={block_id!r}, "
                    f"allocated_width={d_block}, adapted_dimension={total_params}"
                )

            theta_end = theta_start + d_block
            block_seed = gpart_config.proj_seed + ordinal
            for layer in layers:
                layer._gpart_theta_start[adapter_name] = theta_start
                layer._gpart_theta_end[adapter_name] = theta_end
                layer._gpart_d[adapter_name] = d_block
                layer._gpart_proj_seed[adapter_name] = block_seed
                layer._gpart_total_params[adapter_name] = total_params

            if gpart_config.grouping_strategy == "random":
                if gpart_config.assignment_backend == "implicit_stateless_v1":
                    group_counts = self._count_implicit_groups_streaming(
                        gpart_layers=layers,
                        adapter_name=adapter_name,
                        d=d_block,
                        proj_seed=block_seed,
                        include_bias=include_bias,
                    )
                else:
                    group_counts = self._assign_random_indices_streaming(
                        gpart_layers=layers,
                        adapter_name=adapter_name,
                        d=d_block,
                        proj_seed=block_seed,
                        include_bias=include_bias,
                    )
            else:
                params_values = self._collect_param_values(
                    layers,
                    include_bias=include_bias,
                )
                all_indices = self.generate_assignments(
                    total_params=total_params,
                    d=d_block,
                    proj_seed=block_seed,
                    strategy=gpart_config.grouping_strategy,
                    params_values=params_values,
                )
                for layer in layers:
                    base = layer.get_base_layer()
                    layer_params = base.weight.numel() + (
                        base.bias.numel()
                        if include_bias and getattr(base, "bias", None) is not None
                        else 0
                    )
                    offset = layer._gpart_param_offset[adapter_name]
                    layer.gpart_indices[adapter_name] = all_indices[
                        offset : offset + layer_params
                    ].clone().to(base.weight.device)
                group_counts = torch.bincount(all_indices, minlength=d_block)

            group_counts = group_counts.clamp_min(1)
            block_scales = (
                group_counts.float().rsqrt()
                if gpart_config.isometric
                else torch.ones(d_block, dtype=torch.float32)
            )
            scales[theta_start:theta_end] = block_scales
            manifests.append(
                _GPartBlockManifest(
                    block_id=block_id,
                    block_ordinal=ordinal,
                    theta_start=theta_start,
                    theta_end=theta_end,
                    d_block=d_block,
                    total_params=total_params,
                    proj_seed=block_seed,
                    layers=layers,
                )
            )
            theta_start = theta_end

        target_device = self.gpart_theta_d[adapter_name].device
        self.gpart_global_scales[adapter_name] = scales.to(target_device)
        self._gpart_block_manifests[adapter_name] = manifests

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

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        # Fixed Gaussian factors are deliberately retained in FP32 even when
        # the trainable model is cast to a mixed-precision dtype.
        if hasattr(self, "gpart_fastfood_gaussian"):
            for adapter_name in list(self.gpart_fastfood_gaussian.keys()):
                gaussian = self.gpart_fastfood_gaussian[adapter_name]
                if gaussian.dtype != torch.float32:
                    self.gpart_fastfood_gaussian[adapter_name] = gaussian.float()
        return result

    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        for state in (
            self.gpart_theta_d,
            self.gpart_global_scales,
            self.gpart_fastfood_signs,
            self.gpart_fastfood_gaussian,
            self.gpart_fastfood_permutation,
        ):
            if adapter_name in state:
                del state[adapter_name]
        self._gpart_layers.pop(adapter_name, None)
        self._gpart_param_offset.pop(adapter_name, None)
        self._gpart_block_param_offsets.pop(adapter_name, None)
        self._gpart_block_manifests.pop(adapter_name, None)
        self._initialized_adapters.discard(adapter_name)

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        theta_d_params = (
            self.gpart_theta_d[adapter].numel()
            if adapter in self.gpart_theta_d
            else 0
        )
        runtime_buffer_count = sum(
            state[adapter].numel()
            for state in (
                self.gpart_global_scales,
                self.gpart_fastfood_signs,
                self.gpart_fastfood_gaussian,
                self.gpart_fastfood_permutation,
            )
            if adapter in state
        )
        runtime_buffer_count += sum(
            layer.gpart_indices[adapter].numel()
            for layer in self._gpart_layers.get(adapter, [])
            if adapter in layer.gpart_indices
        )
        return theta_d_params, runtime_buffer_count

    def get_fixed_runtime_state_bytes(self, adapter: str = "default") -> int:
        buffer_bytes = sum(
            state[adapter].numel() * state[adapter].element_size()
            for state in (
                self.gpart_global_scales,
                self.gpart_fastfood_signs,
                self.gpart_fastfood_gaussian,
                self.gpart_fastfood_permutation,
            )
            if adapter in state
        )
        buffer_bytes += sum(
            layer.gpart_indices[adapter].numel()
            * layer.gpart_indices[adapter].element_size()
            for layer in self._gpart_layers.get(adapter, [])
            if adapter in layer.gpart_indices
        )
        return buffer_bytes

    def get_runtime_adapter_bytes(self, adapter: str = "default") -> int:
        parameter_bytes = (
            self.gpart_theta_d[adapter].numel()
            * self.gpart_theta_d[adapter].element_size()
            if adapter in self.gpart_theta_d
            else 0
        )
        buffer_bytes = sum(
            state[adapter].numel() * state[adapter].element_size()
            for state in (
                self.gpart_global_scales,
                self.gpart_fastfood_signs,
                self.gpart_fastfood_gaussian,
                self.gpart_fastfood_permutation,
            )
            if adapter in state
        )
        buffer_bytes += sum(
            layer.gpart_indices[adapter].numel()
            * layer.gpart_indices[adapter].element_size()
            for layer in self._gpart_layers.get(adapter, [])
            if adapter in layer.gpart_indices
        )
        return parameter_bytes + buffer_bytes

    def print_savable_parameters(self, adapter: str = "default") -> None:
        gpart_params, _ = self.get_nb_savable_parameters(adapter)
        runtime_bytes = self.get_runtime_adapter_bytes(adapter)
        print(
            f"GPart params to be saved: {gpart_params:,d} "
            f"|| runtime adapter footprint: {runtime_bytes:,d} bytes"
        )
