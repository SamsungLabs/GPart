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

import hashlib
import logging
import sys
import warnings

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import GPartConfig
from .layer import GPartLayer, Linear
from .grouping import (
    generate_random_assignment,
    generate_signed_magnitude_assignment,
)

logger = logging.getLogger(__name__)

# Known module-type keywords for block resolution, in priority order.
_MODULE_TYPE_KEYWORDS: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class GPartModel(BaseTuner):
    """
    Creates GPart model from a pretrained transformers model.

    GPart (Global Partition Fine-Tuning) works by:
    1. Flattening all N trainable model parameters into a single vector.
    2. Randomly assigning each parameter to one of d groups using a seeded RNG.
    3. Learning a single vector theta_d ∈ R^d (one scalar per group).
    4. At each forward pass, adding a delta to each parameter:
       - delta_i = theta_d[group[i]] / sqrt(group_size[group[i]])
         The partition matrix P satisfies P^T P = I_d (isometric embedding).
    5. Only theta_d has requires_grad=True; all base model parameters are frozen.

    When block_granularity="module_type", the adapted parameters are partitioned into
    semantically coherent blocks (e.g., q_proj, k_proj, v_proj, etc.) and each block
    gets its own independent GPart subspace, reducing cross-module gradient interference.
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
        # Block-wise index/scale assignment for the initial adapter.
        # Must run after super().__init__() has finished injecting all layers.
        self._assign_blockwise_indices_and_scales(config[adapter_name], adapter_name)

    # ------------------------------------------------------------------
    # PEFT lifecycle hooks
    # ------------------------------------------------------------------

    def _pre_injection_hook(
        self, model: nn.Module, config: GPartConfig, adapter_name: str
    ) -> None:
        # Create the nested block parameter store.
        # gpart_theta_blocks is an nn.ModuleDict mapping adapter_name -> nn.ParameterDict
        # where each ParameterDict maps block_name -> nn.Parameter (theta vector).
        if not hasattr(self, "gpart_theta_blocks"):
            self.gpart_theta_blocks = nn.ModuleDict({})
        if adapter_name not in self.gpart_theta_blocks:
            self.gpart_theta_blocks[adapter_name] = nn.ParameterDict({})

        # Per-adapter, per-block parameter offset tracking.
        if not hasattr(self, "_gpart_block_param_offset"):
            self._gpart_block_param_offset: dict[str, dict[str, int]] = {}
        if adapter_name not in self._gpart_block_param_offset:
            self._gpart_block_param_offset[adapter_name] = {}

        # Per-adapter block budget cache (populated during injection).
        if not hasattr(self, "_gpart_block_budgets"):
            self._gpart_block_budgets: dict[str, dict[str, int]] = {}
        if adapter_name not in self._gpart_block_budgets:
            self._gpart_block_budgets[adapter_name] = {}

        # Per-adapter block param count tracking (for budget allocation).
        if not hasattr(self, "_gpart_block_param_counts"):
            self._gpart_block_param_counts: dict[str, dict[str, int]] = {}
        if adapter_name not in self._gpart_block_param_counts:
            self._gpart_block_param_counts[adapter_name] = {}

        # Track which adapters have completed __init__-time assignment so that
        # _post_injection_hook can skip the first adapter (handled above).
        if not hasattr(self, "_initialized_adapters"):
            self._initialized_adapters: set[str] = set()

    def _post_injection_hook(
        self, model: nn.Module, config: GPartConfig, adapter_name: str
    ) -> None:
        # Re-run block-wise assignment for any adapter added after
        # __init__ (e.g. via add_adapter). The first adapter is handled in
        # __init__ directly; skip it here to avoid double-assignment.
        if adapter_name in self._initialized_adapters:
            self._assign_blockwise_indices_and_scales(config[adapter_name], adapter_name)
        else:
            self._initialized_adapters.add(adapter_name)

    # ------------------------------------------------------------------
    # Block resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_block_name(current_key: str, target_name: str, block_granularity: str = "module_type") -> str:
        """Map a module's name to its block name based on the granularity setting.

        When block_granularity="global", all modules map to the single "global" block.
        When block_granularity="module_type", modules are mapped to semantic blocks
        like q_proj, k_proj, v_proj, etc.
        """
        if block_granularity == "global":
            return "global"

        name = current_key or target_name
        for keyword in _MODULE_TYPE_KEYWORDS:
            if keyword in name:
                return keyword
        return "other_linear"

    @staticmethod
    def _resolve_block_seed(base_seed: int, block_name: str, block_seed_mode: str = "shared") -> int:
        """Derive a block-local seed from the base seed and block name.

        For "shared" mode, all blocks use the same seed.
        For "offset" mode, each block gets a deterministic unique seed.
        """
        if block_seed_mode == "shared":
            return base_seed
        # Derive a deterministic offset seed from the block name.
        h = hashlib.md5(f"{base_seed}_{block_name}".encode()).hexdigest()
        return base_seed + int(h, 16) % (2**31)

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------

    def _allocate_block_budgets(
        self, adapter_name: str, config: GPartConfig
    ) -> dict[str, int]:
        """Distribute the total d budget across blocks.

        Uses the configured block_budget_rule:
        - "proportional": d_b ≈ d * N_b / N with largest-remainder rounding.
        - "uniform": equal d for each block.
        - "manual": use block_d_map from config.

        Returns a dict mapping block_name -> d_block.
        """
        d = config.d
        block_param_counts = self._gpart_block_param_counts.get(adapter_name, {})
        block_names = list(block_param_counts.keys())

        if not block_names:
            return {}

        if config.block_budget_rule == "manual":
            if config.block_d_map is None:
                raise ValueError(
                    "block_d_map must be provided when block_budget_rule='manual'"
                )
            # Validate that all block names are covered
            for bn in block_names:
                if bn not in config.block_d_map:
                    raise ValueError(
                        f"block_d_map is missing entry for block '{bn}'. "
                        f"Available blocks: {block_names}"
                    )
            total = sum(config.block_d_map[bn] for bn in block_names)
            if total != d:
                raise ValueError(
                    f"block_d_map sums to {total}, but d={d}"
                )
            return {bn: config.block_d_map[bn] for bn in block_names}

        if config.block_budget_rule == "uniform":
            d_per_block = d // len(block_names)
            remainder = d % len(block_names)
            budgets = {}
            for i, bn in enumerate(block_names):
                budgets[bn] = d_per_block + (1 if i < remainder else 0)
            return budgets

        # "proportional" (default)
        total_params = sum(block_param_counts.values())
        if total_params == 0:
            return {bn: 0 for bn in block_names}

        # Compute real-valued allocation
        raw_budgets = {bn: d * block_param_counts[bn] / total_params for bn in block_names}

        # Floor allocation
        floor_budgets = {bn: int(raw_budgets[bn]) for bn in block_names}
        allocated = sum(floor_budgets.values())
        remaining = d - allocated

        # Distribute remaining to largest fractional remainders
        if remaining > 0:
            remainders = {bn: raw_budgets[bn] - floor_budgets[bn] for bn in block_names}
            sorted_blocks = sorted(block_names, key=lambda bn: remainders[bn], reverse=True)
            for i in range(remaining):
                floor_budgets[sorted_blocks[i]] += 1

        # Enforce minimum of 1 per block when budget allows
        for bn in block_names:
            if floor_budgets[bn] < 1 and d >= len(block_names):
                floor_budgets[bn] = 1

        return floor_budgets

    # ------------------------------------------------------------------
    # Layer injection
    # ------------------------------------------------------------------

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

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": gpart_config.fan_in_fan_out,
            "bias": bias,
        }

        # Resolve the block name for this module.
        block_name = self._resolve_block_name(
            current_key, target_name, gpart_config.block_granularity
        )

        # Track per-block parameter offset.
        if block_name not in self._gpart_block_param_offset[adapter_name]:
            self._gpart_block_param_offset[adapter_name][block_name] = 0

        # Count parameters for this layer (needed for budget allocation).
        if isinstance(target, BaseTunerLayer):
            base_for_count = target.get_base_layer()
        else:
            base_for_count = target
        param_count = base_for_count.weight.numel()
        if (
            gpart_config.bias != "none"
            and hasattr(base_for_count, "bias")
            and base_for_count.bias is not None
        ):
            param_count += base_for_count.bias.numel()

        # Accumulate per-block param counts for budget allocation.
        if block_name not in self._gpart_block_param_counts[adapter_name]:
            self._gpart_block_param_counts[adapter_name][block_name] = 0
        self._gpart_block_param_counts[adapter_name][block_name] += param_count

        # Note: We do NOT create theta vectors here because the budget allocation
        # depends on knowing all blocks and their param counts, which we only know
        # after all layers are injected. Theta vectors are created in
        # _assign_blockwise_indices_and_scales instead.

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                gpart_theta_blocks=self.gpart_theta_blocks,
                block_name=block_name,
                d=gpart_config.d,  # placeholder; real d_block set later
                gpart_dropout=gpart_config.gpart_dropout,
                bias_config=gpart_config.bias,
            )
            injected_layer = target
        else:
            new_module = self._create_new_module(
                gpart_config=gpart_config,
                gpart_theta_blocks=self.gpart_theta_blocks,
                adapter_name=adapter_name,
                block_name=block_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
            injected_layer = new_module

        # Record this layer's block-local offset and advance the counter.
        if not hasattr(injected_layer, "_gpart_block_name"):
            injected_layer._gpart_block_name = {}
        injected_layer._gpart_block_name[adapter_name] = block_name

        if not hasattr(injected_layer, "_gpart_block_offset"):
            injected_layer._gpart_block_offset = {}
        injected_layer._gpart_block_offset[adapter_name] = self._gpart_block_param_offset[
            adapter_name
        ][block_name]
        self._gpart_block_param_offset[adapter_name][block_name] += param_count

    @staticmethod
    def _create_new_module(
        gpart_config, gpart_theta_blocks, adapter_name, block_name, target, **kwargs
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

        # Get d_block from the budget allocation.
        d_block = gpart_config.d  # fallback
        if hasattr(gpart_config, "block_granularity") and gpart_config.block_granularity != "global":
            # The block budgets should have been computed during injection.
            # We pass the full d here; the layer will get the correct d_block
            # from the theta vector's numel after initialization.
            d_block = gpart_config.d

        new_module = Linear(
            base_layer=target,
            gpart_theta_blocks=gpart_theta_blocks,
            adapter_name=adapter_name,
            block_name=block_name,
            d=d_block,
            gpart_dropout=gpart_config.gpart_dropout,
            bias_config=gpart_config.bias,
            **kwargs,
        )
        return new_module

    # ------------------------------------------------------------------
    # Core GPart logic
    # ------------------------------------------------------------------

    def _init_gpart_theta_block(
        self, config: GPartConfig, adapter_name: str, block_name: str, d_block: int
    ) -> None:
        """Initialize a block-local theta vector (called once per block per adapter)."""
        theta = torch.zeros(d_block)
        if config.init_bound != 0.0:
            torch.nn.init.uniform_(theta, -config.init_bound, config.init_bound)
        self.gpart_theta_blocks[adapter_name][block_name] = nn.Parameter(theta)

    def generate_assignments(
        self,
        total_params: int,
        d: int,
        proj_seed: int,
        strategy: str = "random",
        params_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate parameter-to-group assignments using the specified strategy.

        Args:
            total_params: Total number of parameters to partition.
            d: Number of groups.
            proj_seed: Random seed (used for "random" strategy).
            strategy: Grouping strategy ("random" or "signed_magnitude").
            params_values: Parameter values tensor (required for "signed_magnitude").

        Returns:
            assignments: LongTensor of shape (total_params,) with group IDs.

        Raises:
            ValueError: If d > total_params or if strategy is unknown.
        """
        if strategy == "random":
            return generate_random_assignment(total_params, d, proj_seed)
        elif strategy == "signed_magnitude":
            if params_values is None:
                raise ValueError(
                    "params_values must be provided for signed_magnitude strategy"
                )
            if params_values.numel() != total_params:
                raise ValueError(
                    f"params_values numel ({params_values.numel()}) must match total_params ({total_params})"
                )
            return generate_signed_magnitude_assignment(params_values, d)
        else:
            raise ValueError(f"Unknown grouping strategy: {strategy}")

    def _assign_blockwise_indices_and_scales(
        self, gpart_config: GPartConfig, adapter_name: str
    ) -> None:
        """
        Distribute block-wise parameter-to-group index slices and per-parameter
        scaling factors to every injected GPartLayer.

        For block_granularity="global", this is equivalent to the original
        _assign_global_indices_and_scales (single block named "global").
        For block_granularity="module_type", each module family gets its own
        independent partition.
        """
        # 1. Collect all injected layers that carry this adapter.
        gpart_layers = [
            module
            for _, module in self.model.named_modules()
            if isinstance(module, GPartLayer)
            and adapter_name in module.gpart_indices
            and hasattr(module, "_gpart_block_name")
            and adapter_name in module._gpart_block_name
            and hasattr(module, "_gpart_block_offset")
            and adapter_name in module._gpart_block_offset
        ]

        # 2. Group layers by block.
        block_to_layers: dict[str, list] = {}
        for layer in gpart_layers:
            block = layer._gpart_block_name[adapter_name]
            block_to_layers.setdefault(block, []).append(layer)

        # 3. Compute budget allocation and create theta vectors.
        #    This must happen after all layers are injected so we know the
        #    per-block param counts.
        block_budgets = self._allocate_block_budgets(adapter_name, gpart_config)
        self._gpart_block_budgets[adapter_name] = block_budgets

        for block_name, d_block in block_budgets.items():
            if block_name not in self.gpart_theta_blocks[adapter_name]:
                self._init_gpart_theta_block(gpart_config, adapter_name, block_name, d_block)

        # 4. Process each block independently.
        include_bias = gpart_config.bias != "none"

        for block_name, block_layers in block_to_layers.items():
            # Sort by block-local offset so slice boundaries are unambiguous.
            block_layers.sort(key=lambda m: m._gpart_block_offset[adapter_name])

            d_block = self.gpart_theta_blocks[adapter_name][block_name].numel()

            # Total parameter count for this block.
            total_params = sum(
                layer.get_base_layer().weight.numel()
                + (
                    layer.get_base_layer().bias.numel()
                    if include_bias
                    and hasattr(layer.get_base_layer(), "bias")
                    and layer.get_base_layer().bias is not None
                    else 0
                )
                for layer in block_layers
            )

            if total_params == 0 or d_block == 0:
                continue

            # For signed_magnitude strategy, collect parameter values for this block.
            params_values = None
            if gpart_config.grouping_strategy == "signed_magnitude":
                params_values = self._collect_param_values(
                    block_layers, include_bias=include_bias
                )

            # Resolve block-local seed.
            seed = self._resolve_block_seed(
                gpart_config.proj_seed, block_name, gpart_config.block_seed_mode
            )

            # Generate block-local assignment vector.
            all_indices = self.generate_assignments(
                total_params,
                d_block,
                seed,
                strategy=gpart_config.grouping_strategy,
                params_values=params_values,
            )

            # Slice indices into each layer using the pre-recorded block-local offsets.
            for layer in block_layers:
                base = layer.get_base_layer()
                w_count = base.weight.numel()
                b_count = (
                    base.bias.numel()
                    if include_bias
                    and hasattr(base, "bias")
                    and base.bias is not None
                    else 0
                )
                layer_params = w_count + b_count
                offset = layer._gpart_block_offset[adapter_name]
                layer.gpart_indices[adapter_name] = all_indices[
                    offset : offset + layer_params
                ].clone()

            # Compute block-local group-level scales.
            group_counts = torch.bincount(all_indices, minlength=d_block)
            group_counts = torch.clamp(group_counts, min=1)
            if gpart_config.isometric:
                # P^T P = I_d: normalize each column to unit norm -> 1/sqrt(n_j)
                scales = 1.0 / torch.sqrt(group_counts.float())
            else:
                # P^T P = diag(n_1,...,n_d): no normalization
                scales = torch.ones(d_block, dtype=torch.float32)

            # Push per-parameter scale slices to each layer.
            for layer in block_layers:
                layer_indices = layer.gpart_indices[adapter_name]
                layer.update_scales(adapter_name, scales[layer_indices])

    def _collect_param_values(
        self, gpart_layers: list, include_bias: bool = True
    ) -> torch.Tensor:
        """
        Collect all parameter values (weights and biases) from the given layers
        into a single flattened tensor in global order.

        Args:
            gpart_layers: List of GPartLayer instances in global order.
            include_bias: Whether to include bias values in the collection.
                Should be False when gpart_config.bias == "none".

        Returns:
            params_values: 1D FloatTensor containing all parameter values.
        """
        parts = []
        for layer in gpart_layers:
            base = layer.get_base_layer()
            # Flatten weight (row-major / C-contiguous order)
            weight_flat = base.weight.data.flatten().float()
            parts.append(weight_flat)

            # Flatten bias if present and included
            if (
                include_bias
                and hasattr(base, "bias")
                and base.bias is not None
            ):
                bias_flat = base.bias.data.flatten().float()
                parts.append(bias_flat)

        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_nb_savable_parameters(self, adapter: str = "default") -> tuple[int, int]:
        """
        Returns (theta trainable parameter count, index+scale buffer count).
        """
        theta_params = sum(
            param.numel()
            for name, param in self.named_parameters()
            if "gpart_theta_blocks" in name
        )
        buffer_count = sum(
            buf.numel()
            for name, buf in self.named_buffers()
            if "gpart_indices" in name or "gpart_scales" in name
        )
        return theta_params, buffer_count

    def print_savable_parameters(self) -> None:
        """Prints the number of savable GPart parameters and total savable parameters."""
        gpart_params, buffer_count = self.get_nb_savable_parameters()
        print(
            f"GPart params to-be-saved (float32-equivalent): {gpart_params:,d} "
            f"|| total params to-be-saved: {(gpart_params + buffer_count):,d}"
        )
