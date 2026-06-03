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

"""
Grouping strategies for GPart (Global Partition Fine-Tuning).

This module provides different strategies for partitioning the flattened parameter
vector into d disjoint groups. The partition matrix P has one nonzero per row with
value 1/sqrt(n_j) for group j, which preserves P^T P = I_d.

Available strategies:
- "random": Random partition using a seeded RNG (original GPart behavior)
- "signed_magnitude": Deterministic partition by sign and magnitude of pretrained weights
"""

import logging
import sys
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def generate_random_assignment(
    total_params: int,
    d: int,
    proj_seed: int,
) -> torch.Tensor:
    """
    Generate a random parameter-to-group assignment using a seeded RNG.

    This is the original GPart grouping strategy. Each parameter is randomly
    assigned to one of d groups, with group sizes as balanced as possible.

    Args:
        total_params: Total number of parameters to partition (N).
        d: Number of groups.
        proj_seed: Random seed for reproducibility.

    Returns:
        assignments: LongTensor of shape (total_params,) where assignments[i] = g
            means parameter i is assigned to group g.

    Raises:
        ValueError: If d > total_params (would result in empty groups).
    """
    if d > total_params:
        raise ValueError(
            f"d={d} cannot exceed total_params={total_params} if groups must be nonempty."
        )

    generator = torch.Generator()
    generator.manual_seed(proj_seed)

    perm = torch.randperm(total_params, generator=generator)
    assignments = torch.empty(total_params, dtype=torch.long)

    base = total_params // d
    rem = total_params % d

    start = 0
    for g in range(d):
        size = base + (1 if g < rem else 0)
        idx = perm[start : start + size]
        assignments[idx] = g
        start += size

    return assignments


def generate_signed_magnitude_assignment(
    params_values: torch.Tensor,
    d: int,
) -> torch.Tensor:
    """
    Generate a deterministic parameter-to-group assignment based on signed magnitude.

    This strategy partitions parameters by:
    1. Splitting all indices into two pools by sign (negative vs non-negative)
    2. Within each pool, sorting indices by absolute value (magnitude) ascending
    3. Allocating groups to each sign pool proportionally to pool size
    4. Inside each sign pool, splitting sorted indices into contiguous equal-count bins

    This is a deterministic alternative to random grouping that may capture
    meaningful structure in the pretrained weights.

    Args:
        params_values: Tensor of shape (total_params,) containing the pretrained
            weight values for each parameter in global order.
        d: Number of groups.

    Returns:
        assignments: LongTensor of shape (total_params,) where assignments[i] = g
            means parameter i is assigned to group g.

    Raises:
        ValueError: If d > total_params or if d < 1.

    Example:
        >>> params = torch.randn(1000)
        >>> assignments = generate_signed_magnitude_assignment(params, d=10)
        >>> assignments.shape
        torch.Size([1000])
    """
    total_params = params_values.numel()

    if d < 1:
        raise ValueError(f"d must be at least 1, got {d}")
    if d > total_params:
        raise ValueError(
            f"d={d} cannot exceed total_params={total_params} if groups must be nonempty."
        )

    # Compute sign and magnitude for each parameter
    signs = (params_values >= 0).long()  # 0 for negative, 1 for non-negative
    magnitudes = params_values.abs()

    # Split into two pools
    neg_mask = signs == 0
    pos_mask = signs == 1

    neg_indices = torch.nonzero(neg_mask, as_tuple=True)[0]  # indices of negative weights
    pos_indices = torch.nonzero(pos_mask, as_tuple=True)[0]  # indices of non-negative weights

    n_neg = neg_indices.numel()
    n_pos = pos_indices.numel()

    logger.info(
        f"Signed-magnitude grouping: N={total_params}, d={d}, "
        f"n_neg={n_neg} ({100*n_neg/total_params:.1f}%), n_pos={n_pos} ({100*n_pos/total_params:.1f}%)"
    )

    # Handle edge cases where one pool is empty
    if n_neg == 0:
        # All non-negative: just sort by magnitude and split into d groups
        logger.info("All weights are non-negative; using single-pool signed-magnitude grouping")
        return _split_by_magnitude(pos_indices, magnitudes, d, start_group=0)
    
    if n_pos == 0:
        # All negative: just sort by magnitude and split into d groups
        logger.info("All weights are negative; using single-pool signed-magnitude grouping")
        return _split_by_magnitude(neg_indices, magnitudes, d, start_group=0)

    # Both pools non-empty: allocate groups proportionally
    # Ensure each pool gets at least one group if d >= 2
    if d == 1:
        # Single group: everything goes together
        assignments = torch.zeros(total_params, dtype=torch.long)
        return assignments

    # Proportional allocation with constraints
    # k_neg = round(d * n_neg / N), k_pos = d - k_neg
    k_neg = round(d * n_neg / total_params)
    k_pos = d - k_neg

    # Ensure each pool gets at least one group
    if k_neg < 1:
        k_neg = 1
        k_pos = d - 1
    if k_pos < 1:
        k_pos = 1
        k_neg = d - 1

    # Sanity check: ensure k_neg + k_pos == d
    assert k_neg + k_pos == d, f"Group allocation error: {k_neg} + {k_pos} != {d}"

    logger.info(f"Group allocation: k_neg={k_neg}, k_pos={k_pos}")

    # Allocate output tensor
    assignments = torch.empty(total_params, dtype=torch.long)

    # Split negative pool into k_neg groups
    if k_neg > 0 and n_neg > 0:
        neg_assignments = _split_by_magnitude(
            neg_indices, magnitudes, k_neg, start_group=0
        )
        assignments[neg_indices] = neg_assignments

    # Split positive pool into k_pos groups (starting from group k_neg)
    if k_pos > 0 and n_pos > 0:
        pos_assignments = _split_by_magnitude(
            pos_indices, magnitudes, k_pos, start_group=0
        )
        # Offset positive pool groups to start after negative pool groups
        assignments[pos_indices] = pos_assignments + k_neg

    return assignments


def _split_by_magnitude(
    indices: torch.Tensor,
    magnitudes: torch.Tensor,
    k: int,
    start_group: int = 0,
) -> torch.Tensor:
    """
    Split a set of indices into k groups based on sorted magnitudes.

    Uses equal-count binning (not equal-width) to avoid empty groups and
    keep group sizes balanced.

    Args:
        indices: 1D LongTensor of parameter indices to partition.
        magnitudes: 1D FloatTensor of magnitudes for all parameters.
        k: Number of groups to create.
        start_group: Starting group ID (useful for concatenating multiple pools).

    Returns:
        assignments: LongTensor of shape (indices.shape[0],) with group IDs
            in the range [start_group, start_group + k).
            assignments[i] is the group ID for indices[i].
    """
    n = indices.numel()

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n:
        # More groups than items: each item gets its own group
        # This shouldn't happen in normal usage but handle gracefully
        assignments = torch.arange(n, dtype=torch.long) + start_group
        return assignments

    # Sort indices by (magnitude, original_index) for deterministic tie-breaking
    # This ensures stable ordering even when magnitudes are equal
    sub_magnitudes = magnitudes[indices]
    
    # Create sorting key: (magnitude, index) pairs
    # Use a tuple sort: first by magnitude, then by index for ties
    sort_keys = sub_magnitudes * (n + 1) + indices.float() / (n + 1)
    sort_order = torch.argsort(sort_keys, stable=True)

    # Create assignments tensor - assignments[i] = group for indices[i]
    assignments = torch.empty(n, dtype=torch.long)

    # Split into k contiguous equal-count chunks in sorted order
    # Chunk j gets positions from floor(j*n/k) to floor((j+1)*n/k)
    for g in range(k):
        start_idx = (g * n) // k
        end_idx = ((g + 1) * n) // k

        if end_idx > start_idx:
            # Items at positions [start_idx, end_idx) in sorted order
            # get assigned to group (start_group + g)
            # sort_order[start_idx:end_idx] gives us the original positions
            assignments[sort_order[start_idx:end_idx]] = start_group + g

    return assignments
