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

"""Matrix-free Fastfood projections used by the GPart DID baseline.

For an intrinsic vector of length ``d``, the implementation pads to the next
power-of-two block length ``L`` and stacks independently sampled Fastfood
blocks until the flattened adapted parameter space of length ``D`` is filled:

    P theta = [H G_b Pi_b H B_b pad(theta)]_b[:D].

``H`` is the unnormalized Walsh-Hadamard matrix. Both modes scale the result by
``1 / sqrt(D * L)``. With ``isometric=True``, every projected entry has
variance ``1 / D`` and ``E[P.T @ P] = I``. With ``isometric=False``, the
intrinsic coordinates are first multiplied by fixed unequal positive gains
whose root-mean-square is one, giving an RMS-matched anisotropic projection.
The normalized Fastfood projection is an expected/approximate isometry, not
the exact isometry of the normalized GPart partition matrix.
"""

from __future__ import annotations

import math

import torch


_FASTFOOD_ANISOTROPY_GAIN_RATIO = 4.0
_FASTFOOD_ANISOTROPY_SEED_OFFSET = 0x6A09E667F3BCC909
_TORCH_SEED_MODULUS = (1 << 63) - 1


def fastfood_block_size(d: int) -> int:
    """Return the smallest power of two greater than or equal to ``d``."""

    d = int(d)
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    return 1 << (d - 1).bit_length()


def fastfood_num_blocks(total_params: int, block_size: int) -> int:
    """Return the number of blocks needed to cover ``total_params``."""

    total_params = int(total_params)
    block_size = int(block_size)
    if total_params <= 0:
        raise ValueError(f"total_params must be positive, got {total_params}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return (total_params + block_size - 1) // block_size


def generate_fastfood_state(
    total_params: int,
    d: int,
    proj_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate fixed Fastfood factors on CPU without consuming global RNG.

    Returns:
        A tuple ``(B, G, permutation)`` with shapes ``(K, L)``. ``B`` is
        int8 Rademacher state, ``G`` is FP32 standard-normal state, and the
        permutation is int32. The compact dtypes keep the RoBERTa-base Q/V
        runtime state at roughly 122 MiB.
    """

    block_size = fastfood_block_size(d)
    num_blocks = fastfood_num_blocks(total_params, block_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(proj_seed))

    signs = torch.empty((num_blocks, block_size), dtype=torch.int8)
    gaussian = torch.empty((num_blocks, block_size), dtype=torch.float32)
    permutation = torch.empty((num_blocks, block_size), dtype=torch.int32)

    for block in range(num_blocks):
        signs[block] = (
            torch.randint(
                0,
                2,
                (block_size,),
                generator=generator,
                dtype=torch.int8,
            )
            .mul_(2)
            .sub_(1)
        )
        gaussian[block] = torch.randn(
            block_size,
            generator=generator,
            dtype=torch.float32,
        )
        permutation[block] = torch.randperm(
            block_size,
            generator=generator,
            dtype=torch.int32,
        )

    return signs, gaussian, permutation


def generate_fastfood_directional_gains(
    d: int,
    proj_seed: int,
) -> torch.Tensor:
    """Generate deterministic RMS-one anisotropic gains on CPU.

    The positive gains are logarithmically spaced with an exact maximum to
    minimum ratio of four, normalized to unit RMS, and shuffled with a
    domain-separated projection seed. The separate generator leaves both the
    global RNG and the existing Fastfood factor stream unchanged.
    """

    d = int(d)
    if d < 2:
        raise ValueError(
            "RMS-preserving Fastfood anisotropy requires d >= 2, "
            f"got d={d}"
        )

    half_log_ratio = 0.5 * math.log(_FASTFOOD_ANISOTROPY_GAIN_RATIO)
    gains = torch.linspace(
        -half_log_ratio,
        half_log_ratio,
        d,
        dtype=torch.float64,
    ).exp_()
    gains.div_(gains.square().mean().sqrt())

    generator = torch.Generator(device="cpu")
    anisotropy_seed = (
        int(proj_seed) + _FASTFOOD_ANISOTROPY_SEED_OFFSET
    ) % _TORCH_SEED_MODULUS
    generator.manual_seed(anisotropy_seed)
    order = torch.randperm(d, generator=generator)
    return gains[order].float()


def walsh_hadamard_transform(value: torch.Tensor) -> torch.Tensor:
    """Apply an unnormalized Walsh-Hadamard transform on the last axis."""

    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError(
            f"Walsh-Hadamard width must be a positive power of two, got {width}"
        )

    result = value
    half = 1
    while half < width:
        paired = result.reshape(*result.shape[:-1], -1, 2, half)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        result = torch.stack((left + right, left - right), dim=-2).reshape(
            *result.shape
        )
        half *= 2
    return result


def _validate_fastfood_state(
    theta: torch.Tensor,
    signs: torch.Tensor,
    gaussian: torch.Tensor,
    permutation: torch.Tensor,
    total_params: int,
) -> int:
    if signs.ndim != 2:
        raise ValueError(f"Fastfood signs must be rank 2, got shape {signs.shape}")
    if gaussian.shape != signs.shape or permutation.shape != signs.shape:
        raise ValueError(
            "Fastfood signs, Gaussian factors, and permutations must have "
            f"identical shapes, got {signs.shape}, {gaussian.shape}, "
            f"{permutation.shape}"
        )

    block_size = signs.shape[1]
    if block_size != fastfood_block_size(theta.numel()):
        raise ValueError(
            f"Fastfood block size {block_size} does not match theta length "
            f"{theta.numel()}"
        )
    if signs.shape[0] != fastfood_num_blocks(total_params, block_size):
        raise ValueError(
            f"Fastfood state has {signs.shape[0]} blocks, but "
            f"{fastfood_num_blocks(total_params, block_size)} are required"
        )
    return block_size


def _projection_scale(total_params: int, block_size: int) -> float:
    return 1.0 / math.sqrt(int(total_params) * int(block_size))


def _apply_directional_gains(
    theta: torch.Tensor,
    directional_gains: torch.Tensor | None,
    isometric: bool,
) -> torch.Tensor:
    if isometric:
        return theta
    if directional_gains is None:
        raise ValueError(
            "directional_gains are required for non-isometric Fastfood"
        )
    if directional_gains.shape != theta.shape:
        raise ValueError(
            "Fastfood directional gains must match theta shape, got "
            f"{directional_gains.shape} and {theta.shape}"
        )
    return theta * directional_gains.to(device=theta.device, dtype=theta.dtype)


def fastfood_project_slice(
    theta: torch.Tensor,
    signs: torch.Tensor,
    gaussian: torch.Tensor,
    permutation: torch.Tensor,
    *,
    directional_gains: torch.Tensor | None = None,
    total_params: int,
    start_offset: int,
    numel: int,
    isometric: bool,
) -> torch.Tensor:
    """Project ``theta`` and return a contiguous slice of the global result."""

    total_params = int(total_params)
    start_offset = int(start_offset)
    numel = int(numel)
    block_size = _validate_fastfood_state(
        theta,
        signs,
        gaussian,
        permutation,
        total_params,
    )
    if start_offset < 0 or numel < 0 or start_offset + numel > total_params:
        raise ValueError(
            f"Invalid Fastfood slice [{start_offset}, {start_offset + numel}) "
            f"for total_params={total_params}"
        )
    if numel == 0:
        return theta.new_empty(0)

    projected_theta = _apply_directional_gains(
        theta,
        directional_gains,
        isometric,
    )
    first_block = start_offset // block_size
    final_block = (start_offset + numel - 1) // block_size
    block_slice = slice(first_block, final_block + 1)
    num_blocks = final_block - first_block + 1

    padded = theta.new_zeros(block_size)
    padded[: theta.numel()] = projected_theta
    work = padded.unsqueeze(0).expand(num_blocks, -1)
    work = work * signs[block_slice].to(device=theta.device, dtype=theta.dtype)
    work = walsh_hadamard_transform(work)
    block_permutation = permutation[block_slice].to(device=theta.device)
    work = torch.gather(work, dim=-1, index=block_permutation)
    work = work * gaussian[block_slice].to(
        device=theta.device,
        dtype=theta.dtype,
    )
    work = walsh_hadamard_transform(work)
    work = work * _projection_scale(total_params, block_size)

    relative_start = start_offset - first_block * block_size
    return work.reshape(-1)[relative_start : relative_start + numel]


def fastfood_project_transpose_slice(
    grad_slice: torch.Tensor,
    signs: torch.Tensor,
    gaussian: torch.Tensor,
    permutation: torch.Tensor,
    *,
    directional_gains: torch.Tensor | None = None,
    theta_numel: int,
    total_params: int,
    start_offset: int,
    isometric: bool,
) -> torch.Tensor:
    """Apply the transpose of a sliced global Fastfood projection."""

    theta_reference = grad_slice.new_empty(int(theta_numel))
    block_size = _validate_fastfood_state(
        theta_reference,
        signs,
        gaussian,
        permutation,
        total_params,
    )
    start_offset = int(start_offset)
    numel = grad_slice.numel()
    if start_offset < 0 or start_offset + numel > total_params:
        raise ValueError(
            f"Invalid Fastfood transpose slice [{start_offset}, "
            f"{start_offset + numel}) for total_params={total_params}"
        )
    if numel == 0:
        return grad_slice.new_zeros(theta_numel)

    first_block = start_offset // block_size
    final_block = (start_offset + numel - 1) // block_size
    block_slice = slice(first_block, final_block + 1)
    num_blocks = final_block - first_block + 1
    relative_start = start_offset - first_block * block_size

    work = grad_slice.new_zeros((num_blocks, block_size))
    work.reshape(-1)[relative_start : relative_start + numel] = grad_slice
    work = walsh_hadamard_transform(work)
    work = work * gaussian[block_slice].to(
        device=grad_slice.device,
        dtype=grad_slice.dtype,
    )

    block_permutation = permutation[block_slice].to(device=grad_slice.device)
    unpermuted = torch.zeros_like(work)
    unpermuted.scatter_(dim=-1, index=block_permutation, src=work)
    work = walsh_hadamard_transform(unpermuted)
    work = work * signs[block_slice].to(
        device=grad_slice.device,
        dtype=grad_slice.dtype,
    )
    work = work * _projection_scale(total_params, block_size)
    grad_theta = work.sum(dim=0)[:theta_numel]
    return _apply_directional_gains(
        grad_theta,
        directional_gains,
        isometric,
    )


def fastfood_runtime_bytes(
    signs: torch.Tensor,
    gaussian: torch.Tensor,
    permutation: torch.Tensor,
) -> int:
    """Return the fixed runtime-state footprint in bytes."""

    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (signs, gaussian, permutation)
    )
