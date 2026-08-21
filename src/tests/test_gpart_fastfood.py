import copy
import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft import PeftModel, get_peft_model
from peft.tuners.gpart import GPartConfig
from peft.tuners.gpart.fastfood import (
    fastfood_project_slice,
    fastfood_project_transpose_slice,
    generate_fastfood_directional_gains,
    generate_fastfood_state,
)
from peft.tuners.gpart.layer import Linear, _FastfoodGPartLinearFunction


class TinyModel(nn.Module):
    def __init__(self, hidden=5):
        super().__init__()
        self.l1 = nn.Linear(hidden, 4, bias=True)
        self.l2 = nn.Linear(4, 3, bias=True)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class Conv1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Conv1D(4, 5)

    def forward(self, x):
        return self.proj(x)


def make_model(*, dropout=0.0, init_bound=0.2, bias="all", seed=9):
    return get_peft_model(
        TinyModel(),
        GPartConfig(
            d=7,
            target_modules=["l1", "l2"],
            bias=bias,
            projection_type="fastfood",
            gpart_dropout=dropout,
            init_bound=init_bound,
            proj_seed=seed,
            assignment_backend="stateless",
        ),
    )


def gpart_layers(model):
    return [module for module in model.modules() if isinstance(module, Linear)]


@pytest.mark.parametrize(
    ("deprecated_backend", "replacement"),
    [
        ("legacy_streaming", "materialized"),
        ("implicit_stateless_v1", "stateless"),
    ],
)
def test_config_normalizes_deprecated_assignment_backends(
    deprecated_backend, replacement, tmp_path
):
    with pytest.warns(
        FutureWarning,
        match=rf"assignment_backend='{deprecated_backend}'.*'{replacement}'",
    ):
        config = GPartConfig(
            target_modules=["l1"], assignment_backend=deprecated_backend
        )
    assert config.assignment_backend == replacement

    config.save_pretrained(tmp_path)
    config_path = tmp_path / "adapter_config.json"
    payload = json.loads(config_path.read_text())
    payload["assignment_backend"] = deprecated_backend
    config_path.write_text(json.dumps(payload))
    with pytest.warns(
        FutureWarning,
        match=rf"assignment_backend='{deprecated_backend}'.*'{replacement}'",
    ):
        loaded = GPartConfig.from_pretrained(tmp_path)
    assert loaded.assignment_backend == replacement


def explicit_matrix(
    signs,
    gaussian,
    permutation,
    directional_gains,
    d,
    total_params,
    isometric,
):
    block_size = signs.shape[1]
    hadamard = torch.ones(1, 1, dtype=gaussian.dtype)
    while hadamard.shape[0] < block_size:
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            ),
            dim=0,
        )
    padding = torch.zeros(block_size, d, dtype=gaussian.dtype)
    padding[:d] = torch.eye(d, dtype=gaussian.dtype)
    blocks = []
    for block in range(signs.shape[0]):
        permutation_matrix = torch.eye(block_size, dtype=gaussian.dtype)[
            permutation[block].long()
        ]
        blocks.append(
            hadamard
            @ torch.diag(gaussian[block])
            @ permutation_matrix
            @ hadamard
            @ torch.diag(signs[block].to(gaussian.dtype))
            @ padding
        )
    matrix = torch.cat(blocks, dim=0)[:total_params]
    matrix = matrix / (total_params * block_size) ** 0.5
    if not isometric:
        matrix = matrix @ torch.diag(directional_gains.to(gaussian.dtype))
    return matrix


@pytest.mark.parametrize("isometric", [False, True])
def test_transform_and_transpose_match_explicit_matrix_for_cross_block_slice(isometric):
    d, total_params = 5, 13
    signs, gaussian, permutation = generate_fastfood_state(total_params, d, 7)
    gaussian = gaussian.double()
    theta = torch.randn(d, dtype=torch.double)
    directional_gains = (
        None
        if isometric
        else generate_fastfood_directional_gains(d, 7).double()
    )
    matrix = explicit_matrix(
        signs,
        gaussian,
        permutation,
        directional_gains,
        d,
        total_params,
        isometric,
    )

    actual = fastfood_project_slice(
        theta,
        signs,
        gaussian,
        permutation,
        directional_gains=directional_gains,
        total_params=total_params,
        start_offset=0,
        numel=total_params,
        isometric=isometric,
    )
    torch.testing.assert_close(actual, matrix @ theta)

    start_offset, numel = 6, 6
    grad_slice = torch.randn(numel, dtype=torch.double)
    actual_transpose = fastfood_project_transpose_slice(
        grad_slice,
        signs,
        gaussian,
        permutation,
        directional_gains=directional_gains,
        theta_numel=d,
        total_params=total_params,
        start_offset=start_offset,
        isometric=isometric,
    )
    torch.testing.assert_close(
        actual_transpose,
        matrix[start_offset : start_offset + numel].T @ grad_slice,
    )


def test_normalized_projection_is_isometric_in_expectation():
    d, total_params = 5, 8192
    signs, gaussian, permutation = generate_fastfood_state(total_params, d, 29)
    generator = torch.Generator(device="cpu").manual_seed(31)
    ratios = []
    for _ in range(16):
        theta = torch.randn(d, generator=generator)
        projected = fastfood_project_slice(
            theta,
            signs,
            gaussian,
            permutation,
            total_params=total_params,
            start_offset=0,
            numel=total_params,
            isometric=True,
        )
        ratios.append(float(projected.norm() / theta.norm()))
    assert 0.95 < sum(ratios) / len(ratios) < 1.05


def test_missing_projection_metadata_loads_as_partition(tmp_path):
    config = GPartConfig(target_modules=["l1"])
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "adapter_config.json"
    payload = json.loads(config_path.read_text())
    payload.pop("projection_type")
    config_path.write_text(json.dumps(payload))
    loaded = GPartConfig.from_pretrained(tmp_path)
    assert loaded.projection_type == "partition"


def test_directional_gains_are_deterministic_shuffled_and_rms_matched():
    torch.manual_seed(807)
    expected_state = torch.get_rng_state().clone()
    gains_a = generate_fastfood_directional_gains(101, 3)
    assert torch.equal(expected_state, torch.get_rng_state())

    gains_b = generate_fastfood_directional_gains(101, 3)
    gains_c = generate_fastfood_directional_gains(101, 4)
    torch.testing.assert_close(gains_a, gains_b)
    assert not torch.equal(gains_a, gains_c)
    torch.testing.assert_close(
        torch.sort(gains_a).values,
        torch.sort(gains_c).values,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(gains_a > 0)
    torch.testing.assert_close(gains_a.square().mean(), torch.tensor(1.0))
    torch.testing.assert_close(
        gains_a.max() / gains_a.min(),
        torch.tensor(4.0),
    )


def test_non_isometric_fastfood_rejects_one_intrinsic_dimension():
    with pytest.raises(
        ValueError,
        match="non-isometric Fastfood requires d >= 2",
    ):
        GPartConfig(
            d=1,
            target_modules=["l1"],
            projection_type="fastfood",
            isometric=False,
        )


def test_anisotropic_projection_preserves_rms_but_not_directional_gram():
    d, total_params = 8, 8192
    signs, gaussian, permutation = generate_fastfood_state(total_params, d, 23)
    gains = generate_fastfood_directional_gains(d, 23)
    basis = torch.eye(d)

    isometric_matrix = torch.stack(
        [
            fastfood_project_slice(
                coordinate,
                signs,
                gaussian,
                permutation,
                total_params=total_params,
                start_offset=0,
                numel=total_params,
                isometric=True,
            )
            for coordinate in basis
        ],
        dim=1,
    )
    anisotropic_matrix = torch.stack(
        [
            fastfood_project_slice(
                coordinate,
                signs,
                gaussian,
                permutation,
                directional_gains=gains,
                total_params=total_params,
                start_offset=0,
                numel=total_params,
                isometric=False,
            )
            for coordinate in basis
        ],
        dim=1,
    )
    isometric_gram = isometric_matrix.T @ isometric_matrix
    anisotropic_gram = anisotropic_matrix.T @ anisotropic_matrix

    torch.testing.assert_close(
        anisotropic_gram.trace(),
        isometric_gram.trace(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert anisotropic_gram.diag().max() / anisotropic_gram.diag().min() > 10
    assert not torch.allclose(anisotropic_gram, torch.eye(d), rtol=0.1, atol=0.1)


def test_state_is_deterministic_seeded_and_independent_of_global_rng():
    torch.manual_seed(808)
    expected_state = torch.get_rng_state().clone()
    state_a = generate_fastfood_state(25, 7, 3)
    assert torch.equal(expected_state, torch.get_rng_state())
    state_b = generate_fastfood_state(25, 7, 3)
    state_c = generate_fastfood_state(25, 7, 4)
    assert all(torch.equal(a, b) for a, b in zip(state_a, state_b))
    assert any(not torch.equal(a, c) for a, c in zip(state_a, state_c))
    assert state_a[0].dtype == torch.int8
    assert state_a[1].dtype == torch.float32
    assert state_a[2].dtype == torch.int32


@pytest.mark.parametrize("isometric", [False, True])
def test_custom_backward_gradcheck(isometric):
    torch.manual_seed(5)
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    theta = torch.randn(5, dtype=torch.double, requires_grad=True)
    signs, gaussian, permutation = generate_fastfood_state(19, 5, 11)
    gaussian = gaussian.double()
    directional_gains = (
        torch.ones(1, dtype=torch.double)
        if isometric
        else generate_fastfood_directional_gains(theta.numel(), 11).double()
    )

    def function(x_value, theta_value):
        return _FastfoodGPartLinearFunction.apply(
            x_value,
            theta_value,
            signs,
            gaussian,
            permutation,
            directional_gains,
            19,
            3,
            4,
            3,
            4,
            isometric,
            False,
            0.0,
            False,
        )

    assert torch.autograd.gradcheck(function, (x, theta), fast_mode=True)


def test_zero_initialization_matches_base_and_has_dense_nonzero_gradient():
    torch.manual_seed(12)
    base = TinyModel()
    reference = copy.deepcopy(base)
    model = get_peft_model(
        base,
        GPartConfig(
            d=7,
            target_modules=["l1", "l2"],
            projection_type="fastfood",
            init_bound=0.0,
            proj_seed=4,
        ),
    )
    x = torch.randn(3, 5)
    torch.testing.assert_close(model(x), reference(x), rtol=0.0, atol=0.0)
    model(x).square().sum().backward()
    grad = model.base_model.gpart_theta_d["default"].grad
    assert grad is not None
    assert torch.count_nonzero(grad) == grad.numel()


def test_dropout_backward_preserves_rng_and_matches_direct_autograd_reference():
    torch.manual_seed(33)
    model = make_model(dropout=0.25)
    layer = gpart_layers(model)[0]
    layer.train()
    base = layer.get_base_layer()
    theta = model.base_model.gpart_theta_d["default"]
    x_actual = torch.randn(2, 5, requires_grad=True)
    x_reference = x_actual.detach().clone().requires_grad_(True)
    grad_output = torch.randn(2, 4)

    torch.manual_seed(123)
    actual = layer(x_actual)
    state_before_backward = torch.get_rng_state().clone()
    actual_grads = torch.autograd.grad(actual, (x_actual, theta), grad_output)
    assert torch.equal(state_before_backward, torch.get_rng_state())

    torch.manual_seed(123)
    delta = fastfood_project_slice(
        theta,
        model.base_model.gpart_fastfood_signs["default"],
        model.base_model.gpart_fastfood_gaussian["default"],
        model.base_model.gpart_fastfood_permutation["default"],
        total_params=layer._gpart_total_params["default"],
        start_offset=layer._gpart_param_offset["default"],
        numel=base.weight.numel() + base.bias.numel(),
        isometric=True,
    )
    delta_weight = delta[: base.weight.numel()].view_as(base.weight)
    delta_weight = delta_weight * F.dropout(
        torch.ones_like(delta_weight), p=0.25, training=True
    )
    reference = base(x_reference) + F.linear(
        x_reference, delta_weight, delta[base.weight.numel() :]
    )
    reference_grads = torch.autograd.grad(
        reference, (x_reference, theta), grad_output
    )
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_grads[0], reference_grads[0])
    torch.testing.assert_close(actual_grads[1], reference_grads[1])


def test_save_reload_runtime_state_merge_disable_and_lifecycle(tmp_path):
    torch.manual_seed(101)
    base = TinyModel()
    reload_base = copy.deepcopy(base)
    model = get_peft_model(
        base,
        GPartConfig(
            d=7,
            target_modules=["l1", "l2"],
            bias="all",
            projection_type="fastfood",
            isometric=False,
            proj_seed=14,
        ),
    )
    model.base_model.gpart_theta_d["default"].data.uniform_(-0.2, 0.2)
    x = torch.randn(2, 5)
    expected = model(x)
    with model.disable_adapter():
        disabled = model(x)
    assert not torch.allclose(disabled, expected)

    model.merge_adapter()
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)
    model.unmerge_adapter()
    torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)

    assert model.base_model.get_fixed_runtime_state_bytes() > 0
    assert not any("fastfood" in key for key in model.state_dict())
    model.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "adapter_config.json").read_text())
    assert payload["projection_type"] == "fastfood"
    assert payload["isometric"] is False
    assert payload["proj_seed"] == 14
    torch.testing.assert_close(
        model.base_model.gpart_global_scales["default"],
        generate_fastfood_directional_gains(7, 14),
    )
    loaded = PeftModel.from_pretrained(reload_base, tmp_path)
    torch.testing.assert_close(loaded(x), expected)

    model.add_adapter(
        "partition",
        GPartConfig(
            d=5,
            target_modules=["l1", "l2"],
            projection_type="partition",
            proj_seed=77,
        ),
    )
    assert "partition" in model.base_model.gpart_theta_d
    model.set_adapter("partition")
    model(x)
    model.delete_adapter("partition")
    assert "partition" not in model.base_model.gpart_theta_d


def test_conv1d_and_dtype_movement():
    model = get_peft_model(
        Conv1DModel(),
        GPartConfig(
            d=7,
            target_modules=["proj"],
            bias="all",
            projection_type="fastfood",
            init_bound=0.1,
        ),
    )
    x = torch.randn(2, 3, 5)
    assert model(x).shape == (2, 3, 4)
    model.half()
    assert model.base_model.gpart_fastfood_gaussian["default"].dtype == torch.float32
    assert model(x.half()).dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_determinism_and_mixed_precision():
    torch.manual_seed(55)
    cpu = make_model(init_bound=0.1)
    cuda = copy.deepcopy(cpu).cuda().half()
    x = torch.randn(2, 5)
    first = cuda(x.cuda().half())
    second = cuda(x.cuda().half())
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    first.float().sum().backward()
    assert torch.isfinite(cuda.base_model.gpart_theta_d["default"].grad).all()
