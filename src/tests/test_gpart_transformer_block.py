import copy
import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from peft import PeftModel, get_peft_model
from peft.tuners.gpart import GPartConfig, generate_implicit_group_ids
from peft.tuners.gpart.layer import Linear
from peft.tuners.gpart.model import GPartModel


class TinyBlock(nn.Module):
    def __init__(self, hidden: int = 4):
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=True)
        self.value = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x):
        return self.value(torch.tanh(self.query(x)))


class TinyTransformer(nn.Module):
    def __init__(self, block_count: int = 3, hidden: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TinyBlock(hidden=hidden) for _ in range(block_count)]
        )
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ScalarBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.query(x)


class ScalarTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ScalarBlock(), ScalarBlock()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def make_block_model(
    *,
    d: int = 10,
    backend: str = "implicit_stateless_v1",
    bias: str = "none",
    init_bound: float = 0.2,
    block_count: int = 3,
):
    return get_peft_model(
        TinyTransformer(block_count=block_count),
        GPartConfig(
            d=d,
            target_modules=["query", "value"],
            partition_scope="transformer_block",
            assignment_backend=backend,
            bias=bias,
            init_bound=init_bound,
            proj_seed=17,
        ),
    )


def block_layers(model):
    return [
        layer
        for layer in model.base_model._gpart_layers["default"]
        if isinstance(layer, Linear)
    ]


def explicit_layer_output(layer, x, adapter="default"):
    base = layer.get_base_layer()
    theta, scales = layer._get_theta_and_scales(adapter)
    theta = theta.to(device=base.weight.device, dtype=base.weight.dtype)
    scales = scales.to(device=base.weight.device, dtype=base.weight.dtype)
    bias_numel = (
        base.bias.numel()
        if layer._gpart_update_bias[adapter] and base.bias is not None
        else 0
    )
    if layer._gpart_assignment_backend[adapter] == "implicit_stateless_v1":
        group_ids = generate_implicit_group_ids(
            start_offset=layer._gpart_param_offset[adapter],
            numel=base.weight.numel() + bias_numel,
            d=layer._gpart_d[adapter],
            proj_seed=layer._gpart_proj_seed[adapter],
            device=base.weight.device,
        )
    else:
        group_ids = layer.gpart_indices[adapter].to(base.weight.device)
    delta = (theta * scales).index_select(0, group_ids)
    weight_numel = base.weight.numel()
    delta_weight = delta[:weight_numel].view_as(base.weight)
    delta_bias = delta[weight_numel:] if bias_numel else None
    return base(x) + F.linear(x, delta_weight, delta_bias)


def test_config_defaults_validation_and_legacy_reload(tmp_path):
    assert GPartConfig().partition_scope == "global"
    with pytest.raises(ValueError, match="partition scope"):
        GPartConfig(partition_scope="invalid")
    with pytest.raises(ValueError, match="only supported"):
        GPartConfig(
            partition_scope="transformer_block",
            projection_type="fastfood",
        )

    config = GPartConfig(target_modules=["query"])
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "adapter_config.json"
    payload = json.loads(config_path.read_text())
    payload.pop("partition_scope")
    config_path.write_text(json.dumps(payload))
    assert GPartConfig.from_pretrained(tmp_path).partition_scope == "global"


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        (
            "roberta.encoder.layer.7.attention.self.query",
            None,
            "roberta.encoder.layer.7",
        ),
        ("model.layers.10.self_attn.q_proj", None, "model.layers.10"),
        ("transformer.h.2.attn.c_attn", None, "transformer.h.2"),
        (
            "encoder.block.4.layer.0.SelfAttention.q",
            None,
            "encoder.block.4",
        ),
        ("encoder.stages.3.query", "stages", "encoder.stages.3"),
    ],
)
def test_transformer_block_resolution(path, pattern, expected):
    assert GPartModel._resolve_transformer_block_id(path, pattern) == expected


def test_natural_block_order_and_equal_budget_manifest():
    model = make_block_model(d=10)
    manifests = model.base_model._gpart_block_manifests["default"]

    assert [manifest.block_id for manifest in manifests] == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
    ]
    assert [manifest.d_block for manifest in manifests] == [4, 3, 3]
    assert [
        (manifest.theta_start, manifest.theta_end) for manifest in manifests
    ] == [(0, 4), (4, 7), (7, 10)]
    assert [manifest.proj_seed for manifest in manifests] == [17, 18, 19]
    assert sum(manifest.d_block for manifest in manifests) == 10
    assert model.base_model.gpart_theta_d["default"].numel() == 10

    sorted_ids = sorted(
        ["blocks.10", "blocks.2", "blocks.1"],
        key=GPartModel._natural_path_key,
    )
    assert sorted_ids == ["blocks.1", "blocks.2", "blocks.10"]


def test_layers_pattern_override_and_selected_active_blocks():
    class WrappedTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyTransformer()

        def forward(self, x):
            return self.encoder(x)

    model = get_peft_model(
        WrappedTransformer(),
        GPartConfig(
            d=6,
            target_modules=["query"],
            layers_to_transform=[1, 2],
            layers_pattern="blocks",
            partition_scope="transformer_block",
        ),
    )
    manifests = model.base_model._gpart_block_manifests["default"]
    assert [manifest.block_id for manifest in manifests] == [
        "encoder.blocks.1",
        "encoder.blocks.2",
    ]
    assert [manifest.d_block for manifest in manifests] == [3, 3]


def test_unresolved_target_and_invalid_allocations_fail_clearly():
    with pytest.raises(ValueError, match="Could not resolve.*classifier"):
        get_peft_model(
            TinyTransformer(),
            GPartConfig(
                d=2,
                target_modules=["classifier"],
                partition_scope="transformer_block",
            ),
        )

    with pytest.raises(ValueError, match="at least one coordinate"):
        make_block_model(d=2)

    with pytest.raises(ValueError, match="allocated_width=2.*adapted_dimension=1"):
        get_peft_model(
            ScalarTransformer(),
            GPartConfig(
                d=4,
                target_modules=["query"],
                partition_scope="transformer_block",
            ),
        )


@pytest.mark.parametrize("backend", ["legacy_streaming", "implicit_stateless_v1"])
def test_block_local_offsets_assignments_and_scales(backend):
    model = make_block_model(d=10, backend=backend, bias="all")
    base_model = model.base_model
    scales = base_model.gpart_global_scales["default"].cpu()

    for manifest in base_model._gpart_block_manifests["default"]:
        expected_offset = 0
        ids = []
        for layer in manifest.layers:
            assert layer._gpart_param_offset["default"] == expected_offset
            assert layer._gpart_d["default"] == manifest.d_block
            assert layer._gpart_proj_seed["default"] == manifest.proj_seed
            base = layer.get_base_layer()
            layer_numel = base.weight.numel() + base.bias.numel()
            if backend == "implicit_stateless_v1":
                layer_ids = generate_implicit_group_ids(
                    expected_offset,
                    layer_numel,
                    manifest.d_block,
                    manifest.proj_seed,
                    "cpu",
                )
                assert "default" not in layer.gpart_indices
            else:
                layer_ids = layer.gpart_indices["default"].cpu().long()
            ids.append(layer_ids)
            expected_offset += layer_numel

        all_ids = torch.cat(ids)
        assert int(all_ids.min()) >= 0
        assert int(all_ids.max()) < manifest.d_block
        counts = torch.bincount(all_ids, minlength=manifest.d_block).clamp_min(1)
        expected_scales = counts.float().rsqrt()
        torch.testing.assert_close(
            scales[manifest.theta_start : manifest.theta_end],
            expected_scales,
        )


def test_signed_magnitude_grouping_is_independent_per_block():
    model = get_peft_model(
        TinyTransformer(),
        GPartConfig(
            d=10,
            target_modules=["query", "value"],
            partition_scope="transformer_block",
            grouping_strategy="signed_magnitude",
            assignment_backend="legacy_streaming",
            bias="all",
            proj_seed=17,
        ),
    )
    base_model = model.base_model

    for manifest in base_model._gpart_block_manifests["default"]:
        values = base_model._collect_param_values(
            manifest.layers,
            include_bias=True,
        )
        expected = base_model.generate_assignments(
            total_params=manifest.total_params,
            d=manifest.d_block,
            proj_seed=manifest.proj_seed,
            strategy="signed_magnitude",
            params_values=values,
        )
        actual = torch.cat(
            [layer.gpart_indices["default"].cpu() for layer in manifest.layers]
        )
        assert torch.equal(actual, expected)


def test_forward_and_gradients_use_only_the_block_theta_slice():
    torch.manual_seed(31)
    model = make_block_model(d=10, bias="all")
    layer = model.base_model._gpart_block_manifests["default"][1].layers[0]
    theta = model.base_model.gpart_theta_d["default"]

    x_actual = torch.randn(3, 4, requires_grad=True)
    x_reference = x_actual.detach().clone().requires_grad_(True)
    output_actual = layer(x_actual)
    output_reference = explicit_layer_output(layer, x_reference)
    torch.testing.assert_close(output_actual, output_reference)

    grad_actual = torch.autograd.grad(output_actual.sum(), theta)[0]
    grad_reference = torch.autograd.grad(output_reference.sum(), theta)[0]
    torch.testing.assert_close(grad_actual, grad_reference)

    start = layer._gpart_theta_start["default"]
    end = layer._gpart_theta_end["default"]
    assert torch.count_nonzero(grad_actual[:start]) == 0
    assert torch.count_nonzero(grad_actual[end:]) == 0
    assert torch.count_nonzero(grad_actual[start:end]) > 0


def test_zero_init_merge_disable_save_reload_and_adapter_deletion(tmp_path):
    torch.manual_seed(41)
    base = TinyTransformer()
    reference = copy.deepcopy(base)
    reload_base = copy.deepcopy(base)
    model = get_peft_model(
        base,
        GPartConfig(
            d=9,
            target_modules=["query", "value"],
            partition_scope="transformer_block",
            assignment_backend="implicit_stateless_v1",
            proj_seed=23,
        ),
    )
    x = torch.randn(2, 4)
    torch.testing.assert_close(model(x), reference(x))

    model.base_model.gpart_theta_d["default"].data.uniform_(-0.1, 0.1)
    unmerged = model(x)
    model.merge_adapter()
    torch.testing.assert_close(model(x), unmerged, rtol=1e-5, atol=1e-6)
    model.unmerge_adapter()
    torch.testing.assert_close(model(x), unmerged, rtol=1e-5, atol=1e-6)

    with model.disable_adapter():
        torch.testing.assert_close(model(x), reference(x))

    model.save_pretrained(tmp_path)
    loaded = PeftModel.from_pretrained(reload_base, tmp_path)
    torch.testing.assert_close(loaded(x), unmerged)
    assert (
        loaded.peft_config["default"].partition_scope
        == "transformer_block"
    )

    model.add_adapter(
        "other",
        GPartConfig(
            d=6,
            target_modules=["query", "value"],
            partition_scope="transformer_block",
            assignment_backend="implicit_stateless_v1",
            proj_seed=99,
        ),
    )
    assert "other" in model.base_model._gpart_block_manifests
    model.delete_adapter("other")
    assert "other" not in model.base_model._gpart_block_manifests
    assert all(
        "other" not in layer._gpart_theta_start
        for layer in block_layers(model)
    )


def test_runtime_parameter_count_and_device_dtype_movement():
    model = make_block_model(d=10)
    savable, _ = model.base_model.get_nb_savable_parameters("default")
    assert savable == 10

    model = model.to(dtype=torch.float64)
    assert model.base_model.gpart_theta_d["default"].dtype == torch.float64
    assert model.base_model.gpart_global_scales["default"].dtype == torch.float64
    output = model(torch.randn(2, 4, dtype=torch.float64))
    assert output.dtype == torch.float64
