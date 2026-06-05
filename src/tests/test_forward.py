import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(module, method_name, x, warmup=20, iters=100, backward=False):
    fn = getattr(module, method_name)

    for _ in range(warmup):
        out = fn(x)
        if backward:
            loss = out.float().mean()
            loss.backward()
            module.zero_grad(set_to_none=True)
    synchronize(x.device)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn(x)
        if backward:
            loss = out.float().mean()
            loss.backward()
            module.zero_grad(set_to_none=True)
        synchronize(x.device)
        end = time.perf_counter()
        times.append(end - start)

    t = torch.tensor(times, dtype=torch.float64)
    return {
        "mean_ms": t.mean().item() * 1000,
        "median_ms": t.median().item() * 1000,
        "std_ms": t.std(unbiased=False).item() * 1000,
        "min_ms": t.min().item() * 1000,
        "max_ms": t.max().item() * 1000,
    }


def print_stats(title, stats):
    print(title)
    print(f"  mean   : {stats['mean_ms']:.3f} ms")
    print(f"  median : {stats['median_ms']:.3f} ms")
    print(f"  std    : {stats['std_ms']:.3f} ms")
    print(f"  min/max: {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms")
    print()


class GPartLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, base_weight, base_bias, theta, indices, scales, dropout_p, training
    ):
        # x: [..., in_features]
        # base_weight: [out_features, in_features]
        # base_bias: [out_features] or None
        # theta: [d]
        # indices/scales: [num_weight_params + num_bias_params]

        delta_flat = theta.index_select(0, indices) * scales

        w_numel = base_weight.numel()
        delta_w = delta_flat[:w_numel].view_as(base_weight)

        if training and dropout_p > 0.0:
            keep_prob = 1.0 - dropout_p
            mask = (torch.rand_like(delta_w) < keep_prob).to(delta_w.dtype) / keep_prob
            delta_w = delta_w * mask
        else:
            mask = torch.ones_like(delta_w)

        eff_weight = base_weight + delta_w

        has_bias = base_bias is not None
        if has_bias:
            b_numel = base_bias.numel()
            delta_b = delta_flat[w_numel : w_numel + b_numel].view_as(base_bias)
            eff_bias = base_bias + delta_b
        else:
            eff_bias = None

        out = F.linear(x, eff_weight, eff_bias)

        ctx.save_for_backward(x, eff_weight, indices, scales, mask)
        ctx.has_bias = has_bias
        ctx.weight_shape = tuple(base_weight.shape)
        ctx.theta_numel = theta.numel()
        ctx.theta_dtype = theta.dtype
        ctx.theta_device = theta.device
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, eff_weight, indices, scales, mask = ctx.saved_tensors
        out_features, in_features = ctx.weight_shape

        x2 = x.reshape(-1, in_features)
        go2 = grad_output.reshape(-1, out_features).to(eff_weight.dtype)

        need_grad_x = ctx.needs_input_grad[0]
        need_grad_w = ctx.needs_input_grad[1]
        need_grad_b = ctx.needs_input_grad[2] and ctx.has_bias
        need_grad_theta = ctx.needs_input_grad[3]

        grad_x = None
        grad_base_weight = None
        grad_base_bias = None
        grad_theta = None

        if need_grad_x:
            grad_x = go2.matmul(eff_weight).view_as(x)

        grad_eff_weight = None
        grad_eff_bias = None

        if need_grad_w or need_grad_theta:
            grad_eff_weight = go2.transpose(0, 1).matmul(x2)
            if need_grad_w:
                grad_base_weight = grad_eff_weight

        if ctx.has_bias and (need_grad_b or need_grad_theta):
            grad_eff_bias = go2.sum(dim=0)
            if need_grad_b:
                grad_base_bias = grad_eff_bias

        if need_grad_theta:
            # Because dropout was applied only to delta_w in forward,
            # the gradient flowing back to the pre-dropout delta_w
            # must be masked by the same dropout mask.
            grad_delta_w = grad_eff_weight * mask

            flat_parts = [grad_delta_w.reshape(-1)]
            if ctx.has_bias:
                flat_parts.append(grad_eff_bias.reshape(-1))

            grad_delta_flat = torch.cat(flat_parts, dim=0)

            src = grad_delta_flat * scales.to(grad_delta_flat.dtype)
            grad_theta = torch.zeros(
                ctx.theta_numel,
                device=ctx.theta_device,
                dtype=src.dtype,
            )
            grad_theta.index_add_(0, indices, src)
            grad_theta = grad_theta.to(ctx.theta_dtype)

        return (
            grad_x,
            grad_base_weight,
            grad_base_bias,
            grad_theta,
            None,
            None,
            None,
            None,
        )


class DummyGPartLinear(nn.Module):
    def __init__(
        self,
        in_features=4096,
        out_features=11008,
        d=16384,
        gpart_dropout=0.0,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.base_layer = nn.Linear(
            in_features, out_features, bias=True, device=device, dtype=dtype
        )
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False

        param_count = out_features * in_features + out_features

        self._gpart_theta_d_ref = nn.ParameterDict(
            {
                "default": nn.Parameter(
                    torch.zeros(d, device=device, dtype=dtype)
                )
            }
        )

        indices = torch.randint(0, d, (param_count,), device=device, dtype=torch.long)
        counts = torch.bincount(indices, minlength=d).clamp_min(1)
        scales = (1.0 / torch.sqrt(counts.float()))[indices].to(
            device=device, dtype=dtype
        )

        self.gpart_indices = {"default": indices}
        self.gpart_scales = {"default": scales}
        self.gpart_dropout = {
            "default": (
                nn.Dropout(p=gpart_dropout) if gpart_dropout > 0 else nn.Identity()
            )
        }

    def forward_old(self, x):
        previous_dtype = x.dtype
        result = self.base_layer(x)

        for active_adapter in self.active_adapters:
            theta_d = self._gpart_theta_d_ref[active_adapter]
            indices = self.gpart_indices[active_adapter]
            scales = self.gpart_scales[active_adapter]
            dropout = self.gpart_dropout[active_adapter]

            delta_flat = theta_d[indices] * scales
            weight_shape = self.base_layer.weight.shape
            delta_weight = delta_flat[: weight_shape.numel()].view(weight_shape)
            delta_weight = dropout(delta_weight)

            delta_bias = None
            if self.base_layer.bias is not None:
                bias_start = weight_shape.numel()
                bias_end = bias_start + self.base_layer.bias.numel()
                delta_bias = delta_flat[bias_start:bias_end].view_as(
                    self.base_layer.bias
                )

            delta_result = F.linear(x.to(delta_weight.dtype), delta_weight, delta_bias)
            result = result + delta_result

        return result.to(previous_dtype)

    def forward(self, x):
        previous_dtype = x.dtype
        base = self.base_layer
        x_in = x.to(base.weight.dtype)

        eff_weight = base.weight
        eff_bias = base.bias

        for active_adapter in self.active_adapters:
            theta = self._gpart_theta_d_ref[active_adapter]
            indices = self.gpart_indices[active_adapter]
            scales = self.gpart_scales[active_adapter]
            dropout = self.gpart_dropout[active_adapter]

            delta_flat = theta[indices] * scales

            w_numel = base.weight.numel()
            delta_w = delta_flat[:w_numel].view_as(base.weight)
            delta_w = dropout(delta_w)
            eff_weight = eff_weight + delta_w

            if base.bias is not None:
                delta_b = delta_flat[w_numel : w_numel + base.bias.numel()].view_as(
                    base.bias
                )
                eff_bias = eff_bias + delta_b if eff_bias is not None else delta_b

        out = F.linear(x_in, eff_weight, eff_bias)
        return out.to(previous_dtype)

    def forward_custom(self, x):
        previous_dtype = x.dtype
        base = self.base_layer
        x_in = x.to(base.weight.dtype)

        out = GPartLinearFn.apply(
            x_in,
            base.weight,
            base.bias,
            self._gpart_theta_d_ref["default"],
            self.gpart_indices["default"],
            self.gpart_scales["default"],
            (
                self.gpart_dropout["default"].p
                if isinstance(self.gpart_dropout["default"], nn.Dropout)
                else 0.0
            ),
            self.training,
        )
        return out.to(previous_dtype)


class DummyUniLoRALinear(nn.Module):
    def __init__(
        self,
        in_features=4096,
        out_features=11008,
        r=16,
        theta_d_length=16384,
        unilora_dropout=0.0,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        super().__init__()
        self.base_layer = nn.Linear(
            in_features, out_features, bias=True, device=device, dtype=dtype
        )
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.mode = "Full"

        self.unilora_theta_d = nn.ParameterDict(
            {
                "default": nn.Parameter(
                    torch.empty(theta_d_length, device=device, dtype=dtype).uniform_(
                        -0.02, 0.02
                    )
                )
            }
        )

        idx_A = torch.randint(
            0, theta_d_length, (r, in_features), device=device, dtype=torch.long
        )
        idx_B = torch.randint(
            0, theta_d_length, (out_features, r), device=device, dtype=torch.long
        )

        all_idx = torch.cat([idx_A.reshape(-1), idx_B.reshape(-1)], dim=0)
        counts = torch.bincount(all_idx, minlength=theta_d_length).clamp_min(1)
        inv_sqrt = 1.0 / torch.sqrt(counts.float())

        scale_A = inv_sqrt[idx_A].to(device=device, dtype=dtype)
        scale_B = inv_sqrt[idx_B].to(device=device, dtype=dtype)

        self.unilora_indices_A = {"default": idx_A}
        self.unilora_indices_B = {"default": idx_B}
        self.unilora_scales_A = {"default": scale_A}
        self.unilora_scales_B = {"default": scale_B}
        self.unilora_dropout = {
            "default": (
                nn.Dropout(p=unilora_dropout) if unilora_dropout > 0 else nn.Identity()
            )
        }

        fixed_A_fp32 = torch.empty(r, in_features, device=device, dtype=torch.float32)
        fixed_B_fp32 = torch.empty(out_features, r, device=device, dtype=torch.float32)
        nn.init.orthogonal_(fixed_A_fp32)
        nn.init.orthogonal_(fixed_B_fp32)
        self.fixed_A = {"default": fixed_A_fp32.to(dtype=dtype)}
        self.fixed_B = {"default": fixed_B_fp32.to(dtype=dtype)}

    def _get_lora_matrices(self, adapter):
        base = self.base_layer
        target_device = base.weight.device
        target_dtype = base.weight.dtype

        idx_A = self.unilora_indices_A[adapter]
        idx_B = self.unilora_indices_B[adapter]
        scale_A = self.unilora_scales_A[adapter].to(
            device=target_device, dtype=target_dtype
        )
        scale_B = self.unilora_scales_B[adapter].to(
            device=target_device, dtype=target_dtype
        )
        theta = self.unilora_theta_d[adapter].to(
            device=target_device, dtype=target_dtype
        )

        if self.mode == "B_only":
            A = self.fixed_A[adapter].to(device=target_device, dtype=target_dtype)
            B = theta[idx_B] * scale_B
        elif self.mode == "A_only":
            A = theta[idx_A] * scale_A
            B = self.fixed_B[adapter].to(device=target_device, dtype=target_dtype)
        else:
            A = theta[idx_A] * scale_A
            B = theta[idx_B] * scale_B

        return A, B

    def forward(self, x):
        previous_dtype = x.dtype
        base = self.base_layer
        x = x.to(device=base.weight.device, dtype=base.weight.dtype)

        result = base(x)

        for active_adapter in self.active_adapters:
            A, B = self._get_lora_matrices(active_adapter)
            dropout = self.unilora_dropout[active_adapter]
            x_dropout = dropout(x.to(A.dtype))
            delta_result = F.linear(F.linear(x_dropout, A), B)
            result = result + delta_result

        return result.to(previous_dtype)


@torch.no_grad()
def compare_outputs(gpart, x):
    y_new = gpart.forward(x)
    y_custom = gpart.forward_custom(x)
    diff = (y_new.float() - y_custom.float()).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


@torch.no_grad()
def compare_more(y_ref, y_test):
    diff = (y_ref.float() - y_test.float()).abs()
    rel = diff / y_ref.float().abs().clamp_min(1e-8)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
        "ref_abs_max": y_ref.float().abs().max().item(),
    }


def compare_theta_grad(mod, x, method_a="forward", method_b="forward_custom"):
    mod.zero_grad(set_to_none=True)
    out_a = getattr(mod, method_a)(x)
    out_a.float().mean().backward()
    grad_a = mod._gpart_theta_d_ref["default"].grad.detach().clone()

    mod.zero_grad(set_to_none=True)
    out_b = getattr(mod, method_b)(x)
    out_b.float().mean().backward()
    grad_b = mod._gpart_theta_d_ref["default"].grad.detach().clone()

    diff = (grad_a.float() - grad_b.float()).abs()
    rel = diff / grad_a.float().abs().clamp_min(1e-8)
    return {
        "grad_max_abs": diff.max().item(),
        "grad_mean_abs": diff.mean().item(),
        "grad_max_rel": rel.max().item(),
        "grad_mean_rel": rel.mean().item(),
    }


def main():
    batch_size = 4
    seq_len = 2048
    in_features = 4096
    out_features = 11008

    d = 16384
    r = 4
    theta_d_length = 16384
    dropout = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)

    # Keep dropout at 0.0 because the custom backward below is the no-dropout version.
    gpart = DummyGPartLinear(
        in_features=in_features,
        out_features=out_features,
        d=d,
        gpart_dropout=dropout,
        dtype=dtype,
        device=device,
    )

    unilora = DummyUniLoRALinear(
        in_features=in_features,
        out_features=out_features,
        r=r,
        theta_d_length=theta_d_length,
        unilora_dropout=dropout,
        dtype=dtype,
        device=device,
    )

    # ----------------------------
    # Correctness checks
    # ----------------------------
    print("=== Correctness checks ===")

    correctness = compare_outputs(gpart, x)
    print("Correctness check: GPart autograd forward vs custom-bwd forward")
    print(f"  max_abs_diff : {correctness['max_abs_diff']:.6e}")
    print(f"  mean_abs_diff: {correctness['mean_abs_diff']:.6e}")
    print()

    # 1) Forward-output check in eval mode
    # Best done with dropout disabled or with eval() so no random mask is applied.
    gpart.eval()

    with torch.no_grad():
        y_ref = gpart.forward(x)
        y_custom = gpart.forward_custom(x)

    forward_check = compare_more(y_ref, y_custom)
    print("Forward: autograd vs custom-bwd")
    print(f"  max_abs    : {forward_check['max_abs']:.6e}")
    print(f"  mean_abs   : {forward_check['mean_abs']:.6e}")
    print(f"  max_rel    : {forward_check['max_rel']:.6e}")
    print(f"  mean_rel   : {forward_check['mean_rel']:.6e}")
    print(f"  ref_abs_max: {forward_check['ref_abs_max']:.6e}")
    print()

    # 2) Gradient check
    # Keep eval() here too if you want deterministic behavior with dropout layers present.
    grad_check = compare_theta_grad(
        gpart, x, method_a="forward", method_b="forward_custom"
    )
    print("Theta grad: autograd vs custom-bwd")
    print(f"  grad_max_abs : {grad_check['grad_max_abs']:.6e}")
    print(f"  grad_mean_abs: {grad_check['grad_mean_abs']:.6e}")
    print(f"  grad_max_rel : {grad_check['grad_max_rel']:.6e}")
    print(f"  grad_mean_rel: {grad_check['grad_mean_rel']:.6e}")
    print()

    # Forward-only
    gpart.eval()
    unilora.eval()

    gpart_old_fwd = benchmark(
        gpart, "forward_old", x, warmup=20, iters=100, backward=False
    )
    gpart_new_fwd = benchmark(gpart, "forward", x, warmup=20, iters=100, backward=False)
    gpart_custom_fwd = benchmark(
        gpart, "forward_custom", x, warmup=20, iters=100, backward=False
    )
    unilora_fwd = benchmark(unilora, "forward", x, warmup=20, iters=100, backward=False)

    print("=== Forward only ===")
    print_stats("GPart old", gpart_old_fwd)
    print_stats("GPart new (autograd)", gpart_new_fwd)
    print_stats("GPart new (custom bwd)", gpart_custom_fwd)
    print_stats("UniLoRA", unilora_fwd)

    print("Forward speedups")
    print(
        f"  GPart old / GPart new-autograd : {gpart_old_fwd['mean_ms'] / gpart_new_fwd['mean_ms']:.3f}x"
    )
    print(
        f"  GPart old / GPart custom-bwd   : {gpart_old_fwd['mean_ms'] / gpart_custom_fwd['mean_ms']:.3f}x"
    )
    print(
        f"  GPart new-autograd / custom    : {gpart_new_fwd['mean_ms'] / gpart_custom_fwd['mean_ms']:.3f}x"
    )
    print(
        f"  GPart old / UniLoRA : {gpart_old_fwd['mean_ms'] / unilora_fwd['mean_ms']:.3f}x"
    )
    print(
        f"  GPart custom-bwd / UniLoRA     : {gpart_custom_fwd['mean_ms'] / unilora_fwd['mean_ms']:.3f}x"
    )
    print()

    # Training-like benchmark
    gpart_train = copy.deepcopy(gpart).train()
    unilora_train = copy.deepcopy(unilora).train()

    gpart_old_train = benchmark(
        gpart_train, "forward_old", x, warmup=10, iters=50, backward=True
    )
    gpart_new_train = benchmark(
        gpart_train, "forward", x, warmup=10, iters=50, backward=True
    )
    gpart_custom_train = benchmark(
        gpart_train, "forward_custom", x, warmup=10, iters=50, backward=True
    )
    unilora_train_stats = benchmark(
        unilora_train, "forward", x, warmup=10, iters=50, backward=True
    )

    print("=== Forward + backward ===")
    print_stats("GPart old", gpart_old_train)
    print_stats("GPart new (autograd)", gpart_new_train)
    print_stats("GPart new (custom bwd)", gpart_custom_train)
    print_stats("UniLoRA", unilora_train_stats)

    print("Training speedups")
    print(
        f"  GPart old / GPart new-autograd : {gpart_old_train['mean_ms'] / gpart_new_train['mean_ms']:.3f}x"
    )
    print(
        f"  GPart old / GPart custom-bwd   : {gpart_old_train['mean_ms'] / gpart_custom_train['mean_ms']:.3f}x"
    )
    print(
        f"  GPart new-autograd / custom    : {gpart_new_train['mean_ms'] / gpart_custom_train['mean_ms']:.3f}x"
    )
    print(
        f"  GPart old / UniLoRA : {gpart_old_train['mean_ms'] / unilora_train_stats['mean_ms']:.3f}x"
    )
    print(
        f"  GPart custom-bwd / UniLoRA     : {gpart_custom_train['mean_ms'] / unilora_train_stats['mean_ms']:.3f}x"
    )
    print()

    print("Setup")
    print(f"  device      : {device}")
    print(f"  dtype       : {dtype}")
    print(f"  input shape : {tuple(x.shape)}")
    print(f"  weight shape: {(out_features, in_features)}")
    print(f"  GPart d     : {d}")
    print(f"  UniLoRA r   : {r}")
    print(f"  Uni theta_d : {theta_d_length}")


if __name__ == "__main__":
    main()
