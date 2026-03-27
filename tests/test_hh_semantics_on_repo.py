import os
import sys
import torch

# 让脚本能从仓库根目录运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import moore.utils.mixture_layers as mixture_layers


torch.manual_seed(0)


def orth_error_stats_tensor(basis: torch.Tensor):
    """
    basis: [K, B, D]
    检查每个 sample 上 expert 轴是否正交:
        V_s V_s^T ≈ I_K
    """
    assert basis.dim() == 3, f"expected [K,B,D], got {basis.shape}"

    x = basis.transpose(0, 1)  # [B, K, D]
    gram = torch.matmul(x, x.transpose(1, 2))  # [B, K, K]
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    err = torch.linalg.norm(gram - eye, dim=(1, 2))  # [B]

    return {
        "mean": err.mean().item(),
        "max": err.max().item(),
        "p95": torch.quantile(err, 0.95).item(),
    }


def projection_matrix_from_row_basis(V: torch.Tensor):
    """
    V: [K, D]
    行向量张成子空间；若行正交，则投影矩阵可写为 P = V^T V
    """
    return V.transpose(0, 1) @ V  # [D, D]


def subspace_distance(y1: torch.Tensor, y2: torch.Tensor):
    """
    y1, y2: [K, B, D]
    比较每个 sample 上两组 expert 张成的子空间是否一致。
    用投影矩阵 Fro 距离:
        ||P1 - P2||_F
    """
    assert y1.shape == y2.shape
    K, B, D = y1.shape

    dists = []
    for b in range(B):
        V1 = y1[:, b, :]   # [K, D]
        V2 = y2[:, b, :]   # [K, D]
        P1 = projection_matrix_from_row_basis(V1)
        P2 = projection_matrix_from_row_basis(V2)
        dists.append(torch.linalg.norm(P1 - P2))

    dists = torch.stack(dists)
    return {
        "mean": dists.mean().item(),
        "max": dists.max().item(),
        "p95": torch.quantile(dists, 0.95).item(),
    }


def make_collinear_input(K, B, D, noise=1e-4, device="cpu"):
    """
    构造近共线输入，测试 rank-deficient / near-rank-deficient 情况
    """
    base = torch.randn(1, B, D, device=device)
    x = base.repeat(K, 1, 1)
    x = x + noise * torch.randn_like(x)
    return x


def make_random_input(K, B, D, device="cpu"):
    return torch.randn(K, B, D, device=device)


def run_case(name: str, x: torch.Tensor):
    print(f"\n========== {name} ==========")
    print("input shape:", tuple(x.shape))

    # 你仓库当前的 HH 实现
    hh = mixture_layers.OrthogonalLayer1D(
        hh_canon_sign=True,
        hh_rank_tol=1e-6
    )

    with torch.no_grad():
        y_hh = hh(x.clone())

    print("hh shape:", tuple(y_hh.shape))
    assert y_hh.shape == x.shape, f"HH output shape mismatch: {y_hh.shape} vs {x.shape}"

    hh_orth = orth_error_stats_tensor(y_hh)
    print("HH orth error:", hh_orth)

    # 如果层里记录了 last_stats，也打印出来
    if hasattr(hh, "last_stats"):
        print("HH last_stats:", hh.last_stats)

    return {
        "hh_output": y_hh,
        "hh_orth": hh_orth,
    }


def run_pairwise_subspace_check():
    """
    因为你当前仓库没有保留 GS 后端，
    这里提供一个“自带 GS 参考实现”，只用于测试，不接入训练。
    """

    class ReferenceGS(torch.nn.Module):
        def __init__(self, eps=1e-8):
            super().__init__()
            self.eps = eps

        def forward(self, x):
            # x: [K, B, D]
            K, B, D = x.shape
            basis = []

            for i in range(K):
                v = x[i]  # [B, D]
                if i > 0:
                    for b in basis:
                        coeff = (v * b).sum(dim=-1, keepdim=True) / (
                            (b * b).sum(dim=-1, keepdim=True) + self.eps
                        )
                        v = v - coeff * b

                v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + self.eps)
                basis.append(v)

            return torch.stack(basis, dim=0)

    x = make_random_input(6, 32, 64)

    hh = mixture_layers.OrthogonalLayer1D(
        hh_canon_sign=True,
        hh_rank_tol=1e-6
    )
    gs = ReferenceGS()

    with torch.no_grad():
        y_hh = hh(x.clone())
        y_gs = gs(x.clone())

    print("\n========== GS vs HH reference check ==========")
    print("GS shape:", tuple(y_gs.shape))
    print("HH shape:", tuple(y_hh.shape))

    assert y_gs.shape == x.shape
    assert y_hh.shape == x.shape

    gs_orth = orth_error_stats_tensor(y_gs)
    hh_orth = orth_error_stats_tensor(y_hh)
    sub_dist = subspace_distance(y_gs, y_hh)

    print("GS orth error:", gs_orth)
    print("HH orth error:", hh_orth)
    print("GS-HH subspace distance:", sub_dist)


if __name__ == "__main__":
    # case 1: 你当前主实验量级附近
    out1 = run_case("random_k3_b32_d64", make_random_input(3, 32, 64))

    # case 2: 你当前更容易出问题的设定
    out2 = run_case("random_k6_b32_d64", make_random_input(6, 32, 64))

    # case 3: 小维度压力测试
    out3 = run_case("random_k6_b8_d16", make_random_input(6, 8, 16))

    # case 4: 近共线压力测试
    out4 = run_case("collinear_k6_b32_d64", make_collinear_input(6, 32, 64, noise=1e-4))

    # 可选：和参考 GS 做子空间对照
    run_pairwise_subspace_check()