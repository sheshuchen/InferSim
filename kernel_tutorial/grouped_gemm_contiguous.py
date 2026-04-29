"""
Grouped Contiguous GEMM 的简化 PyTorch 实现

与 masked 版本的区别：
- masked: a 形状 (num_groups, max_m, k)，masked_m 标记每 group 有效行数
- contiguous: a 形状 (m, k)，所有 group 数据连续拼接，m_indices 标记每行属于哪个 group
"""

import random

import torch


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(x: int, alignment: int) -> int:
    """向上对齐到 alignment 的倍数。"""
    return (x + alignment - 1) // alignment * alignment


def m_grouped_fp8_gemm_nt_contiguous(
    a_fp8,
    b_fp8,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    disable_ue8m0_cast: bool = True,
):
    """
    连续布局的 grouped FP8 GEMM。

    a_fp8: (a_data, a_scales)
        a_data:  (m, k), 所有 group 的输入连续拼接
        a_scales: (m, ceil_div(k, 128)), per-token scale
    b_fp8: (b_data, b_scales)
        b_data:  (num_groups, n, k)
        b_scales: (num_groups, ceil_div(n, 128), ceil_div(k, 128)), per-block scale
    d: (m, n), 输出缓冲区
    m_indices: (m,), 每行对应的 group id，-1 表示 padding
    """
    a_data, a_scales = a_fp8
    b_data, b_scales = b_fp8

    m, k = a_data.shape
    num_groups, n, _ = b_data.shape

    # 反量化
    a_sq = a_scales.repeat_interleave(128, dim=-1)[:, :k]
    a_dq = a_data.to(torch.float32) * a_sq

    b_sq = b_scales.repeat_interleave(128, dim=1)[:, :n, :]
    b_sq = b_sq.repeat_interleave(128, dim=-1)[:, :, :k]
    b_dq = b_data.to(torch.float32) * b_sq

    # 逐 group 计算
    for g in range(num_groups):
        rows = (m_indices == g).nonzero(as_tuple=True)[0]
        if len(rows) == 0:
            continue
        a_g = a_dq[rows]       # (num_rows, k)
        b_g = b_dq[g]          # (n, k)
        d[rows] = (a_g @ b_g.t()).to(d.dtype)

    # padding 清零
    d[m_indices == -1] = 0


def generate_test_data(num_groups, expected_m, n, k, alignment=128):
    """生成 contiguous 布局的测试数据。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actual_ms = [int(expected_m * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(m, alignment) for m in actual_ms]
    m = sum(aligned_ms)

    a_raw = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_raw = torch.randn((num_groups, n, k), device=device, dtype=torch.bfloat16)

    # 参考结果（用 float32 计算，避免低精度影响验证）
    ref_d = torch.zeros((m, n), device=device, dtype=torch.float32)
    start = 0
    for i, am in enumerate(aligned_ms):
        ref_d[start : start + am] = a_raw[start : start + am].float() @ b_raw[i].t().float()
        start += am

    # m_indices: 标记每行属于哪个 group，padding 为 -1
    m_indices = torch.full((m,), -1, device=device, dtype=torch.int32)
    start = 0
    for i, (actual, aligned) in enumerate(zip(actual_ms, aligned_ms)):
        m_indices[start : start + actual] = i
        start += aligned

    # 包装成伪 FP8（scale=1）
    a_fp8 = (a_raw, torch.ones((m, ceil_div(k, 128)), device=device, dtype=torch.float32))
    b_fp8 = (
        b_raw,
        torch.ones((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device=device, dtype=torch.float32),
    )
    d = torch.zeros((m, n), device=device, dtype=torch.float32)

    return a_fp8, b_fp8, d, m_indices, ref_d, actual_ms, aligned_ms


def test():
    print("=" * 50)
    print("Testing grouped contiguous GEMM")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    cases = [
        (4, 64, 256, 128),
        (8, 128, 512, 256),
        (16, 256, 1024, 512),
    ]

    for num_groups, expected_m, n, k in cases:
        a_fp8, b_fp8, d, m_indices, ref_d, actual_ms, aligned_ms = generate_test_data(
            num_groups, expected_m, n, k
        )

        m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, m_indices)

        # 只比较有效行（排除 padding）
        valid_mask = m_indices != -1
        diff = torch.abs(d[valid_mask] - ref_d[valid_mask].to(torch.float32)).max().item()

        status = "PASS" if diff < 1e-2 else "FAIL"
        print(
            f"groups={num_groups:2}, actual_ms={actual_ms}, aligned_ms={aligned_ms}, "
            f"n={n:4}, k={k:4} | {status} (max_diff={diff:.2e})"
        )

    print("\n" + "=" * 50)
    print("Done")
    print("=" * 50)


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    test()
