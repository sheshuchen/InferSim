"""
Grouped Masked GEMM 的简化 PyTorch 实现

本文件用纯 PyTorch 实现 deep_gemm.m_grouped_fp8_gemm_nt_masked 的核心逻辑，
帮助你理解 "分组 + Mask + FP8 GEMM" 的原理，不依赖 DeepGEMM 库。

核心概念:
- Grouped: 将多个独立的矩阵乘法打包成一次 kernel 调用
- Masked: 每个 group 实际只有前 masked_m[g] 行是有效的，其余是 padding
- FP8: 数据以 8-bit 浮点存储，配合 scale 进行反量化后计算
- NT: Normal-Transposed，即 A 不做转置，B 做转置（计算 A @ B.T）
"""

import random
from typing import Tuple

import torch


def ceil_div(a: int, b: int) -> int:
    """整数向上取整除法，例如 ceil_div(130, 128) = 2。"""
    return (a + b - 1) // b


def m_grouped_fp8_gemm_nt_masked(
    a_fp8: Tuple[torch.Tensor, torch.Tensor],
    b_fp8: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    disable_ue8m0_cast: bool = True,
) -> None:
    """
    简化版 grouped masked FP8 GEMM（NT 布局）。

    与 deep_gemm.m_grouped_fp8_gemm_nt_masked 接口保持一致，但内部用 PyTorch 实现，
    不调用任何 CUDA kernel，适合在 CPU/GPU 上学习调试。

    参数说明:
        a_fp8: (a_data, a_scales)
            - a_data:  (num_groups, max_m, k)
                左矩阵的 FP8 数据。max_m 是预留的最大行数，实际只用前 masked_m[g] 行。
            - a_scales: (num_groups, max_m, ceil_div(k, 128))
                左矩阵的 per-token 反量化系数。每行（每个 token）的每 128 个 k 列共享一个 scale。
                这叫 "per-token scaling"，因为每行独立计算 scale。
        b_fp8: (b_data, b_scales)
            - b_data:  (num_groups, n, k)
                右矩阵的 FP8 数据。注意形状是 (n, k)，计算时会转置为 (k, n)。
            - b_scales: (num_groups, ceil_div(n, 128), ceil_div(k, 128))
                右矩阵的 per-block 反量化系数。每 128×128 的子块共享一个 scale。
                这叫 "per-block scaling"，因为 scale 是按二维块划分的。
        d: (num_groups, max_m, n)
            输出缓冲区。计算结果会写入 d[g, :masked_m[g], :] 的前若干行，其余行保持原值（通常为 0）。
        masked_m: (num_groups,), dtype=int
            每个 group 实际有效的 m 维度行数。例如 masked_m[0]=30 表示第 0 个 group 只有前 30 行有效数据。
        expected_m_per_group: int
            期望的每个 group 行数。在真实 DeepGEMM 中用于决定 tiling 策略，
            本简化实现中仅作兼容性保留，不参与计算。
        disable_ue8m0_cast: bool
            是否禁用 UE8M0 格式的 scale 转换。DeepGEMM 内部使用，本实现忽略此参数。

    计算逻辑（对每个 group g）:
        valid_m = masked_m[g]
        A_g = dequantize(a_fp8[g, :valid_m, :])      # 形状 (valid_m, k)
        B_g = dequantize(b_fp8[g, :, :])              # 形状 (n, k)
        d[g, :valid_m, :] = A_g @ B_g.T               # 形状 (valid_m, n)
    """
    a_data, a_scales = a_fp8
    b_data, b_scales = b_fp8

    num_groups, max_m, k = a_data.shape
    n = b_data.shape[1]

    # 检查输入维度是否匹配
    assert b_data.shape[2] == k, "a 和 b 的 k 维度必须相同"
    assert d.shape == (num_groups, max_m, n), f"d 的形状应为 {(num_groups, max_m, n)}，实际为 {d.shape}"
    assert masked_m.shape == (num_groups,), f"masked_m 的形状应为 ({num_groups},)，实际为 {masked_m.shape}"

    # ------------------------------------------------------------------
    # Step 1: 反量化（Dequantize）FP8 数据
    # ------------------------------------------------------------------
    # FP8 数据在存储时范围很小（e4m3fn 最大值约 448），直接乘上 scale 就能恢复到原始数值范围。
    # 这是低精度推理的核心技巧：用 8-bit 存数据，用 32-bit 的 scale 保精度。

    # 反量化 a（per-token scale）
    # a_scales 形状: (num_groups, max_m, ceil_div(k, 128))
    # 需要把最后一维扩展 128 倍，变成 (num_groups, max_m, k)
    a_scales_expanded = a_scales.repeat_interleave(128, dim=-1)[..., :k]
    # 先将 FP8 转成 float32（避免溢出），再乘 scale
    a_dequant = a_data.to(torch.float32) * a_scales_expanded

    # 反量化 b（per-block scale）
    # b_scales 形状: (num_groups, ceil_div(n, 128), ceil_div(k, 128))
    # 先在 n 维度扩展 128 倍
    b_scales_n = b_scales.repeat_interleave(128, dim=1)[:, :n, :]
    # 再在 k 维度扩展 128 倍
    b_scales_expanded = b_scales_n.repeat_interleave(128, dim=-1)[..., :k]
    b_dequant = b_data.to(torch.float32) * b_scales_expanded

    # ------------------------------------------------------------------
    # Step 2: 逐 group 执行 Masked GEMM
    # ------------------------------------------------------------------
    # "Grouped" 的含义：把多个小矩阵乘法打包在一起，共享同一个 kernel 调度。
    # 在真实 GPU kernel 中，这些 group 会被分配到不同的 thread block；
    # 在 PyTorch 实现中，我们用一个 for-loop 逐个计算。

    for g in range(num_groups):
        valid_m = int(masked_m[g].item())

        # 跳过空 group（虽然实际很少出现）
        if valid_m <= 0:
            continue

        # 取出第 g 个 group 的有效数据
        a_g = a_dequant[g, :valid_m, :]   # (valid_m, k)
        b_g = b_dequant[g, :, :]          # (n, k)

        # NT 布局：A 不转置，B 转置
        # 即计算 C = A @ B^T
        # a_g 形状: (valid_m, k)
        # b_g.t() 形状: (k, n)
        # result 形状: (valid_m, n)
        result = torch.matmul(a_g, b_g.t())

        # 将结果写回输出缓冲区的前 valid_m 行
        d[g, :valid_m, :] = result.to(d.dtype)

        # 注意：d[g, valid_m:, :] 保持原值（通常为 0，即 padding 区域）


# =============================================================================
# 以下为测试辅助函数
# =============================================================================

def generate_test_data(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple:
    """
    生成测试数据，模拟 deepgemm_grouped_gemm_masked.py 中的 generate_m_grouped_masked。

    为了兼容所有环境（包括不支持 FP8 的 CPU），本函数默认生成 bfloat16 数据，
    并构造 "伪 FP8 格式" 的 tuple（scale 全为 1.0），这样 m_grouped_fp8_gemm_nt_masked
    可以直接调用，无需修改接口。

    返回:
        a_fp8: (a_data, a_scales)  —— 左矩阵
        b_fp8: (b_data, b_scales)  —— 右矩阵
        d:     输出缓冲区
        masked_m: 每个 group 的有效行数
        ref_d: 用标准 torch.einsum 计算的参考结果，用于验证正确性
    """
    # 生成随机 bf16 数据作为 "原始数据"
    a_raw = torch.randn((num_groups, max_m, k), device=device, dtype=dtype)
    b_raw = torch.randn((num_groups, n, k), device=device, dtype=dtype)

    # 用标准 einsum 计算参考结果: (g,m,k) @ (g,n,k)^T -> (g,m,n)
    # 注意：这里用 float32 计算，避免 bfloat16 精度不足影响验证
    ref_d = torch.einsum("gmk,gnk->gmn", a_raw.to(torch.float32), b_raw.to(torch.float32))

    # 生成 masked_m：每个 group 的有效行数在 expected_m_per_group 附近波动
    masked_m = torch.empty((num_groups,), device=device, dtype=torch.int)
    for j in range(num_groups):
        # 随机波动 0.7~1.3 倍，模拟实际推理中各专家接收到的 token 数不均匀
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    # 确保不超过 max_m
    masked_m = torch.clamp(masked_m, 1, max_m)

    # 构造 "伪 FP8" 数据：直接把 bf16 数据包装成 FP8 接口，scale 全设为 1.0
    # 这样反量化后数据不变，方便在任意设备上测试算法逻辑
    a_data = a_raw
    a_scales = torch.ones(
        (num_groups, max_m, ceil_div(k, 128)), device=device, dtype=torch.float32
    )

    b_data = b_raw
    b_scales = torch.ones(
        (num_groups, ceil_div(n, 128), ceil_div(k, 128)), device=device, dtype=torch.float32
    )

    # 输出缓冲区，初始化为 0
    # 测试时使用 float32，避免低精度影响正确性验证
    d = torch.zeros((num_groups, max_m, n), device=device, dtype=torch.float32)

    return (a_data, a_scales), (b_data, b_scales), d, masked_m, ref_d


def test_correctness():
    """运行多组随机测试，验证简化实现的输出与参考结果一致。"""
    print("=" * 60)
    print("开始测试 grouped masked GEMM 正确性")
    print("=" * 60)

    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")

    # 测试多组形状参数
    test_cases = [
        # (num_groups, max_m, expected_m, n, k)
        (4, 128, 64, 256, 128),
        (8, 256, 128, 512, 256),
        (16, 512, 256, 1024, 512),
    ]

    for num_groups, max_m, expected_m, n, k in test_cases:
        print(f"测试配置: groups={num_groups}, max_m={max_m}, expected_m={expected_m}, n={n}, k={k}")

        a_fp8, b_fp8, d, masked_m, ref_d = generate_test_data(
            num_groups, max_m, expected_m, n, k, device=device
        )

        # 调用我们的简化实现
        m_grouped_fp8_gemm_nt_masked(
            a_fp8, b_fp8, d, masked_m, expected_m_per_group=expected_m
        )

        # 逐 group 比较前 masked_m[g] 行（统一转成 float32 比较）
        max_diff = 0.0
        d_float = d.to(torch.float32)
        for g in range(num_groups):
            valid_m = int(masked_m[g].item())
            diff = torch.abs(
                d_float[g, :valid_m, :] - ref_d[g, :valid_m, :]
            ).max().item()
            max_diff = max(max_diff, diff)

        # 浮点矩阵乘法的数值误差通常在 1e-4 ~ 1e-3 量级
        if max_diff < 1e-2:
            print(f"  通过! 最大误差: {max_diff:.2e}")
        else:
            print(f"  失败! 最大误差: {max_diff:.2e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def demo_masking_behavior():
    """
    演示 Masked GEMM 的核心行为：不同 group 的有效行数不同，
    padding 区域保持为 0。
    """
    print("\n" + "=" * 60)
    print("演示: Masked GEMM 的 padding 行为")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_groups = 2
    max_m = 8
    k = 4
    n = 4

    # 构造简单的确定性数据，方便观察
    a_data = torch.ones((num_groups, max_m, k), device=device, dtype=torch.bfloat16)
    b_data = torch.ones((num_groups, n, k), device=device, dtype=torch.bfloat16)
    a_scales = torch.ones((num_groups, max_m, 1), device=device, dtype=torch.float32)
    b_scales = torch.ones((num_groups, 1, 1), device=device, dtype=torch.float32)

    d = torch.zeros((num_groups, max_m, n), device=device, dtype=torch.bfloat16)

    # 第 0 个 group 只有前 3 行有效，第 1 个 group 只有前 5 行有效
    masked_m = torch.tensor([3, 5], device=device, dtype=torch.int)

    m_grouped_fp8_gemm_nt_masked(
        (a_data, a_scales),
        (b_data, b_scales),
        d,
        masked_m,
        expected_m_per_group=4,
    )

    print(f"masked_m = {masked_m.tolist()}")
    print(f"输出 d[0] (group 0, 只有前 3 行有值):")
    print(d[0].to(torch.float32))
    print(f"\n输出 d[1] (group 1, 只有前 5 行有值):")
    print(d[1].to(torch.float32))
    print("\n可以看到：有效行被填满，padding 行保持为 0。")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    # 运行正确性测试
    test_correctness()

    # 运行 masking 行为演示
    demo_masking_behavior()
