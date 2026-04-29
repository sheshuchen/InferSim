"""
Fused MoE 的简化 PyTorch 实现

将 MoE（混合专家）的前向计算拆分为清晰的三步：
1. Router: 计算每个 token 应该发给哪些专家
2. MLP: 对每个 token 执行选中的专家的 up-gate-down 计算
3. Combine: 将各专家的输出按权重加权求和
"""

import random

import torch
import torch.nn.functional as F


def select_expert_topk(gating_logits, topk):
    """
    根据 router 输出的 logits，为每个 token 选择 top-k 个专家。

    参数:
        gating_logits: (num_tokens, num_experts)  ——  router 输出的原始分数
        topk: int                               ——  每个 token 选择的专家数

    返回:
        topk_idx:    (num_tokens, topk)          ——  选中的专家编号
        topk_weight: (num_tokens, topk)          ——  对应专家的权重（softmax 后）
    """
    # 对 logits 做 softmax 得到概率分布
    weights = F.softmax(gating_logits, dim=-1)
    # 取 top-k 个专家及其权重
    topk_weight, topk_idx = torch.topk(weights, topk, dim=-1)
    # 对权重重新归一化，使每个 token 的 topk 权重之和为 1
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return topk_idx, topk_weight


def fused_moe_pytorch(
    x,
    w1,
    w2,
    gating_logits,
    topk,
):
    """
    纯 PyTorch 实现的 MoE 前向计算。

    参数:
        x: (num_tokens, hidden_size)
            输入的 hidden states，每个 token 一行。

        w1: (num_experts, shard_intermediate_size * 2, hidden_size)
            每个专家的 gate + up 合并权重。
            前 shard_intermediate_size 行是 gate 投影，
            后 shard_intermediate_size 行是 up 投影。

        w2: (num_experts, hidden_size, shard_intermediate_size)
            每个专家的 down 投影权重。

        gating_logits: (num_tokens, num_experts)
            Router 网络输出的原始分数，未经 softmax。

        topk: int
            每个 token 路由到的专家数量。

    返回:
        output: (num_tokens, hidden_size)
            融合后的输出，与输入 x 形状相同。
    """
    num_tokens, hidden_size = x.shape
    num_experts = w1.shape[0]
    intermediate_size = w1.shape[1] // 2

    # Step 1: Router —— 为每个 token 选择 top-k 专家
    topk_idx, topk_weight = select_expert_topk(gating_logits, topk)

    # 输出缓冲区
    output = torch.zeros_like(x)

    # Step 2 & 3: 逐个专家处理其接收到的 token，然后加权求和
    for expert_id in range(num_experts):
        # 找到所有选中该专家的 (token, 该专家在 topk 中的位置)
        token_mask = (topk_idx == expert_id)
        if not token_mask.any():
            continue

        # 收集该专家接收到的所有 token
        token_ids = token_mask.nonzero(as_tuple=True)[0]          # (num_assigned,)
        topk_positions = token_mask.nonzero(as_tuple=True)[1]     # (num_assigned,)

        x_i = x[token_ids]  # (num_assigned, hidden_size)

        # Gate + Up 投影: w1 前半是 gate，后半是 up
        gate_up = x_i @ w1[expert_id].t()  # (num_assigned, intermediate_size * 2)
        gate = gate_up[:, :intermediate_size]
        up = gate_up[:, intermediate_size:]

        # SiLU 激活: silu(x) = x * sigmoid(x)
        activated = gate * torch.sigmoid(gate)

        # 逐元素相乘（门控机制）
        h = activated * up  # (num_assigned, intermediate_size)

        # Down 投影
        out_i = h @ w2[expert_id].t()  # (num_assigned, hidden_size)

        # 按该 token 给该专家的权重进行缩放，并累加到输出
        weights = topk_weight[token_ids, topk_positions]  # (num_assigned,)
        output[token_ids] += out_i * weights.unsqueeze(-1)

    return output


def test():
    print("=" * 50)
    print("Testing fused MoE")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # 模拟一个 MoE 层的配置
    num_tokens = 128
    hidden_size = 64
    num_experts = 4
    intermediate_size = 128
    topk = 2

    torch.manual_seed(42)
    random.seed(42)

    # 输入
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    # Router logits
    gating_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    # 权重
    w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16)

    # 执行
    output = fused_moe_pytorch(x, w1, w2, gating_logits, topk)

    print(f"Input  x shape:           {x.shape}")
    print(f"w1 shape:                 {w1.shape}")
    print(f"w2 shape:                 {w2.shape}")
    print(f"Gating logits shape:      {gating_logits.shape}")
    print(f"Output shape:             {output.shape}")

    # 验证输出形状
    assert output.shape == x.shape
    print("\nShape check: PASS")

    # 检查每个 token 确实路由到了 topk 个专家
    topk_idx, _ = select_expert_topk(gating_logits, topk)
    print(f"\nExample routing for token 0: experts = {topk_idx[0].tolist()}")

    print("\n" + "=" * 50)
    print("Done")
    print("=" * 50)


if __name__ == "__main__":
    test()
