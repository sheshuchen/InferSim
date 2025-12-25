import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.model_config import ModelConfig
from hardware.gpu import gpu_map

DEQUANT_CYCLES = {
    90: {
        "fp8_to_half": 1 / 64,
        "half_to_fp32": 1 / 64,
        "fp32_to_bf16": 1 / 16,
        "bf16_mul_fp32": 1 / 256,
    }
}

BLOCK_M = 64
QUANT_TILE_SIZE = 128
SIZEOF_BF16 = 2
SIZEOF_FP8 = 1


def sparse_mla_fp8(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    topk: int,
    dim: int,
    gpu_type: str,
    dim_rope: int,
) -> tuple[float, float]:
    gpu = gpu_map[gpu_type]

    dim_nope = dim

    compute_volume_flop = (
        batch_size * num_heads * seq_len * 2 * topk * (dim_nope + dim_rope + dim_nope)
    )

    num_heads_per_block = min(BLOCK_M, num_heads)

    time_mma_per_token = (
        num_heads_per_block
        * (dim_nope + dim_nope + dim_rope)
        * 2
        / gpu.fp16_tflops
        / 1e12
        * gpu.num_sm
    )

    time_load_kv_per_token = (
        (dim_nope + dim_rope) * SIZEOF_BF16 / gpu.mem_bw / 1e9 * gpu.num_sm
    )

    time_mma_per_block = BLOCK_M * time_mma_per_token
    time_load_kv_per_block = BLOCK_M * time_load_kv_per_token
    print(
        f"time_load_kv_per_block: {time_load_kv_per_block * 1e6:.3f}us, time_mma_per_block: {time_mma_per_block * 1e6:.3f}us"
    )
    time_per_block = max(seq_len * time_load_kv_per_block, seq_len * time_mma_per_block)

    sum_block = topk // BLOCK_M * batch_size
    num_block_per_sm = (sum_block + gpu.num_sm - 1) / gpu.num_sm

    time = num_block_per_sm * time_per_block

    time_ms = time * 1000
    theoretical_max_tflops = compute_volume_flop / time / 1e12

    return time_ms, theoretical_max_tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse MLA FP8 simulation")
    parser.add_argument(
        "--seq_len", type=int, required=True, help="Sequence length for query"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "hf_configs", "deepseek_v3.2_config.json"
        ),
        help="Path to model configuration file (default: deepseek_v3.2_config.json)",
    )
    parser.add_argument(
        "--gpu_type", type=str, required=True, help="GPU type (e.g., H20, H800)"
    )

    args = parser.parse_args()

    config = ModelConfig(args.config_path)

    time_ms, theoretical_max_tflops = sparse_mla_fp8(
        batch_size=128,
        num_heads=16,
        seq_len=args.seq_len,
        topk=config.index_topk,
        dim=384,
        gpu_type=args.gpu_type,
        dim_rope=config.qk_rope_head_dim,
    )
    print(f"Time: {time_ms:.3f} ms")
    print(f"TFLOPS: {theoretical_max_tflops:.2f}")
