import torch
import triton
import triton.language as tl


MAX_FLOAT32 = float("inf")

# fmt: off
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#         triton.Config({}, num_warps=32),
#     ],
#     key=["N"],
# )
@triton.jit
def kernel_fused_min_dot(
    # Pointers to matrices
    OUT, OUT_VAL, A, B,
    # Matrix dimensions
    M, N, K,
    stride_am, stride_bn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    return_min_values: tl.constexpr
):
    # fmt: on

    """
    Kernel for computing Out = min(A x B, dim = 1)

    - A has shape (M, K)
    - B has shape (K, N)
    - Output has shape (M,)

    This kernel will consolidate over K
    """

    # Position of elements processed by this program
    # and compute the block that each program will go through
    row = tl.program_id(0)
    rm = row * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Keep track of the global distance minima
    global_min = tl.zeros((BLOCK_M,), dtype=tl.float32) + MAX_FLOAT32
    global_arg_min = tl.zeros((BLOCK_M,), dtype=tl.int32)

    # # block level matrix multiplication.
    # # We fetch a block memory block from both inputs, matmul and accumulate, then repeat
    mask_rm = rm < M
    mask_rn = rn < N

    a_ptrs = A + rm[:, None] * stride_am

    for i in range(0, N, BLOCK_N):
        b_ptrs = B + (i + rn[None, :]) * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for j in range(0, K, BLOCK_K):
            rrk = rk + j
            a = tl.load(a_ptrs + rrk[None, :], mask=((rrk[None, :] < K) & mask_rm[:, None]), other=0.0)  # not very optimized, we keep reloading this
            b = tl.load(b_ptrs + rrk[:, None], mask=((rrk[:, None] < K) & mask_rn[None, :]), other=0.0)
            a_2 = a * a
            b_2 = b * b

            # acc += tl.sum(a_2, axis=1)[:, None]
            # acc += tl.sum(b_2, axis=0)[None, :]
            acc -= 2. * tl.dot(a, b)

        min_values = tl.min(acc, axis=1)
        arg_mins = tl.argmin(acc, axis=1).to(tl.int32) + i  # keep track of where we are in the cols

        # Update the global trackers
        global_arg_min = tl.where(min_values < global_min, arg_mins, global_arg_min)
        global_min = tl.where(min_values < global_min, min_values, global_min)

    # write back result
    out_ptrs = OUT + rm
    tl.store(out_ptrs, global_arg_min, mask=mask_rm)

    if return_min_values:
        min_values_ptrs = OUT_VAL + rm
        tl.store(min_values_ptrs, global_min, mask=mask_rm)

def argmin_cdist(a: torch.Tensor, b: torch.Tensor, return_min_values:bool = False):
    """
    Compute e = argmin(a @ b)
    """
    a = a.contiguous()  # noops if not required
    b = b.contiguous()

    M, K = a.shape
    N, K = b.shape

    outputs = torch.empty((M,), device=a.device, dtype=torch.int32)
    min_values = torch.empty((M,), device=a.device, dtype=torch.float32) if return_min_values else outputs  # will not be used

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),) # noqa

    # fmt: off
    kernel_fused_min_dot[grid](
        outputs, min_values, a, b,
        M, N, K,
        a.stride(0), b.stride(0),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=16,
        num_warps=1,
        return_min_values=return_min_values
    )
    # fmt: on

    return outputs, min_values


if __name__ == "__main__":
    import time

    # TODO: Handle the batch dimension
    # TODO: Proper benchmark once parity is solved / scheduling
    # TODO: Add the remaining distance computations (a**2 + b**2 - 2 a.b)

    # Repro:
    torch.manual_seed(12344)

    M, N, K = 2048, 2048, 16
    dtype = torch.float16
    a = torch.randn((M, K), dtype=dtype, device=torch.device("cuda"))
    b = torch.randn((N, K), dtype=dtype, device=torch.device("cuda"))

    # Compute using triton
    torch.cuda.synchronize()
    start = time.time_ns()
    res, _ = argmin_cdist(a, b, return_min_values=False)
    torch.cuda.synchronize()
    stop = time.time_ns()
    print(f"Triton computed in {(stop-start) / 1e6:.1f}ms")

    # Sanity check with pytorch - cdist way
    torch.cuda.synchronize()
    start = time.time_ns()
    res_torch = torch.argmin(torch.cdist(a.unsqueeze(0),b.unsqueeze(0)), -1)
    torch.cuda.synchronize()
    stop = time.time_ns()
    print(f"PyTorch cdist computed in {(stop-start) / 1e6:.1f}ms")

    if not torch.allclose(res, res_torch.to(torch.int32)):
        # DEBUG - visualize possible mismatchs
        print("Not at parity with PyTorch")
        import seaborn as sns
        fig = sns.scatterplot(list(range(M)), (res-res_torch.squeeze()).abs().cpu()).get_figure()
        fig.savefig("error_plot.png")
    else:
        print("All good, perfect parity")