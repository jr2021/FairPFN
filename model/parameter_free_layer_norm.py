import torch
from torch import nn

import utils

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ModuleNotFoundError:
    HAS_TRITON = False

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex

    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False


if HAS_TRITON:

    @triton.jit
    def _layer_norm_fwd_fused(
        X,  # pointer to the input
        Y,  # pointer to the output
        # B2,  # pointer to the secondary biases
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Compute mean
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.0)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Write mean / rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            # b2 = tl.load(B2 + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat  # + b2
            # Write output
            tl.store(Y + cols, y, mask=mask)

    @triton.jit
    def _layer_norm_bwd_dx_fused(
        DX,  # pointer to the input gradient
        DY,  # pointer to the output gradient
        X,  # pointer to the input
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        Lock,  # pointer to the lock
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        # Map the program id to the elements of X, DX, and DY it should compute.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        # Offset locks and weights/biases gradient pointer for parallel reduction
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd
        wdy = dy
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

    class LayerNormF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, normalized_shape, eps):
            # allocate output
            y = torch.empty_like(x)
            # reshape input data into 2D tensor
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            mean = torch.empty((M,), dtype=torch.float32, device="cuda")
            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
            # Less than 64KB per feature: enqueue fused kernel
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError(
                    "This layer norm doesn't support feature dim >= 64KB."
                )
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # enqueue kernel
            _layer_norm_fwd_fused[(M,)](
                x_arg,
                y,
                mean,
                rstd,
                x_arg.stride(0),
                N,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            ctx.save_for_backward(x, mean, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):
            x, m, v = ctx.saved_tensors
            # heuristics for amount of parallel reduction stream for DW/DB
            N = x.shape[1]
            GROUP_SIZE_M = 64
            if N <= 8192:
                GROUP_SIZE_M = 96
            if N <= 4096:
                GROUP_SIZE_M = 128
            if N <= 1024:
                GROUP_SIZE_M = 256
            if N <= 512:
                GROUP_SIZE_M = 512
            if N <= 256:
                GROUP_SIZE_M = 1024
            if N <= 128:
                GROUP_SIZE_M = 2048
            if N <= 64:
                GROUP_SIZE_M = 4096
            if N <= 32:
                GROUP_SIZE_M = 8192
            # allocate output
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
            dx = torch.empty_like(dy)
            # enqueue kernel using forward pass heuristics
            # also compute partial sums for DW and DB
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            _layer_norm_bwd_dx_fused[(M,)](
                dx,
                dy,
                x,
                m,
                v,
                locks,
                x_arg.stride(0),
                N,
                ctx.eps,
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_warps=ctx.num_warps,
            )
            grid = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
            # accumulate partial sums in separate kernel
            return dx, None, None


class ParameterFreeLayerNormTriton(nn.LayerNorm):
    # def __init__(self, d_model, eps=layer_norm_eps, dtype=dtype, device=device):
    def __init__(
        self, normalized_shape, eps: float = 1e-5, device=None, dtype=None
    ) -> None:
        super().__init__(normalized_shape, eps, False, device, dtype)

    def forward(self, x):
        if x.is_cuda and not HAS_TRITON:
            utils.print_once(
                "Using CUDA, but Triton is not available. Falling back to PyTorch, this will be slower. Install Triton to enable faster training."
            )

        if x.is_cuda and HAS_TRITON:
            # return super()(x)
            x_shape = x.shape
            x_type = x.dtype
            r = LayerNormF.apply(x.reshape(-1, x.shape[-1]), (x.shape[-1],), self.eps)
            return r.reshape(*x_shape).to(x_type)
        else:
            return super().forward(x)
