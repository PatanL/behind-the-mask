"""Run a block-FP8 checkpoint (e.g. Qwen3.8-27B-FP8) without the hub's Triton kernel, which doesn't import with this
container's Triton: each FP8 linear layer is dequantized to bf16 on the fly (its 128x128 block scales applied) and
multiplied with a plain matmul. Slower than the kernel, but the weights stay FP8 in memory. While generating (a few
rows at a time) a small Triton kernel of our own reads the FP8 weights straight into the matmul instead: the same
arithmetic, about the speed of reading the weights once.

    python run_fp8.py build_vectors.py --chat Qwen/Qwen3.8-27B-FP8 --base none --out runs/q27b --reuse
"""
import torch
import torch.nn.functional as F
from transformers.integrations import finegrained_fp8 as _f


try:   # a few rows at a time (generating): read the FP8 weights straight into the matmul, scales applied per block
    import triton
    import triton.language as tl

    @triton.jit
    def _w8a16(x, w, s, y, M, N, K, sxm, swn, ssn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(x + rm[:, None] * sxm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(w + rn[:, None] * swn + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K), other=0.0)
            sc = tl.load(s + (rn // BK) * ssn + k0 // BK, mask=rn < N, other=0.0)   # (BK is the 128 scale block)
            b = (b.to(tl.float32) * sc[:, None]).to(tl.bfloat16)
            acc += tl.dot(a, tl.trans(b))
        tl.store(y + rm[:, None] * sym + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

    def _fast(x2, weight, scale):
        M, K = x2.shape
        N = weight.shape[0]
        BM = max(16, triton.next_power_of_2(M))
        BN = 16 if N <= 8192 else 64   # (enough tiles to fill the GPU)
        y = torch.empty(M, N, device=x2.device, dtype=torch.bfloat16)
        _w8a16[(triton.cdiv(M, BM), triton.cdiv(N, BN))](x2, weight, scale, y, M, N, K, x2.stride(0), weight.stride(0),
                                                        scale.stride(0), y.stride(0), BM=BM, BN=BN, BK=128, num_warps=4, num_stages=4)
        return y
except Exception as e:   # (no Triton: the dequantizing path below)
    print(f"[fp8] no Triton kernel ({e})", flush=True)
    _fast = None


def fp8_linear(input, weight, weight_scale_inv, block_size=None, bias=None, activation_scale=None, allow_deepgemm=True):
    rows = input.numel() // input.shape[-1]
    if (_fast is not None and block_size is not None and tuple(block_size) == (128, 128) and rows <= 64 and input.is_cuda
            and weight.is_contiguous() and weight.shape[1] % 128 == 0):
        x2 = input.reshape(rows, input.shape[-1]).to(torch.bfloat16).contiguous()
        y = _fast(x2, weight, weight_scale_inv.float().contiguous()).reshape(*input.shape[:-1], weight.shape[0])
        return y if bias is None else y + bias.to(torch.bfloat16)
    w = weight.to(torch.bfloat16)
    if block_size is None:
        w = w * weight_scale_inv.float().to(torch.bfloat16)
    else:
        bo, bi = block_size
        (o, i), (so, si) = w.shape, weight_scale_inv.shape
        if o == so * bo and i == si * bi:   # (whole blocks: the scales broadcast over a block view, no full-size copy)
            w = (w.view(so, bo, si, bi) * weight_scale_inv.to(torch.bfloat16)[:, None, :, None]).view(o, i)
        else:
            s = weight_scale_inv.float().repeat_interleave(bo, 0)[:o].repeat_interleave(bi, 1)[:, :i]
            w = (w.float() * s).to(torch.bfloat16)
    return F.linear(input.to(torch.bfloat16), w, None if bias is None else bias.to(torch.bfloat16))


_f.fp8_linear = fp8_linear


# Some layers (here every MLP projection) are loaded as plain nn.Linear holding the FP8 weights, without their block
# scales; after loading, each such layer gets its scales read from the checkpoint and the same dequantizing forward.
def _blockwise(layer, scale):
    def forward(x):
        bo, bi = -(-layer.weight.shape[0] // scale.shape[0]), -(-layer.weight.shape[1] // scale.shape[1])
        return fp8_linear(x, layer.weight, scale, block_size=(bo, bi), bias=layer.bias)
    return forward


def fix_unconverted(model, repo: str) -> int:
    import json, os
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(repo, allow_patterns=["*.json"])
    wmap = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not hasattr(mod, "weight_scale_inv") and mod.weight.dtype == torch.float8_e4m3fn:
            # (loaded as a text-only model its layers are "model.layers...": the checkpoint says "model.language_model.layers...")
            cands = [f"{name}.weight_scale_inv", f"model.language_model.{name[len('model.'):]}.weight_scale_inv" if name.startswith("model.") else ""]
            key = next((k for k in cands if k in wmap), None)
            if key is None:
                print(f"[fp8] no scales for {name}", flush=True)
                continue
            with safe_open(os.path.join(d, wmap[key]), "pt", device=str(mod.weight.device)) as f:
                scale = f.get_tensor(key).float()
            mod.register_buffer("weight_scale_inv", scale, persistent=False)
            mod.forward = _blockwise(mod, scale)
            n += 1
    return n


from transformers import AutoModelForCausalLM as _Auto
_orig = _Auto.from_pretrained.__func__


def quantize_linear(layer: torch.nn.Linear):
    """Store a bf16 linear layer as FP8 with 128x128 block scales (the checkpoint's own format) and run it through the
    same kernel: half the bytes read each step. Used for the output layer (BTM_LMHEAD_FP8=1): 2.5 GB -> 1.3 GB."""
    w = layer.weight.data
    o, i = w.shape
    if o % 128 or i % 128:
        return False
    blocks = w.float().view(o // 128, 128, i // 128, 128)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)          # [o/128, i/128]: e4m3's largest value is 448
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(o, i)
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.register_buffer("weight_scale_inv", scale, persistent=False)
    layer.forward = _blockwise(layer, scale)
    return True


def _from_pretrained(cls, repo, *args, **kwargs):
    import os
    model = _orig(cls, repo, *args, **kwargs)
    n = fix_unconverted(model, repo)
    if n:
        print(f"[fp8] {n} plain linear layers given their block scales", flush=True)
    head = getattr(model, "lm_head", None)
    if os.environ.get("BTM_LMHEAD_FP8") == "1" and isinstance(head, torch.nn.Linear) and head.weight.dtype == torch.bfloat16:
        if quantize_linear(head):
            torch.cuda.empty_cache()
            print("[fp8] output layer stored as FP8", flush=True)
    return model


_Auto.from_pretrained = classmethod(_from_pretrained)
