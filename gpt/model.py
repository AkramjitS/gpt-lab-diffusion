import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
from .helper import *

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

# -----------------------------------------------------------------------------
# Activation: ReLU squared
class SquareReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x).square()

# -----------------------------------------------------------------------------
# Sinusoidal positional embedding for timesteps
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * -(math.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=None):
        super().__init__()
        # Calculate head_dim based on model dimensions and num_heads
        self.num_heads = num_heads
        # If head_dim not specified, calculate it based on the model dimension
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(head_dim)
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            block_mask=block_mask, 
            scale=self.scale
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hdim = int(mlp_ratio * dim)
        self.c_fc = CastedLinear(dim, hdim)
        self.squared_relu = SquareReLU()
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = self.squared_relu(x) # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len)
        self.mlp = MLP(dim, mlp_ratio)
        self.lambdas = nn.Parameter(torch.tensor([1.]))

    def forward(self, x: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x
        x = x + self.attn(norm(x), block_mask)
        x = x + self.mlp(norm(x))
        return x
    
# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    #return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)
    return (-v % n) + v
    
class Diffusion(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        mask_token_id: int, 
        bos_token_id: int,
        num_layers: int, 
        num_heads: int, 
        model_dim: int, 
        max_seq_len: int, 
        mlp_ratio: int,
        num_steps: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.num_steps = num_steps
        
        self.embed = nn.Embedding(vocab_size, model_dim)
        # sinusoidal time embedding + MLP
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmb(model_dim),
            nn.Linear(model_dim, model_dim),
            SquareReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, mlp_ratio, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=False, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        
    def get_p_mask(self, t: Tensor) -> Tensor:
        # linear mask schedule from 0 to 1 over num_steps
        return (1 - 1e-10) * (t.float() / (self.num_steps - 1)) + 1e-10

    def forward(
        self, 
        input_seq: Tensor, 
        t: Tensor,
        target_seq: Tensor = None,
        mask: Tensor = None
    ) -> Tensor:
        assert input_seq.ndim == 1 # shape (B*N)
        device = input_seq.device
        L = input_seq.size(0)
        
        # sample mask and ensure at least one masked token
        if mask is None or target_seq is not None:
            p_mask = self.get_p_mask(t)
            mask = torch.rand(L, device=device) < p_mask
            if not mask.any():
                mask[torch.randint(high=L, size=(1,), device=device)] = True
        else:
            p_mask = None

        # creating flex-attentio mask
        seq_src = target_seq if target_seq is not None else input_seq
        docs = (seq_src == self.bos_token_id).cumsum(0)
        def doc_causal(b, h, q_idx, kv_idx):
            same = docs[q_idx] == docs[kv_idx]
            #qm = mask[q_idx]
            #kvv = ~mask[kv_idx]
            #return torch.where(qm, kvv & same, same)
            return same
        # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
        block_mask = create_block_mask(doc_causal, B=None, H=None, Q_LEN=L, KV_LEN=L)
        
        # compute time embedding
        te = self.time_emb(t).view(1, 1, -1)

        x = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        x += te

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, block_mask)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))

        if target_seq is None:
            return logits
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_seq.to(torch.int64).view(-1), 
            reduction="none"
        )

        return loss[mask].sum() / mask.sum().clamp_min(1)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @torch.no_grad()
    def generate(
        self, 
        input_seq, 
        max_new_tokens, 
        temperature=1.0, 
        top_k=None
    ) -> Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert input_seq.ndim == 1
        def cdiv(m, n):
            return (m + (n - 1)) // n
        
        device = input_seq.device
        seq_len = input_seq.size(0)
        mask_append_seq = torch.full((max_new_tokens,), fill_value=self.mask_token_id, dtype=input_seq.dtype, device=input_seq.device)
        masked_seq = torch.cat([input_seq, mask_append_seq], dim=0)
        masked_len = masked_seq.size(0)
        pad_ct = (-masked_len) % 128
        if pad_ct:
            masked_seq = torch.cat((masked_seq, torch.zeros(pad_ct, dtype=masked_seq.dtype, device=device)), dim=0)
        if masked_seq.size(0) > self.max_seq_len:
            masked_seq = masked_seq[:self.max_seq_len]
        
        self.eval()  # Ensure model is in evaluation mode
        mask = masked_seq[:masked_len] == self.mask_token_id
        
        outputs = [masked_seq[:masked_len].clone()]
        for t_id in range(self.num_steps - 1, -1, -1):
            # 1) Reset only currently masked positions
            masked_seq[:masked_len][mask] = self.mask_token_id

            # 2) Forward pass (batch size 1)
            t = torch.tensor([t_id], device=device)
            logits = self.forward(masked_seq[:masked_len], t, mask=mask)  # (1, L, V)

            # 3) Gather logits at current mask positions
            logits = logits[0, mask, :] / temperature

            # 4) Optionally top-k filter (unchanged)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1].unsqueeze(1)] = -float('Inf')

            # 5) Sample
            probs = F.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(probs, 1).squeeze(1)

            # 6) Update only masked positions
            positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
            masked_seq[positions] = sampled_ids.to(masked_seq.dtype)

            outputs.append(masked_seq[:masked_len].clone())
            if t_id == 0:
                break

            # 7) Rebuild dynamic mask for next iteration (your existing logic)
            confidences = torch.gather(probs, 1, sampled_ids.unsqueeze(1)).squeeze(1)
            p_mask = self.get_p_mask(torch.tensor([t_id - 1], device=device))
            rand_mask = torch.rand(sampled_ids.size(0), device=device) < p_mask
            keep_high = confidences >= 0.5
            new_cont_mask = (~keep_high) & rand_mask

            mask[:] = False
            mask[positions] = new_cont_mask

            if torch.all(~mask[seq_len:masked_len]):
                break

        return outputs