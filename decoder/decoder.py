"""
完整 Transformer Decoder — 整合仓库所有概念
  - RMSNorm (Pre-norm)
  - RoPE 旋转位置编码
  - GQA 分组查询注意力 + KV Cache
  - SwiGLU FFN
  - 自回归生成循环 (Pre-fill → Decode)

对着仓库各目录可找到对应的独立实现：
  attention/   → MHA, KV Cache
  gqa/         → GQA, repeat_kv
  rope/        → RoPE, 频率预计算
  ffn/         → SwiGLU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════
# 1. RMSNorm
# ═══════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama 系标配)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ═══════════════════════════════════════════════════════════
# 2. RoPE — 旋转位置编码
# ═══════════════════════════════════════════════════════════
def precompute_rope_freqs(d_head, max_seq_len, theta=10000.0):
    """
    预计算 cos 和 sin 表
    返回: cos, sin 各 (max_seq_len, d_head // 2)
    """
    # 每个维度对的频率: θ_i = 10000^{-2i/d}
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (max_seq_len, d_head//2)
    return angles.cos(), angles.sin()


def apply_rotary_emb(x, cos, sin, offset=0):
    """
    对 Q 或 K 施加旋转位置编码（实数版本）

    x:      (batch, seq_len, num_heads, d_head)
    cos, sin: 预计算的频率表 (max_seq_len, d_head//2)
    offset:   当使用 KV Cache 时，新 token 的起始位置
    """
    seq_len = x.shape[1]
    d_head_half = x.shape[-1] // 2

    # 取出奇数位和偶数位
    x_even = x[..., 0::2]  # (batch, seq_len, num_heads, d_head//2)
    x_odd  = x[..., 1::2]

    # 取对应位置片段的 cos/sin，调整形状以广播
    c = cos[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_head//2)
    s = sin[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)

    # 二维旋转: [x_even, x_odd] 旋转角度 θ 后:
    #   x_even' = x_even * cos - x_odd * sin
    #   x_odd'  = x_even * sin + x_odd * cos
    x_rotated_even = x_even * c - x_odd * s
    x_rotated_odd  = x_even * s + x_odd * c

    # 交错拼回: (e1, o1, e2, o2, ...)
    return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)


# ═══════════════════════════════════════════════════════════
# 3. GQA + KV Cache — 核心注意力
# ═══════════════════════════════════════════════════════════
def repeat_kv(kv, n_rep):
    """GQA: 在 head 维度广播 KV 以匹配 Q 头数"""
    if n_rep == 1:
        return kv
    # kv: (batch, seq_len, num_kv_heads, d_head)
    return kv.repeat_interleave(n_rep, dim=2)


def gqa_attention(q, k, v, cos, sin, past_kv=None, past_len=0, mask=None):
    """
    GQA 注意力 + RoPE + KV Cache，一次调用完成

    参数:
        q, k, v: 投影后的 (batch, seq_len, num_heads, d_head)
                 注意: k, v 的 num_heads 是 num_kv_heads (比 q 少)
        cos, sin: RoPE 频率表
        past_kv:  (past_k, past_v) 或 None
        past_len: 缓存的已有长度 (用于 RoPE offset)
        mask:     因果 mask, (1, 1, seq_len_q, seq_len_k) 或 None

    返回:
        output:    (batch, seq_len, num_q_heads, d_head)
        new_kv:    (new_k, new_v)
    """
    _, seq_len_q, num_q_heads, d_head = q.shape
    _, seq_len_k, num_kv_heads, _ = k.shape
    n_rep = num_q_heads // num_kv_heads

    # —— RoPE: Q 和 K 各自旋转，位置 = past_len + 序号 ——
    q = apply_rotary_emb(q, cos, sin, offset=past_len)
    k = apply_rotary_emb(k, cos, sin, offset=past_len)

    # —— KV Cache: 拼上历史 ——
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=1)
        v = torch.cat([past_v, v], dim=1)
    new_kv = (k, v)

    # —— GQA: KV 头数不够，广播补齐 ——
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    # —— 转置: (batch, num_heads, seq_len, d_head) ——
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # —— Scaled Dot-Product Attention ——
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)

    if mask is not None:
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    # —— 恢复形状 ——
    output = output.transpose(1, 2)  # (batch, seq_len, num_q_heads, d_head)
    return output, new_kv


# ═══════════════════════════════════════════════════════════
# 4. SwiGLU FFN
# ═══════════════════════════════════════════════════════════
class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络: down(SiLU(gate(x)) * up(x))"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════
# 5. Transformer Block — 拼在一起
# ═══════════════════════════════════════════════════════════
class DecoderBlock(nn.Module):
    """一个完整的 Decoder 层: Attention (GQA + RoPE + KV Cache) → FFN (SwiGLU)"""
    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_multiplier=4):
        super().__init__()
        assert dim % num_q_heads == 0, "dim 必须能被 num_q_heads 整除"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须是 num_kv_heads 的倍数"

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = dim // num_q_heads

        # Q、K、V 投影
        self.q_proj = nn.Linear(dim, num_q_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # SwiGLU FFN
        hidden_dim = int(dim * ffn_multiplier * 2 / 3)  # Llama 风格近似
        self.ffn = SwiGLUFFN(dim, hidden_dim)

        # Pre-norm
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm  = RMSNorm(dim)

    def forward(self, x, cos, sin, past_kv=None, past_len=0):
        """
        x:        (batch, seq_len, dim)
        past_kv:  (k, v) 上一轮的缓存或 None
        past_len: 已有序列长度（用于 RoPE 偏移 + mask 偏移）
        """
        # —— 1. Pre-norm + 投影 ——
        normed = self.attn_norm(x)
        batch, seq_len, _ = normed.shape

        q = self.q_proj(normed).view(batch, seq_len, self.num_q_heads, self.d_head)
        k = self.k_proj(normed).view(batch, seq_len, self.num_kv_heads, self.d_head)
        v = self.v_proj(normed).view(batch, seq_len, self.num_kv_heads, self.d_head)

        # —— 2. 因果 Mask (仅在 Pre-fill 时需要) ——
        mask = None
        if past_kv is None and seq_len > 1:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # —— 3. Attention (GQA + RoPE + KV Cache) ——
        attn_out, new_kv = gqa_attention(
            q, k, v, cos, sin, past_kv, past_len, mask
        )
        attn_out = self.o_proj(attn_out.reshape(batch, seq_len, -1))

        # —— 4. 残差连接 ——
        x = x + attn_out

        # —— 5. FFN + 残差 ——
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_kv


# ═══════════════════════════════════════════════════════════
# 6. 完整模型: 多层 Decoder + LM Head
# ═══════════════════════════════════════════════════════════
class MiniLLM(nn.Module):
    """
    最小完整 LLM: Embedding → N × DecoderBlock → RMSNorm → LM Head

    用法:
        model = MiniLLM(vocab_size=1000, dim=256, num_q_heads=8, num_kv_heads=4,
                        num_layers=4, max_seq_len=512)
        output = model.generate(prompt_tokens, max_new_tokens=20)
    """
    def __init__(self, vocab_size, dim, num_q_heads, num_kv_heads,
                 num_layers, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.d_head = dim // num_q_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_q_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # 预计算 RoPE cos/sin 表 (不参与训练)
        cos, sin = precompute_rope_freqs(self.d_head, max_seq_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, tokens, past_kvs=None):
        """
        tokens:   (batch, seq_len)
        past_kvs: 每层的 KV Cache 列表，或 None

        返回: (logits, new_past_kvs)
        """
        batch, seq_len = tokens.shape
        past_len = past_kvs[0][0].shape[1] if past_kvs is not None else 0

        x = self.embed(tokens)
        new_past_kvs = []

        for layer in self.layers:
            past_kv = past_kvs.pop(0) if past_kvs is not None else None
            x, new_kv = layer(x, self.rope_cos, self.rope_sin, past_kv, past_len)
            new_past_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_past_kvs

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0):
        """
        自回归生成 — 展示 Pre-fill → Decode 全流程

        prompt: (1, prompt_len) 的 token tensor
        返回: (1, prompt_len + max_new_tokens) 的完整序列
        """
        # ═══ Pre-fill: 一次性吃进 prompt，首 token 也顺便产出 ═══
        logits, past_kvs = self.forward(prompt)
        next_token = self._sample(logits[:, -1], temperature)
        generated = [next_token]

        # ═══ Decode: 逐 token 生成，KV Cache 每步复用 ═══
        for _ in range(max_new_tokens - 1):
            logits, past_kvs = self.forward(next_token, past_kvs)
            next_token = self._sample(logits[:, -1], temperature)
            generated.append(next_token)

        return torch.cat([prompt] + generated, dim=1)

    def _sample(self, logits, temperature):
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1)


# ═══════════════════════════════════════════════════════════
# 7. 跑一下
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 50)
    print("MiniLLM — 完整 Decoder Demo")
    print("=" * 50)

    VOCAB_SIZE = 1000
    DIM = 256
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 4    # GQA: 每 2 个 Q 头共享 1 组 KV
    NUM_LAYERS = 4
    MAX_SEQ_LEN = 256

    model = MiniLLM(VOCAB_SIZE, DIM, NUM_Q_HEADS, NUM_KV_HEADS,
                    NUM_LAYERS, MAX_SEQ_LEN)
    model.eval()

    # 随机造一条 prompt
    prompt = torch.randint(0, VOCAB_SIZE, (1, 8))
    print(f"\nPrompt tokens: {prompt.tolist()}")

    # 生成
    output = model.generate(prompt, max_new_tokens=16, temperature=0.8)
    print(f"Output tokens:  {output.tolist()}")
    print(f"\n输入 {prompt.shape[1]} tokens → 输出 {output.shape[1]} tokens")

    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total/1e6:.2f}M")
    print(f"\n各组件:")
    print(f"  Embedding:   {model.embed.weight.numel()/1e6:.2f}M")
    print(f"  Decoder×{NUM_LAYERS}:  {sum(p.numel() for l in model.layers for p in l.parameters())/1e6:.2f}M")
    print(f"  LM Head:     {model.lm_head.weight.numel()/1e6:.2f}M")
    print(f"  RoPE cos/sin: {model.rope_cos.numel() + model.rope_sin.numel()} (不参与训练)")

    # 验证 KV Cache 效果
    print(f"\n--- KV Cache 效果验证 ---")
    prompt_len = prompt.shape[1]
    print(f"Pre-fill: 输入 {prompt_len} tokens, 每个 token attend 到 {prompt_len} 个位置")
    print(f"Decode 第1步: 输入 1 个 token, Q shape = (1,1,{NUM_Q_HEADS},{DIM//NUM_Q_HEADS}), "
          f"K shape = (1,{prompt_len+1},{NUM_KV_HEADS},{DIM//NUM_Q_HEADS})")
    print(f"  → 只算了 1 个新 token 的 K,V，历史的 K,V 来自缓存")
    print(f"\n--- Repo module mapping ---")
    print(f"  attention/kv-cache.py   -> KV Cache concat in gqa_attention()")
    print(f"  gqa/kv-cache.py         -> repeat_kv() for GQA broadcast")
    print(f"  rope/notes.md           -> precompute_rope_freqs() + apply_rotary_emb()")
    print(f"  ffn/swiglu.py           -> SwiGLUFFN class")
    print(f"  flash-attention/notes.md -> orthogonal: replace attention to save VRAM")
