import torch
import torch.nn.functional as F

def repeat_kv(kv, n_rep):
    """
    把 K 或 V 在 head 维度上重复 n_rep 次
    kv: (batch, seq_len, num_kv_heads, d_head)
    n_rep: 每个 KV 头要服务几个 Q 头
    """
    if n_rep == 1:
        return kv
    return kv.repeat_interleave(n_rep, dim=2)  # 在 head 维度上重复

def gqa_attention_with_cache(q, k, v, past_kv=None, mask=None, n_rep=1):
    """
    GQA（分组查询注意力）+ KV Cache
    
    Args:
        q: (batch, 1, num_q_heads, d_head)  # Decode 阶段，只有 1 个新 token
        k: (batch, 1, num_kv_heads, d_head)
        v: (batch, 1, num_kv_heads, d_head)
        past_kv: (past_k, past_v)，每个形状 (batch, seq_len_hist, num_kv_heads, d_head)
        mask: 因果掩码（可选）
    
    Returns:
        output: (batch, 1, num_q_heads, d_head)
        new_past_kv: 更新后的缓存
    """
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    n_rep = num_q_heads // num_kv_heads  # 每个 KV 头服务几个 Q 头
    
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=1)  # 序列维度拼接
        v = torch.cat([past_v, v], dim=1)
    new_past_kv = (k, v)
    
    k = repeat_kv(k, n_rep)  # 在 head 维度上重复
    v = repeat_kv(v, n_rep)
    
    d_head = q.shape[-1]
    
    q = q.transpose(1, 2)  # (batch, num_q_heads, seq_len_q, d_head)
    k = k.transpose(1, 2)  # (batch, num_q_heads, seq_len_k, d_head)
    v = v.transpose(1, 2)  # (batch, num_q_heads, seq_len_k, d_head)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # (batch, num_q_heads, seq_len_q, d_head)
    
    output = output.transpose(1, 2)  # (batch, seq_len_q, num_q_heads, d_head)
    return output, new_past_kv