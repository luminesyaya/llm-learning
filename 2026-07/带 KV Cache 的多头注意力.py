import torch
import torch.nn.functional as F

def kv_cache_attention(q, k, v, past_kv=None, mask=None):
    """
    带 KV Cache 的多头注意力
    
    Args:
        q: (batch, seq_len_q, num_heads, d_head)
        k: (batch, seq_len_k, num_heads, d_head)
        v: (batch, seq_len_k, num_heads, d_head)
        past_kv: (past_k, past_v)，每个 (batch, seq_len_hist, num_heads, d_head)
        mask: 因果掩码，(seq_len_q, seq_len_k)，或 None
    
    Returns:
        output: (batch, seq_len_q, num_heads, d_head)
        new_past_kv: 更新后的缓存
    """
    # 1. 处理缓存
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=1)  # 序列维度拼接
        v = torch.cat([past_v, v], dim=1)
    new_past_kv = (k, v)
    
    # 2. 提取 d_head
    d_head = q.shape[-1]
    
    # 3. 转置：把 head 移到第 2 维
    q = q.transpose(1, 2)  # (batch, num_heads, seq_len_q, d_head)
    k = k.transpose(1, 2)  # (batch, num_heads, seq_len_k, d_head)
    v = v.transpose(1, 2)  # (batch, num_heads, seq_len_k, d_head)
    
    # 4. 计算分数（缩放！）
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    # scores: (batch, num_heads, seq_len_q, seq_len_k)
    
    # 5. 掩码（如果传入）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 6. Softmax + 加权求和
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len_q, d_head)
    
    # 7. 恢复形状
    output = output.transpose(1, 2)  # (batch, seq_len_q, num_heads, d_head)
    
    return output, new_past_kv