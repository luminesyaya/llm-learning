"""
完整预训练脚本 — 在 MiniLLM 上跑通训练闭环

训练目标: Next-Token-Prediction + CrossEntropyLoss
优化器:   AdamW + gradient clipping
调度器:   Cosine schedule with warmup
数据:     随机序列（验证闭环用，真实训练替换为 tokenized corpus）

用法:
    python pretraining/train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math

from decoder.decoder import MiniLLM


# ═══════════════════════════════════════════════════════════
# 1. 随机数据集（占位 — 真实训练替换为 tokenized corpus）
# ═══════════════════════════════════════════════════════════
def make_random_batch(batch_size, seq_len, vocab_size):
    """生成随机 token 序列，labels = input_ids（语言模型自监督）"""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    return tokens, tokens.clone()


# ═══════════════════════════════════════════════════════════
# 2. 训练步骤 — 核心闭环
# ═══════════════════════════════════════════════════════════
def train_step(model, input_ids, labels):
    """
    一次训练步：forward → loss → backward → 返回 loss 值

    Args:
        model:     MiniLLM
        input_ids: (B, T) token ids
        labels:    (B, T) 同 input_ids（语言模型自监督）

    Returns:
        loss: 标量，next-token-prediction 的交叉熵损失
    """
    # —— Forward ——
    logits, _ = model(input_ids)  # (B, T, V)

    # —— Shift 对齐：位置 t 预测 t+1 ——
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()       # (B, T-1)

    # —— Flatten + CrossEntropy ——
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss


# ═══════════════════════════════════════════════════════════
# 3. 评估：计算困惑度（Perplexity）
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, num_batches=5, batch_size=4, seq_len=64, vocab_size=1000):
    """在随机数据上评估 loss 和困惑度"""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        tokens, labels = make_random_batch(batch_size, seq_len, vocab_size)
        loss = train_step(model, tokens, labels)
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / num_batches
    return avg_loss, math.exp(avg_loss)


# ═══════════════════════════════════════════════════════════
# 4. 完整训练循环
# ═══════════════════════════════════════════════════════════
def main():
    # —— 超参数 ——
    VOCAB_SIZE = 1000
    DIM = 256
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 4
    NUM_LAYERS = 4
    MAX_SEQ_LEN = 256

    BATCH_SIZE = 16
    SEQ_LEN = 64
    TOTAL_STEPS = 500
    WARMUP_STEPS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    LOG_EVERY = 50
    EVAL_EVERY = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Steps: {TOTAL_STEPS} | Warmup: {WARMUP_STEPS} | LR: {LEARNING_RATE}")

    # —— 模型 ——
    model = MiniLLM(
        vocab_size=VOCAB_SIZE, dim=DIM,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN
    ).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params/1e6:.2f}M")

    # —— 优化器 ——
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # —— 调度器: Linear warmup → Cosine decay ——
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS]
    )

    # —— 训练 ——
    print(f"\n{'Step':>6s} {'Loss':>8s} {'PPL':>8s} {'LR':>10s}")
    print("-" * 35)

    total_loss = 0.0

    for step in range(1, TOTAL_STEPS + 1):
        tokens, labels = make_random_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        tokens, labels = tokens.to(device), labels.to(device)

        # Forward + loss
        loss = train_step(model, tokens, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # —— Logging ——
        if step % LOG_EVERY == 0:
            avg_loss = total_loss / LOG_EVERY
            ppl = math.exp(avg_loss)
            lr = scheduler.get_last_lr()[0]
            print(f"{step:>6d} {avg_loss:>8.4f} {ppl:>8.2f} {lr:>10.2e}")
            total_loss = 0.0

        # —— Evaluation ——
        if step % EVAL_EVERY == 0:
            eval_loss, eval_ppl = evaluate(model, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
            print(f"  [Eval @ step {step}] loss={eval_loss:.4f} ppl={eval_ppl:.2f}")

    print(f"\nDone. Final LR: {scheduler.get_last_lr()[0]:.2e}")

    # —— 验证生成（确保训练没把模型搞坏） ——
    print("\n--- Sanity check: generate ---")
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 8)).to(device)
    output = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Prompt:  {prompt.tolist()}")
    print(f"Output:  {output.tolist()}")


if __name__ == '__main__':
    main()
