# LLM Learning Notes

从零学习大语言模型核心技术的笔记与代码实现。

## 目录

- `attention/` — 注意力机制
  - `notes.md` — Scaled Dot-Product Attention、Multi-Head Attention 原理笔记
  - `multi-head-attention.py` — 标准多头注意力的 PyTorch 实现
  - `kv-cache.py` — 带 KV Cache 的多头注意力实现
- `gqa/` — 分组查询注意力
  - `kv-cache.py` — GQA + KV Cache 实现
- `flash-attention/` — Flash Attention
  - `notes.md` — Flash Attention 面试高频问题笔记
- `rope/` — 旋转位置编码
  - `notes.md` — RoPE 原理与长上下文扩展方案
- `ffn/` — 前馈网络
  - `swiglu.py` — SwiGLU 激活函数的 FFN 实现
- `decoder/` — 完整 Decoder 实现
  - `decoder.py` — 整合 RMSNorm + RoPE + GQA + KV Cache + SwiGLU 的完整 Decoder + 自回归生成
- `pretraining/` — 预训练
  - `notes.md` — Next-Token-Prediction 训练目标、CrossEntropyLoss、Shift 对齐
  - `train.py` — 完整训练脚本 (AdamW + gradient clipping + warmup cosine scheduler)
- `optimizer/` — 优化器与训练稳定性
  - `notes.md` — SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW 进化史，Gradient Clipping，Weight Decay
- `scaling-law/` — 规模规律
  - `notes.md` — Scaling Law 应用篇：超参数外推、Chinchilla 配比、面试 6 问
- `data-engineering/` — 数据工程
  - `notes.md` — 数据来源、质量过滤、MinHash 去重、Data Mixture、Token Budget
  - `pipeline.py` — 数据处理管道 Demo（规则过滤 + MinHash 去重 + PII 脱敏）
