# 注意力机制（Attention Mechanism）

---

## 1. 直觉：从"通读"到"检索"

传统 Seq2Seq 把整个源句压成一个定长向量（encoder 最后一步的隐状态），句子一长就成了瓶颈。

**注意力的核心思想**：解码每一步时，不要只看那一坨压缩向量，而是让 decoder 动态地"回看" encoder 所有时刻的隐状态，决定当前该关注哪些位置。

形式上，每一步解码都做一次**软寻址（soft addressing）**——把所有位置的 value 按"查询和键的匹配度"加权求和。

---

## 2. Scaled Dot-Product Attention（缩放点积注意力）

这是 Transformer 使用的基本形式。

### 2.1 三个输入

| 符号 | 维度 | 含义 |
|---|---|---|
| `Q` (Query) | n × d_k | 当前要"查"的东西 |
| `K` (Key) | m × d_k | 每个位置的"标签" |
| `V` (Value) | m × d_v | 每个位置的"内容" |

### 2.2 核心公式

```
Attention(Q, K, V) = softmax( Q · K^T / √d_k ) · V
```

### 2.3 逐步拆解

**Step 1 — 计算注意力分数（Attention Scores）**

```
S = Q × K^T    （形状：n × m）
```

S[i][j] 是第 i 个 query 与第 j 个 key 的内积。内积越大 → 方向越一致 → 越"相关"。

**Step 2 — 缩放（Scaling）**

```
S_hat = S / √d_k
```

**为什么要除以 √d_k？**

假设 q, k 的各分量独立同分布，均值 0，方差 1，则内积 `q · k = Σ q_i · k_i` 的方差是 `d_k`。

当 d_k 很大时（如 64 或 128），内积数值范围会很大，导致 softmax 落入**饱和区**——大的巨大、小的接近零，概率分布趋近 one-hot，梯度几乎为零。

除以 √d_k 把方差拉回 1，保持梯度正常流动。

**Step 3 — Softmax 归一化**

```
W = softmax(S_hat)
W[i][j] = exp(S_hat[i][j]) / Σ_k exp(S_hat[i][k])
```

每一行 W[i] 是一个概率分布（和为 1），表示第 i 个 query 对各 key 位置的注意力分配。

**Step 4 — 加权求和**

```
Output = W × V    （形状：n × d_v）
```

第 i 行的输出 = 所有 value 的加权和，权重是第 i 个 query 的注意力分布。

### 2.4 单条 query 的向量形式

对单个 query 向量 q（维度 d_k）：

```
α_j = exp(q·k_j / √d_k) / Σ_t exp(q·k_t / √d_k)    ← 注意力权重

output = Σ_j α_j · v_j                                ← 加权求和
```

---

## 3. 自注意力 vs 交叉注意力

| 模式 | Q 来源 | K 来源 | V 来源 | 含义 |
|---|---|---|---|---|
| Self-Attention | 自身序列 | 自身序列 | 自身序列 | 序列内部交互，"自己看自己" |
| Cross-Attention | Decoder | Encoder | Encoder | Decoder 查询 Encoder 的信息 |
| Masked Self-Attention | Decoder | Decoder | Decoder | 加因果遮罩，只看当前位置之前 |

---

## 4. 自注意力的完整矩阵计算

给定输入 `X`（n 个 token，d_model 维嵌入），通过三个可训练的权重矩阵投影：

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

其中：
- W_Q, W_K：d_model × d_k
- W_V：d_model × d_v

代入注意力公式：

```
SelfAttention(X) = softmax( X·W_Q · W_K^T · X^T / √d_k ) · X · W_V
```

### 直观理解

- **W_Q**（Query 投影）：把当前位置投射成"我在找什么"
- **W_K**（Key 投影）：把每个位置投射成"我是什么（可以匹配的标签）"
- **W_V**（Value 投影）：把每个位置投射成"如果被关注，我输出什么内容"

Q 和 K 的相似度决定"关注多少"；V 决定"关注到什么内容"。

---

## 5. 多头注意力（Multi-Head Attention）

单头注意力可能只捕捉到一种"相关性模式"（比如都盯着主语）。多头注意力让模型并行学习多种不同的注意力模式。

### 5.1 公式

设 d_k = d_v = d_model / h（h 为头数）：

```
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)    ← 第 i 个头，在降维子空间计算

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O   ← 拼接后投影
```

### 5.2 参数量

每个头：d_k × d_model × 3 = d_model² / h × 3

h 个头总计：≈ 3 × d_model²

**与单头（d_k = d_model）一致，没有额外开销。**

### 5.3 各头学到的模式示例

- 头 1：关注相邻词（局部语法）
- 头 2：关注主语-谓语关联（远距离依赖）
- 头 3：关注标点/分隔符（句子边界）
- 头 4：关注指代（代词 → 先行词）

---

## 6. 因果遮罩（Causal Mask / Masked Self-Attention）

### 6.1 问题

Decoder 在做自注意力时，位置 i 不应该看到位置 i+1, i+2, ... 的内容（那是"未来"）。

### 6.2 解决方案

在 softmax 之前，把未来位置的分数设为 -∞：

```
M[i][j] = 0      (i ≥ j, 当前位置及之前)
        = -∞     (i < j, 未来位置，不能看)
```

因为 exp(-∞) = 0，这些位置的权重直接归零。

```
MaskedAttention(Q, K, V) = softmax( Q·K^T / √d_k + M ) · V
```

### 6.3 可视化（注意力矩阵）

序列 "The cat sat" 的因果遮罩效果（√ = 可关注，× = 被遮罩）：

```
        The   cat   sat
The  [   √     ×     ×  ]
cat  [   √     √     ×  ]
sat  [   √     √     √  ]
```

---

## 7. 计算复杂度

对序列长度 n，维度 d：

| 步骤 | 复杂度 |
|---|---|
| Q × K^T | O(n² · d) ← **瓶颈** |
| Softmax | O(n²) |
| × V | O(n² · d) |
| **总计** | **O(n² · d)** |

**与序列长度平方成正比**。这就是长序列对 Transformer 是挑战的原因。

**优化方向**：
- **FlashAttention**：利用 GPU 内存层级，分块计算 + 减少 HBM 读写，不改变复杂度但大幅提速
- **稀疏注意力**：每个位置只看局部窗口 + 少数全局位置，降到 O(n · k · d)
- **线性注意力**：用核函数改写 attention，将 Q·K^T · V 的计算顺序改为 Q · (K^T · V)，降到 O(n · d²)

---

## 8. 位置编码（Positional Encoding）

注意力本身对位置不敏感——打乱输入序列，输出只跟着打乱，不会变值。

### 8.1 正弦位置编码（Transformer 原版）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- pos：位置索引
- i：维度索引
- 每个维度对应不同频率的正弦波，模型可学会关注相对位置

### 8.2 可学习位置编码

直接为每个位置训练一个向量，加（或拼）到 token embedding 上。GPT 系列常用。

### 8.3 RoPE（旋转位置编码）

通过旋转矩阵将位置信息编码到 Q 和 K 的内积中，使得 Q·K 只依赖于相对位置。Llama、Qwen 等主流模型使用。

---

## 9. 注意力在 LLM 架构中的位置

以 GPT 这类 **decoder-only** 架构为例（目前主流 LLM）：

```
输入 Token
    ↓
[Embedding + Positional Encoding]     ← 初始表示
    ↓
┌──────────────────────────┐
│  Transformer Block × N   │        ← 堆叠 N 层（如 GPT-3: 96 层）
│                          │
│  ┌ Masked Multi-Head ──┐ │
│  │ Self-Attention      │ │        ← 注意力在这里！让每个 token 看上下文
│  └──────────────────────┘ │
│         (+ 残差 + LayerNorm)
│  ┌──────────────────────┐ │
│  │ Feed-Forward Network│ │        ← 位置独立的知识存储/变换
│  └──────────────────────┘ │
│         (+ 残差 + LayerNorm)
└──────────────────────────┘
    ↓
[LM Head]                            ← 输出投影：向量 → 词表概率
```

---

## 10. 一句话总结

> **注意力机制 = 对 Value 做加权求和，权重来自 Query 与各 Key 的相似度。**
>
> 它是 Transformer 的核心计算原语，让每个位置的输出动态融合上下文中所有位置的信息，权重完全由数据学习决定。

---

## 11. 关键要点速记

| 要点 | 说明 |
|---|---|
| 本质 | 软寻址：按 Q-K 匹配度，加权取 V |
| 缩放因子 | √d_k，防止方差膨胀导致 softmax 饱和 |
| Self-Attention | Q = K = V = 同一序列，学习序列内部关系 |
| Cross-Attention | Q 来自 Decoder，K/V 来自 Encoder，实现编码器-解码器交互 |
| 多头 | h 个独立注意力在降维子空间并行计算，学习不同模式 |
| 因果遮罩 | 只允许关注当前位置及之前，保证自回归生成不泄题 |
| 复杂度 | O(n²·d)，瓶颈在 Attention 矩阵的 n×n |
| 位置编码 | 注意力本身无序感知，需额外编码位置信息 |
| RoPE | 主流位置编码方案，相对位置编码 + 绝对位置编码的统一 |

---

*整理时间：2026-07-17*
