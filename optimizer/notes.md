# 优化器：AdamW + Gradient Clipping

> AdamW 决定"往哪走、走多远"，Gradient Clipping 是"安全绳"，防止一步踩空摔死。

---

## 一、优化器的进化史（为什么要 AdamW？）

### 1.1 SGD — 最朴素的走法

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot g$$

| 问题 | 类比 |
|------|------|
| 所有参数共用一个步长 | 大人小孩穿同一双鞋，有人嫌大有人嫌小 |
| 没有惯性 | 每一步都重新感受方向，在平路上反复试探 |
| 容易在窄谷震荡 | 左右来回晃，就是不往前走 |

### 1.2 Momentum — 带惯性下山

$$v_t = \gamma v_{t-1} + \eta g, \quad \theta_{t+1} = \theta_t - v_t$$

方向一致 → 加速；方向改变 → 减速缓冲。

### 1.3 AdaGrad — 每人一双鞋

$$G_t = G_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$

**致命缺陷：** G_t 只增不减 → 学习率持续衰减 → 最后原地踏步。

### 1.4 RMSProp — 只记最近的路况

$$v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t^2$$

指数移动平均，学习率不会永久衰减到零。

### 1.5 Adam = Momentum + RMSProp

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

| 组件 | 作用 |
|------|------|
| m_t（一阶矩） | 动量 → 加速收敛 |
| v_t（二阶矩） | 自适应学习率 → 每个参数单独控制步长 |
| 偏差校正 | 开头几步走得稳 |

### 1.6 AdamW — 纠正 Adam 的设计缺陷

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda \cdot \theta_t$$

核心改进：Weight Decay 从自适应学习率中独立出来。

---

## 二、Adam vs AdamW：核心区别

| | Adam | AdamW |
|--|------|-------|
| 公式 | θ = θ - η(m̂/(√v̂+ε) + λθ) | θ = θ - η·m̂/(√v̂+ε) - ηλθ |
| Weight Decay 位置 | 括号**里面**（与梯度相加） | 括号**外面**（独立项） |
| 是否被 √v̂ 缩放 | ✅ 被缩放 | ❌ 不被缩放 |

**Adam 的问题：** λθ 被 1/√v̂ 缩放。当 √v̂ 很大时，Weight Decay 被过度放大 → 参数被过度压向 0。

**AdamW 怎么修：** Weight Decay 独立于 √v̂，每个参数的衰减强度一致。

```python
# Adam（有问题）
theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)

# AdamW（正确）
theta = theta - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * theta
```

---

## 三、AdamW 完整公式

| 符号 | 含义 | 典型值 |
|------|------|--------|
| η | 学习率 | 1e-4 ~ 3e-4 |
| β1 | 一阶矩衰减率 | 0.9 |
| β2 | 二阶矩衰减率 | 0.999 或 0.95 |
| λ | Weight Decay 系数 | 0.01 ~ 0.1 |
| ε | 防除零 | 1e-8 |

更新步骤：

```
1. m_t = β1·m_{t-1} + (1-β1)·g_t         — 一阶矩（动量）
2. v_t = β2·v_{t-1} + (1-β2)·g_t²        — 二阶矩（自适应学习率）
3. m̂_t = m_t / (1-β1ᵗ)                    — 偏差校正
4. v̂_t = v_t / (1-β2ᵗ)                    — 偏差校正
5. θ = θ - η·m̂_t/(√v̂_t + ε) - ηλθ        — 参数更新
```

---

## 四、Weight Decay（权重衰减）

> 给模型戴紧箍咒，不让参数乱长。每次更新参数自动缩小一点点。

从损失函数角度：L_total = L_original + (λ/2)·||θ||²

| 模型 | Weight Decay | 原因 |
|------|-------------|------|
| 小模型 (< 100M) | 0.01 | 模型小，不用太强 |
| GPT-2 | 0.01 | — |
| LLaMA | **0.1** | 超大模型，必须强正则化 |
| Qwen | **0.1** | 同上 |
| DeepSeek | **0.1** | 同上 |

趋势：模型越大，Weight Decay 越大。

---

## 五、Gradient Clipping（梯度裁剪）

### 为什么需要？

```python
正常梯度: [0.5, -0.3, 0.2]
爆炸梯度: [5234.0, -12456.0, 8901.0]  # 更新会直接崩溃
```

### 公式

$$||g|| = \sqrt{\sum_{i=1}^{N} g_i^2}$$

$$g_{\text{clipped}} = \begin{cases} g, & ||g|| \leq C \\ C \cdot \frac{g}{||g||}, & ||g|| > C \end{cases}$$

> 把梯度的"总长度"限制在 C 以内，方向不变，只改大小。

---

## 六、完整训练流程

```python
for step in range(total_steps):
    logits = model(input_ids)
    loss = cross_entropy(logits, labels)
    loss.backward()                        # 梯度可能爆炸

    torch.nn.utils.clip_grad_norm_(        # 梯度裁剪（安全绳）
        model.parameters(), max_norm=1.0
    )

    optimizer.step()                       # AdamW 更新（方向盘）
    optimizer.zero_grad()
```

---

## 七、面试必背

**Q: Adam 和 AdamW 的区别？**
> AdamW 把 Weight Decay 从自适应学习率中解耦。Adam 里 Weight Decay 被 1/√v̂ 缩放，AdamW 让 Weight Decay 独立，训练更稳定。

**Q: AdamW 的更新公式？**
> θ = θ - η·m̂/(√v̂+ε) - ηλθ

**Q: 为什么需要 Gradient Clipping？**
> Transformer 层数深、参数多，梯度容易爆炸。裁剪把梯度的 L2 范数限制在阈值内，防止更新过大导致训练崩溃。

**Q: 为什么 β2 从 0.999 变成 0.95？**
> 0.999 的有效窗口是 1000 步，训练初期自适应学习率变化太慢。0.95 的窗口是 20 步，响应更快。LLaMA、Qwen、DeepSeek 都用 0.95。

---

## 总结

```
AdamW:
├── 一阶矩 m_t → 动量，加速收敛
├── 二阶矩 v_t → 自适应学习率
├── 偏差校正 → 初始几步稳定
└── Weight Decay → 独立，防过拟合

Gradient Clipping:
├── 计算梯度范数
├── 超过阈值则缩放
└── 保持方向不变
```

> AdamW 是方向盘（决定方向），Gradient Clipping 是安全带（防止翻车）。两者缺一不可。

---

*整理时间：2026-07-19*
