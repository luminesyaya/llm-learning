# 数据工程：从原始网页到训练语料

> 模型架构趋同（都是 Decoder-only Transformer），数据才是护城河。
> GPT-4 之后，每家都在卷数据而不是卷架构。

---

## 目录

1. [全流程概览](#一全流程概览)
2. [数据来源](#二数据来源)
3. [质量清洗](#三质量清洗)
4. [去重](#四去重)
5. [Data Mixture](#五data-mixture)
6. [Token Budget](#六token-budget)
7. [面试必背](#七面试必背)

---

## 一、全流程概览

```
Common Crawl 原始 WARC 文件 (PB 级)
    ↓
[1] HTML 提取              trafilatura / readability → 纯文本
    ↓
[2] 语言识别               扔掉非目标语言（fastText / CLD3）
    ↓
[3] 规则粗筛               长度、字符比例、停用词比 → 扔明显垃圾
    ↓
[4] 模型精筛               KenLM 困惑度 / fastText 分类器 → 打分取阈值
    ↓
[5] 去重                   URL → MinHash LSH → 句级 exact match
    ↓
[6] PII 脱敏               邮箱、电话、身份证号替换为占位符
    ↓
[7] Data Mixture           各领域按权重采样，拼接训练序列
    ↓
[8] Tokenization + 打包    文本 → token IDs → 填满 2048/4096 的序列
    ↓
训练就绪
```

每一步都可以独立展开，每条流水线都对应大量工程细节。

---

## 二、数据来源

### 2.1 核心来源

#### Common Crawl（一切的基础）

- 每月爬取整个互联网，WARC 格式存储
- 单月数据量 200-300 TB，含数十亿页面
- 质量极低：广告、SEO 垃圾、乱码、重复内容占多数
- 几乎所有大模型都从这里起步，关键在于怎么洗

#### Wikipedia

- 多语言、高质量、持续更新
- 规模小（英文 ~6M 文章，~4B tokens）
- 几乎所有模型都用，但占比通常不高（3-5%）

#### Books

- 长文本、高质量、知识密集
- 来源：Books3、Gutenberg、图书馆扫描
- 版权灰色地带（Books3 已因版权争议下架）

#### Academic Text (ArXiv / PubMed)

- 科学论文、数学公式、专业术语
- 质量极高，但领域窄
- 对数学推理、科学问答帮助大

#### Code (GitHub / StackExchange)

- 代码占比虽少（4-5%）但对推理能力提升显著
- The Stack：HuggingFace 整理的公开代码数据集
- 需要处理许可证问题

#### Web Text Aggregates（预处理过的 Web 数据）

| 数据集 | 规模 | 推出方 | 特点 |
|--------|------|--------|------|
| **C4** | 175B tokens | Google | 从 CC 过滤，规则为主 |
| **The Pile** | 800GB | EleutherAI | 22 个子集精心配比 |
| **FineWeb** | 15T tokens | HuggingFace | 2024 年新发布，质量最高 |
| **RedPajama** | 1.2T tokens | Together | 复现 LLaMA 配方 |
| **DCLM** | 多种规模 | 多机构联合 | 系统研究数据清洗效果 |

#### 合成数据 (Synthetic Data)

> **当前最热的趋势。** 用强模型生成训练数据，喂给小模型。

- **Phi-3**：微软用 GPT-4 生成"教科书"级数据，3.8B 的小模型堪比 7B
- **DeepSeek-V3**：用 R1 生成推理链作为训练数据
- **Llama 3**：用 Llama 2 生成合成数据做数据增强

**合成数据的好处**：
- 质量可控（prompt 决定风格和难度）
- 无限扩展（不依赖爬虫）
- 可以针对性补短板（数学弱就大量生成数学题）

**合成数据的风险**：
- 模型"近亲繁殖"（生成的错误被下一代替换不掉）
- 多样性丧失（生成模型有自己的偏见）

---

### 2.2 各模型数据来源对比

| 模型 | Common Crawl | 代码 | 学术 | 书籍 | 合成数据 |
|------|:---:|:---:|:---:|:---:|:---:|
| GPT-3 | 60% (过滤后) | 少量 | 少量 | 少量 | — |
| LLaMA 1 | 67% | 4.5% | 2.5% | 4.5% | — |
| LLaMA 2 | 更多 | 更多 | 更多 | 更多 | — |
| LLaMA 3 | — | — | — | — | 大量 |
| DeepSeek-V3 | 大量 | 大量 | 大量 | 大量 | 大量 |
| Phi-3 | — | — | — | — | **几乎全部** |

> **趋势：从纯爬取 → 爬取+合成混合 → 合成占比越来越高。**

---

## 三、质量清洗

### 3.1 HTML 提取

网页不是纯文本。第一步是剥离 HTML 标签、导航栏、广告、footer。

**常用工具**：

| 工具 | 语言 | 特点 |
|------|------|------|
| **trafilatura** | Python | 最常用，C4/FineWeb 都在用 |
| **readability-lxml** | Python | Mozilla 可读性算法 |
| **jusText** | Python | 启发式去锅炉 |
| **boilerpy3** | Python | boilerpipe 的 Python 移植 |
| **html-text** | Python | 极简，只提取可见文本 |

**Trafulatura 工作原理（简化）**：

```
1. 解析 HTML DOM 树
2. 对每个节点提取特征（文本密度、链接密度、tag 类型）
3. 训练一个分类器判断节点是"正文"还是"非正文"
4. 只保留被判定为正文的节点文本
```

### 3.2 语言识别

目标：扔掉非目标语言的文档。

**常用工具**：

| 工具 | 速度 | 准确率 | 语言数 |
|------|------|--------|--------|
| **fastText lang-id** | 快 | 高 | 176 |
| **CLD3** (Google) | 最快 | 高 | 107 |
| **lingua-py** | 慢 | 最高 | 75 |
| **langdetect** | 中 | 中 | 55 |

实际做法：
- 对长文档，取前 100-200 字符做判断（够用了）
- 混入太多外语的文档也扔掉（如中英混排 > 30% 非目标语言）

### 3.3 启发式规则过滤

这是第一道防线，速度最重要。用 CPU 就能跑，不需要 GPU。

```python
# 一条文档被丢弃，如果满足任一条件：
filters = {
    # 长度
    "char_len < 100":            False,  # 太短没信息
    "word_len < 20":             False,  # 同上

    # 字符分布
    "non_alpha_ratio > 0.5":     False,  # 半数是标点/数字 → 乱码
    "caps_ratio > 0.5":          False,  # 大写过半 → 可能是标题党或垃圾
    "digit_ratio > 0.3":         False,  # 数字太多 → 可能是表格/数据dump

    # 词的统计
    "avg_word_len > 10":         False,  # 词太长 → 可能是无意义字符串
    "avg_word_len < 3":          False,  # 词太短 → 可能是代码片段或垃圾

    # 语言特征
    "stopword_ratio < 0.1":      False,  # 停用词太少 → 不像自然语言
    "unique_word_ratio < 0.3":   False,  # 词重复太多 → 机器人文本

    # 结构特征
    "bullet_ratio > 0.5":        False,  # 列表过多 → 不是完整句子
    "mean_line_len < 20":        False,  # 行太短 → 可能是菜单/目录

    # 垃圾特征
    "curly_bracket_ratio > 0.1": False,  # 太多花括号 → JS/CSS残留
    "lorem_ipsum_detected":      False,  # 检测到占位文本
}
```

**FineWeb 实际使用的 18 条规则（2024）**：

1. 字符数 < 100 → 扔
2. 非字母比例 > 0.5 → 扔
3. 大写比例 > 0.5 → 扔
4. 停用词比例 < 0.1 → 扔
5. 平均词长 > 10 → 扔
6. 平均词长 < 3 → 扔
7. 唯一词比例 < 0.3 → 扔
8. 行均字符 < 20 → 扔
9. 列表项比例 > 0.5 → 扔
10. 省略号 "..." 出现 > 5 次 → 扔
11. 包含 "lorem ipsum" → 扔
12. 花括号比例 > 0.1 → 扔
13. 包含 "javascript" 关键字 → 扔
14. 包含 "cookie" 且字符 < 500 → 扔
15. 包含 "sign up" / "log in" + 字符 < 500 → 扔
16. 包含 "Terms of Service" / "Privacy Policy" → 扔
17. 文档重复 n-gram 过于集中 → 扔
18. 语言检测置信度 < 0.65 → 扔

### 3.4 困惑度过滤（Perplexity Filtering）

> 一个好语言模型对一个正常句子不应该感到"惊讶"。

**做法**：

1. 在高质量语料（Wikipedia）上训练一个轻量级 5-gram 语言模型（KenLM）
2. 对每篇文档，用它计算困惑度（perplexity）
3. 困惑度过高 = 语言不自然 = 可能是垃圾
4. 设置阈值，扔掉困惑度 > 阈值的文档

**为什么用 KenLM 而不是 Transformer？**

- KenLM 极快（CPU 上每秒处理 GB 级文本）
- 效果够用：5-gram 已经能区分"正常英语"和"乱码"
- 对于 TB 级数据，速度是第一优先级

```python
# 伪代码
kenlm_model = kenlm.Model('wiki_5gram.arpa')  # Wikipedia 上训练的 5-gram
score = kenlm_model.score(document)             # log likelihood
ppl = 10 ** (-score / len(document.split()))   # perplexity

if ppl > 1000:   # 困惑度太高
    discard(document)
```

### 3.5 分类器过滤（Classifier-based Filtering）

> 训练一个二分类器，区分"好文本"和"垃圾文本"。

**fastText 做法（CCNet / FineWeb）**：

1. 正例：Wikipedia 文本
2. 负例：从 Common Crawl 随机采样
3. 训练 fastText 二分类器
4. 对所有文档打分，保留分数 > 阈值的

**fastText 为什么好用？**

- 训练极快（分钟级）
- 推理极快（CPU 友好）
- 效果不差（基于 n-gram 特征，对文本质量敏感）

**FineWeb 的多阶段打分**：

```
阶段 1: 规则过滤器 → 扔掉约 50% 的页面
阶段 2: fastText 分类器 → 保留得分 > 0.4 的
阶段 3: KenLM 困惑度 → 保留 ppl < 1000 的
阶段 4: 微调 BERT 打分 → 对余下文本精细排序
```

---

## 四、去重

### 4.1 为什么要去重？

不是"可选的优化"，而是**必须做**：

1. **避免过拟合/记忆**：重复 10 次的数据，模型会背下来而不是理解
2. **测试集污染**：训练集和测试集有重叠 → benchmark 分数虚高
3. **浪费算力**：重复 token 吃掉宝贵的 compute budget
4. **降低多样性**：重复数据挤占了不同内容的曝光机会

| 不去重的后果 | 具体案例 |
|-------------|---------|
| Benchmark 污染 | BigBench 发现 C4 训练集和很多测试集重叠 |
| 记忆效应 | GPT-2 被发现逐字复述训练数据 |
| 性能损失 | DCLM 实验：去重后同算力下 loss 更低 |

### 4.2 去重分级策略

```
第一级: URL 去重
   → 同一 URL = 同一文档，哈希 URL，只保留一个副本
   → 最简单的，但只能去完全一样的页面

第二级: 文档级模糊去重 (MinHash LSH)
   → 内容相似的文档（转载、镜像、改几个词的抄袭）
   → 业界标准方案

第三级: 行级/句级模糊去重 (Suffix Array / Exact Substring)
   → 去掉重复出现的模板文本（网站 footer、法律声明、相同段落）
   → 计算量最大，通常放在最后

第四级: 语义去重 (Embedding-based)
   → 用文本嵌入找语义相同的句子（即使措辞不同）
   → 最激进，DeepSeek 在用
```

### 4.3 MinHash 详解

> 面试必考。MinHash 可以快速估算两个集合的 Jaccard 相似度，而不需要两两比较所有元素。

**第 1 步：Shingling（把文档切成 n-gram 集合）**

```python
doc = "the cat sat on the mat"
# 3-shingles:
shingles = {"the cat sat", "cat sat on", "sat on the", "on the mat"}
```

**第 2 步：MinHash 签名**

取 k 个哈希函数 h₁, h₂, ..., hₖ。对文档的 shingle 集合 S：

```
sig(S) = [min(h₁(s) for s in S), min(h₂(s) for s in S), ..., min(hₖ(s) for s in S)]
```

签名长度 k 通常取 128 或 256。

**第 3 步：估算 Jaccard**

两文档签名中相同位置的比例 ≈ 真实的 Jaccard 相似度：

```python
jaccard_estimate = sum(1 for a, b in zip(sig_A, sig_B) if a == b) / k
```

**第 4 步：LSH（Locality Sensitive Hashing）加速**

> 百万文档两两比较 MinHash 签名仍然是 O(n²)。LSH 把 O(n²) 降到近似 O(n)。

做法：
- 把 128 维签名切成 b 个 band，每个 band 有 r 行（b × r = 128）
- 如果两个文档在**任意一个 band 内完全匹配**，就认为候选对
- 只对候选对做精确 Jaccard 验证
- 调参：b 越大 → 召回越高，误判越多；r 越大 → 精度越高，漏检越多

```
例：k=128, b=16, r=8
  band 0: sig[0:8]   → 哈希 → 桶 A
  band 1: sig[8:16]  → 哈希 → 桶 B
  ...
  band 15: sig[120:128] → 哈希 → 桶 P

  文档 X 和 Y 的 band 3 哈希到同一个桶 → 候选对 → 精确验证
```

**阈值**：相似度 > 0.8 就去掉的文档，对应的 (b, r) 组合选法：

| 目标阈值 | 常用 (b, r) |
|----------|------------|
| 0.5 | b=20, r=6 |
| 0.7 | b=16, r=8 |
| 0.8 | b=10, r=12 |

### 4.4 Suffix Array 去重（句级）

> Google C4 论文里的方法，用于去掉跨文档重复的句子。

**做法**：

1. 把所有文档拼接，建立后缀数组（suffix array）
2. 找所有出现次数 ≥ 2 的连续子序列
3. 这些子序列如果是长于 50 个 token 的完整句子 → 标记为重复
4. 去掉这些重复出现的文本段

**效果**：能去掉"网站 footer"、"版权声明"、"转载说明"这类跨文档模版文本。

### 4.5 各模型去重实践

| 模型 | URL去重 | MinHash | 句级去重 | 语义去重 |
|------|:---:|:---:|:---:|:---:|
| GPT-3 | ✅ | ✅ LSH | ✅ | — |
| LLaMA | ✅ | ✅ | ✅ 行级 | — |
| DeepSeek | ✅ | ✅ | ✅ | ✅ |
| DCLM | ✅ | ✅ | ✅ | — |

> DeepSeek 最激进：在 MinHash 后又加了一层语义去重（用嵌入向量的余弦相似度），进一步压缩数据但保留多样性。

### 4.6 去重的陷阱

- **跨语言去重要小心**：同一内容的多语言翻译不应被去重（对多语言模型是有效数据）
- **代码去重要谨慎**：完全相同的代码片段去重没问题，但相似架构的代码可能是不同项目
- **引文/引用不应去重**：学术论文大量引用同一段话是正常的

---

## 五、Data Mixture

### 5.1 核心问题

> 不同来源的数据知识密度不同。Wikipedia 的一句话可能比一个随机博客的一整段更有价值。怎么给各类数据配权重？

### 5.2 静态配比

预先设定比例，整个训练过程不变。

**LLaMA 1 的配比（经典参考）**：

| 数据源 | 占比 | 为什么是这个数 |
|--------|:---:|------|
| Common Crawl | 67.0% | 主体，量大管饱 |
| C4 | 15.0% | 相对干净的 CC 子集 |
| GitHub | 4.5% | 代码，提升推理 |
| Wikipedia | 4.5% | 高质量知识 |
| Books | 4.5% | 长文本连贯性 |
| ArXiv | 2.5% | 科学技术 |
| Stack Exchange | 2.0% | 问答，提升指令理解 |

**怎么决定这些数字？**

1. 参考已有模型的配比
2. 在小模型上做配比消融实验
3. 根据下游任务调整（代码多点 → 推理强，书籍多点 → 知识强）
4. **大量试出来的，不是算出来的**

### 5.3 动态配比

> 固定比例的问题是：某类数据模型提前"学会"了还在反复喂，浪费算力。

**DeepSeek 的做法（在线数据课程学习）**：

1. 训练过程中持续监控每类数据的 loss
2. 某类 loss 下降快 → 减少该类采样权重（已经学会了，不用喂了）
3. 某类 loss 下降慢 → 增加该类采样权重（还没学会，多喂点）

```python
# 伪代码
if loss_drop_rate[domain] > average_loss_drop_rate:
    sampling_weight[domain] *= 0.9   # 下降太快 → 减少
else:
    sampling_weight[domain] *= 1.1   # 下降太慢 → 增加
sampling_weight = normalize(sampling_weight)
```

**DoReMi（Domain Reweighting with Minimax Optimization，Stanford 2023）**：

> 用小模型自动学习最优配比。

1. 训练一个小的 reference model（如 280M）
2. 同时训练一个 domain weight optimizer
3. Domain weights 被优化来最大化 reference model 的 worst-domain loss
4. 直觉：给模型最不擅长的领域更多权重
5. 最后把这些 weights 用到大模型训练

**效果**：DoReMi 自动找到的配比，在下游任务上比手动调参好 6-8%。

### 5.4 多层配比

实际训练中，配比不止一层：

```
Level 1: 大领域配比（代码 vs 网页 vs 书籍）
Level 2: 子领域配比（Python vs Java vs JS）
Level 3: 具体来源配比（GitHub vs GitLab vs BitBucket）
Level 4: 文档内配比（长文档 vs 短文档）
```

每一层都有自己的权重，最终采样概率 = 各层权重的乘积。

### 5.5 一个关键发现

> 代码只有 4.5%，却对推理能力提升巨大。这说明**不是所有 token 价值相等**，需要从"下游任务需要什么能力"出发设计配比，而不是"什么数据多就多喂什么"。

---

## 六、Token Budget

### 6.1 核心问题

> 给定 N 个参数的模型，应该喂多少 D 个 token？

### 6.2 两条路线

**Kaplan (OpenAI, 2020)**：参数量应该比数据量增长更快

$$N_{opt} \propto C^{0.73}, \quad D_{opt} \propto C^{0.27}$$

→ GPT-3：175B 参数，只用了 300B tokens（D/N = 1.7）

**Chinchilla (DeepMind, 2022)**：参数和数据应等比例增长

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

$$D \approx 20 \times N$$

→ 按这个标准，GPT-3 严重"欠数据"。

### 6.3 Chinchilla 的核心结论

| 参数量 N | 最优 Token 量 D | 最优算力 C (FLOPs) |
|----------|:---:|:---:|
| 1B | 20B | 1.2e20 |
| 7B | 140B | 5.9e21 |
| 13B | 260B | 2.0e22 |
| 70B | 1.4T | 5.9e23 |
| 175B | 3.5T | 3.7e24 |

算力公式：C ≈ 6ND（前向 2ND + 反向 4ND）

### 6.4 为什么实际都"超喂"？

Chinchilla 是 **compute-optimal**（给定算力，loss 最低）。但实际部署时：

| 考量 | 小模型+大数据 | 大模型+小数据 |
|------|:---:|:---:|
| 推理成本 | ✅ 便宜 | ❌ 贵 |
| 推理速度 | ✅ 快 | ❌ 慢 |
| 部署门槛 | ✅ 手机可跑 | ❌ 需多张 GPU |
| 训练成本 | ✅ 便宜 | ❌ 贵 |
| loss 下限 | 稍高 | 稍低 |

> **推理省钱比训练省钱更重要**（训练一次，推理亿次）。

所以都在"超喂"：用小一点的模型（相对 Chinchilla 建议），喂远超 20N 的数据。

| 模型 | N | D | D/N | 为什么 |
|------|:---:|:---:|:---:|------|
| Chinchilla | 70B | 1.4T | 20x | compute-optimal 基线 |
| LLaMA 1 | 65B | 1.4T | 22x | 接近 Chinchilla |
| LLaMA 2 | 70B | 2T | 29x | 开始超喂 |
| LLaMA 3 | 8B | 15T | **1875x** | 极致超喂：小模型+海量数据 |
| DeepSeek-V3 | 671B (37B激活) | 14.8T | — | MoE，激活参数少 |

### 6.5 多个 Epoch 还是更多 Unique Token？

> **优先保证 unique token 量，再考虑多 epoch。**

- Unique token < 20N 时：加 unique token 收益大
- Unique token > 20N 后：多 epoch 也有收益但递减
- 一般不超过 4 个 epoch（FineWeb 实验结论：4 epoch 后指标走平）

### 6.6 训练末尾的退火（Annealing）

> 在训练最后阶段，只用最高质量的数据（Wikipedia、Books），降学习率跑几千步。

**为什么有用？**

- 最后几步影响最终模型质量最大
- 高质量数据让模型"收尾"更漂亮
- LLaMA 3、DeepSeek-V3 都用这个技巧

```python
# 伪代码
if step > total_steps - 5000:
    data_sampler = HighQualitySampler()  # 只用 Wikipedia + Books
    lr = lr * 0.1                        # 降学习率
```

---

## 七、面试必背

### Q1：数据工程全流程是什么？

> 原始网页 → HTML 提取 → 语言过滤 → 规则粗筛 → 模型精筛（fastText/KenLM） → 去重（URL → MinHash LSH → 句级） → PII 脱敏 → 配比混合 → Tokenization → 训练。

### Q2：怎么判断文本质量？有哪些过滤器？

> 三级过滤：规则（长度、字符分布、停用词比）→ 统计（KenLM 困惑度）→ 模型（fastText 分类器、BERT 打分）。FineWeb 用了 18 条规则 + fastText + KenLM 三条流水线。

### Q3：MinHash 怎么用来去重？

> 1. 把文档拆成 n-gram shingles
> 2. 用 k 个哈希函数对 shingle 集合取最小值，得到 k 维签名
> 3. 两文档签名重合度 ≈ Jaccard 相似度
> 4. 用 LSH 把签名分 band 哈希到桶里，同桶的才是候选对
> 5. 只对候选对做精确验证，O(n²) 降到近似 O(n)

### Q4：静态配比和动态配比哪个好？

> 动态配比更好但更复杂。静态配比简单可靠，参考 LLaMA 配比就行。动态配比（如 DeepSeek）根据各类 loss 下降速度自动调整采样权重，理论上更省算力。DoReMi 是一种用小模型自动学配比的方法。

### Q5：Chinchilla 说 D≈20N，为什么 LLaMA 3 敢喂 1875 倍？

> Chinchilla 是 compute-optimal（给定算力求最低 loss）。但实际部署要平衡推理成本。用小模型+超喂数据，推理快且便宜，loss 也只比最优大模型高一点点。对于亿级用户的推理成本来说，这是划算的。

---

## 总结

```
数据工程的三个核心原则:

1. 垃圾进垃圾出 (Garbage In, Garbage Out)
   → 再怎么好的模型架构也救不了烂数据

2. 去重是必须的 (Dedup is Non-Negotiable)
   → 不去重 = 浪费算力 + 记忆效应 + 测试污染

3. 数据价值不等 (Not All Tokens Are Equal)
   → 代码 4.5% 撬动推理能力
   → 合成数据质量可控
   → 配比决定模型"人格"
```

---

*整理时间：2026-07-19*
