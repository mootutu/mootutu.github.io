---
layout: learning-post-layout
title: "3 Attention"
date: 2025-12-26
lang: zh-CN
topic_url: /cn/learning/minimind.html
translate_url: /learning/minimind/part3.html
mathjax: true
---

本章我们将深入理解注意力机制（Attention），并梳理从 **MHA（Multi-Head Attention）** 到 **MQA（Multi-Query Attention）** 再到 **GQA（Grouped Query Attention）** 的演进脉络。这条演进路线的核心动机是：**如何在保持模型质量的同时，降低推理阶段的显存占用和计算开销**。

## 3.1 Attention 的本质：Query、Key、Value

### 直觉理解

Attention 的本质是一种**基于相关性的加权聚合**：给定一个查询（Query），我们希望从一组信息中，根据「相关程度」提取出最有用的内容。

标准的 Scaled Dot-Product Attention 公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $d_k$ 是 Key 向量的维度，除以 $\sqrt{d_k}$ 是为了防止点积值过大导致 softmax 梯度消失。

### Q、K、V 的专业解读

在 Transformer 中，Q、K、V 都是从同一个输入序列（Self-Attention）或不同序列（Cross-Attention）通过线性投影得到的。它们的角色可以从**信息检索**的角度理解：

| 符号 | 名称 | 角色 | 类比 |
|------|------|------|------|
| Q | Query | 「查询向量」：表示当前位置想要获取什么信息 | 搜索引擎中的搜索词 |
| K | Key | 「键向量」：表示每个位置「能提供什么类型的信息」 | 文档的索引/标签 |
| V | Value | 「值向量」：表示每个位置「实际携带的信息内容」 | 文档的正文内容 |

**计算过程的直觉**：
1. **匹配阶段**：用 Query 和所有 Key 做点积（$QK^T$），得到「相似度分数」
2. **归一化阶段**：通过 softmax 将分数转换为「注意力权重」（概率分布）
3. **聚合阶段**：用注意力权重对 Value 进行加权求和，得到最终输出

**为什么要分离 K 和 V？**

一个自然的问题是：为什么不直接用 Query 去匹配 Value，而要引入 Key 作为中间层？

答案在于**解耦「匹配」和「内容」**：
- Key 负责决定「谁应该被关注」（相关性计算）
- Value 负责决定「被关注后提供什么」（信息内容）

这种分离让模型可以学习更灵活的注意力模式。例如，两个位置可能在「语法功能」上相似（Key 相近），但携带的「语义内容」不同（Value 不同）。

### 从线性代数视角

设输入序列为 $X \in \mathbb{R}^{n \times d}$（$n$ 个 token，每个 $d$ 维），则：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$，$W_V \in \mathbb{R}^{d \times d_v}$。

注意力输出为：

$$
\text{head} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \in \mathbb{R}^{n \times d_v}
$$

矩阵 $QK^T \in \mathbb{R}^{n \times n}$ 的第 $(i, j)$ 个元素表示「位置 $i$ 对位置 $j$ 的注意力分数」。

### 深入理解：为什么要除以 $\sqrt{d_k}$？

这个缩放因子看似简单，实际上对训练稳定性至关重要。让我们从方差的角度来分析。

假设 $Q$ 和 $K$ 的每个元素都是独立同分布的随机变量，均值为 0，方差为 1。那么点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的统计特性是：

$$
\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
$$

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k
$$

**问题**：当 $d_k$ 很大时（如 64 或 128），点积的方差也会很大，导致某些值变得极大或极小。

当这些值进入 softmax 时：
- 极大的正值会使 softmax 输出接近 1
- 其他值对应的输出接近 0
- **softmax 进入饱和区，梯度几乎为零**

除以 $\sqrt{d_k}$ 后，缩放后的点积方差变为：

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

这样保证了无论 $d_k$ 多大，点积的分布都保持在合理范围内，softmax 可以正常工作。

### Self-Attention vs Cross-Attention

根据 Q、K、V 的来源不同，Attention 分为两种类型：

| 类型 | Q 来源 | K、V 来源 | 典型应用 |
|------|--------|-----------|----------|
| Self-Attention | 输入序列 X | 输入序列 X | Encoder、Decoder 的自注意力层 |
| Cross-Attention | 目标序列 Y | 源序列 X | Decoder 中对 Encoder 输出的注意力 |

**Self-Attention**：序列内部的 token 互相关注，用于捕捉序列内部的依赖关系。

```
输入: "The cat sat on the mat"
      ↓   ↓   ↓   ↓   ↓   ↓
Q ←── X ──→ K
           ──→ V

每个词都可以关注序列中的其他词
```

**Cross-Attention**：一个序列关注另一个序列，典型应用是机器翻译中 Decoder 关注 Encoder 的输出。

```
Encoder 输出 (源语言): "Le chat est assis"
                              ↓
                         K, V 来自这里

Decoder 输入 (目标语言): "The cat"
                              ↓
                         Q 来自这里

Decoder 在生成每个词时，都会关注整个源语言句子
```

---

## 3.2 Multi-Head Attention（MHA）

### 为什么需要多头？

单头注意力只能学习**一种注意力模式**。但在自然语言中，不同类型的关系需要不同的「关注方式」：
- 语法关系：主语关注谓语
- 指代关系：代词关注其指代的实体
- 语义关系：近义词之间的关联

多头注意力（Multi-Head Attention）让模型可以**并行学习多种注意力模式**。

### 结构

MHA 将输入投影到 $h$ 个不同的子空间，每个子空间独立计算注意力，最后拼接：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

其中每个头：

$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

**关键点**：每个头都有**独立的** $W_Q^i$, $W_K^i$, $W_V^i$ 参数。

用公式表示 MHA 的完整计算过程：

$$
\begin{aligned}
Q_i &= X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i \quad &\text{(每个头独立投影)} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \quad &\text{(计算注意力)} \\
\text{Output} &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O \quad &\text{(拼接并投影)}
\end{aligned}
$$

其中 $i = 1, 2, \ldots, h$。

**参数统计**：
- 每个头：$W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d \times d_k}$
- 共有 $h$ 组独立的 Q、K、V 投影
- KV Cache 需要存储：$h$ 组 K 和 $h$ 组 V

### 实现细节：Split Heads vs Separate Projections

在实际实现中，MHA 有两种等价的实现方式：

**方式一：分离投影（Separate Projections）**

为每个头定义独立的投影矩阵 $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d \times d_k}$：

```python
# 概念上的实现（效率较低）
heads = []
for i in range(n_heads):
    Q_i = X @ W_Q[i]  # [n, d_k]
    K_i = X @ W_K[i]  # [n, d_k]
    V_i = X @ W_V[i]  # [n, d_v]
    head_i = attention(Q_i, K_i, V_i)
    heads.append(head_i)
output = concat(heads) @ W_O
```

**方式二：合并投影 + 分割（实际使用）**

使用一个大的投影矩阵，然后将结果分割成多个头：

```python
# 实际高效实现
W_Q = [d_model, n_heads * d_k]  # 合并所有头的 Q 投影
W_K = [d_model, n_heads * d_k]
W_V = [d_model, n_heads * d_v]

Q = X @ W_Q  # [n, n_heads * d_k]
K = X @ W_K
V = X @ W_V

# 重塑为多头形式
Q = Q.reshape(n, n_heads, d_k).transpose(1, 2)  # [n_heads, n, d_k]
K = K.reshape(n, n_heads, d_k).transpose(1, 2)
V = V.reshape(n, n_heads, d_v).transpose(1, 2)

# 批量计算所有头的注意力
output = batched_attention(Q, K, V)  # [n_heads, n, d_v]
output = output.transpose(1, 2).reshape(n, n_heads * d_v)  # [n, d_model]
output = output @ W_O
```

这两种方式在数学上完全等价，但方式二可以利用矩阵运算的并行性，效率更高。

### head_dim 的设计考量

在实践中，通常有：

$$
d_k = d_v = \text{head_dim} = \frac{d_{\text{model}}}{n_{\text{heads}}}
$$

例如，对于 $d_{\text{model}} = 768$，$n_{\text{heads}} = 12$：

$$
\text{head_dim} = \frac{768}{12} = 64
$$

**为什么这样设计？**

1. **保持计算量不变**：总参数量 $n_{\text{heads}} \times d_{\text{model}} \times \text{head_dim} = d_{\text{model}}^2$，与单头相同
2. **允许并行计算**：所有头可以同时计算
3. **权衡表达能力和效率**：更多的头意味着更多的注意力模式，但每个头的维度更低

### 输出投影 $W_O$ 的作用

将所有头的输出拼接后，还要通过一个线性变换 $W_O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

**为什么需要 $W_O$？**

1. **维度匹配**：确保输出维度与输入一致，便于残差连接
2. **信息融合**：让不同头的信息可以相互交流和组合
3. **增加表达能力**：额外的可学习参数

没有 $W_O$，各个头的输出只是简单拼接，无法相互影响。

### 不同头学到的模式

研究表明，不同的注意力头会自动学习不同类型的语言模式：

| 头的类型 | 关注模式 | 示例 |
|----------|----------|------|
| 位置头 | 关注固定的相对位置 | 总是关注前一个词或后一个词 |
| 语法头 | 关注语法相关的词 | 动词关注其主语/宾语 |
| 指代头 | 关注指代关系 | 代词"它"关注其指代的名词 |
| 稀有词头 | 关注罕见或重要的词 | 专注于专有名词 |
| 分隔符头 | 关注句子边界 | 关注句号、逗号等 |

这种「分工」是模型自动学习的结果，不需要人工设计。

### 优点与代价

**优点**：表达能力强，不同头可以捕捉不同类型的依赖关系。

**代价**：参数量和计算量随头数增加。更重要的是，在**推理阶段**会产生严重的显存瓶颈——这就是 KV Cache 问题。

---

## 3.3 推理瓶颈：KV Cache

### 自回归生成的特点

大语言模型在生成文本时采用**自回归**方式：每次只生成一个 token，然后将其加入输入序列，继续生成下一个。

```
Step 1: "The"           → 生成 "cat"
Step 2: "The cat"       → 生成 "sat"
Step 3: "The cat sat"   → 生成 "on"
...
```

### 问题：重复计算

在第 $t$ 步生成时，需要计算当前 token 对**所有之前 token** 的注意力。如果每次都重新计算所有位置的 K 和 V，计算量是 $O(t)$，总体复杂度变成 $O(n^2)$。

### 解决方案：KV Cache

观察到：之前位置的 K 和 V **不会改变**（因为它们只依赖于之前的输入）。所以我们可以**缓存**已计算的 K、V：

```
Step 1: 计算 K_1, V_1，缓存
Step 2: 计算 K_2, V_2，缓存；复用 K_1, V_1
Step 3: 计算 K_3, V_3，缓存；复用 K_1, V_1, K_2, V_2
...
```

### 新问题：显存占用

KV Cache 的大小为：

$$
\text{KV Cache Size} = 2 \times \text{层数} \times \text{头数} \times \text{序列长度} \times \text{head_dim}
$$

对于一个典型的 LLM（如 LLaMA-7B，32 层，32 头，head_dim=128，序列长度 4096）：

$$
2 \times 32 \times 32 \times 4096 \times 128 \times 2\text{ bytes} \approx 2\text{ GB}
$$

这还只是**单个请求**的缓存。在高并发场景下，KV Cache 成为显存瓶颈。

**这就是优化注意力机制的核心动机：减少 KV Cache 的大小。**

### 深入理解：Prefill 阶段 vs Decode 阶段

LLM 推理可以分为两个截然不同的阶段：

**Prefill 阶段（首次计算）**

处理用户输入的 prompt，一次性计算所有 token 的 K、V 并缓存。

```
输入 prompt: "What is the capital of France?"
                    ↓
        一次性计算所有 token 的 K, V
                    ↓
        缓存: [K_1, K_2, ..., K_n], [V_1, V_2, ..., V_n]
                    ↓
        生成第一个输出 token
```

特点：
- **计算密集型（Compute-bound）**：需要处理整个 prompt
- 可以高度并行化
- 主要瓶颈是计算速度

**Decode 阶段（逐 token 生成）**

每次只生成一个 token，计算其 K、V，追加到缓存。

```
已缓存: [K_1, ..., K_n]
              ↓
生成 token_{n+1}，计算 K_{n+1}, V_{n+1}
              ↓
追加到缓存: [K_1, ..., K_n, K_{n+1}]
              ↓
使用完整缓存计算注意力
              ↓
生成下一个 token
```

特点：
- **显存密集型（Memory-bound）**：每步只计算 1 个 token，但要访问整个 KV Cache
- 难以并行化（自回归依赖）
- 主要瓶颈是显存带宽

这解释了为什么 KV Cache 优化如此重要：Decode 阶段的速度主要受限于读取 KV Cache 的速度，减小 Cache 大小可以直接提升推理速度。

### 为什么只缓存 K、V，不缓存 Q？

这是一个很好的问题。让我们分析注意力计算的结构：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

在自回归生成中，当生成第 $t$ 个 token 时：

| 变量 | 维度 | 来源 | 是否需要缓存？ |
|------|------|------|----------------|
| $Q_t$ | $[1, d_k]$ | 仅当前 token | **不需要** |
| $K_{1:t}$ | $[t, d_k]$ | 所有已生成 token | **需要** |
| $V_{1:t}$ | $[t, d_v]$ | 所有已生成 token | **需要** |

**关键观察**：

1. **Q 只需要当前 token**：在计算第 $t$ 步的注意力时，我们只需要 $Q_t$（当前 token 的 Query），而不需要之前的 $Q_1, \ldots, Q_{t-1}$

2. **K、V 需要所有历史**：当前 token 要关注所有之前的 token，所以需要 $K_{1:t}$ 和 $V_{1:t}$

用代码来理解：

```python
# 第 t 步生成
q_t = compute_query(x_t)      # 只计算当前 token 的 Q
k_t = compute_key(x_t)        # 计算当前 token 的 K
v_t = compute_value(x_t)      # 计算当前 token 的 V

# 更新缓存
k_cache.append(k_t)           # 缓存 K
v_cache.append(v_t)           # 缓存 V

# 计算注意力
# Q: [1, d_k]，只需要当前 token
# K: [t, d_k]，需要所有历史（从缓存读取）
# V: [t, d_v]，需要所有历史（从缓存读取）
attn = softmax(q_t @ k_cache.T / sqrt(d_k)) @ v_cache
```

这就是为什么 KV Cache 只存 K 和 V，不存 Q。

### KV Cache 的内存布局

在实际实现中，KV Cache 通常按以下方式组织：

```
KV Cache 结构:
┌─────────────────────────────────────────────────┐
│  Layer 0                                        │
│  ├── K: [batch, n_heads, max_seq_len, head_dim]│
│  └── V: [batch, n_heads, max_seq_len, head_dim]│
├─────────────────────────────────────────────────┤
│  Layer 1                                        │
│  ├── K: [batch, n_heads, max_seq_len, head_dim]│
│  └── V: [batch, n_heads, max_seq_len, head_dim]│
├─────────────────────────────────────────────────┤
│  ...                                            │
├─────────────────────────────────────────────────┤
│  Layer N-1                                      │
│  ├── K: [batch, n_heads, max_seq_len, head_dim]│
│  └── V: [batch, n_heads, max_seq_len, head_dim]│
└─────────────────────────────────────────────────┘
```

**预分配策略**：通常会预先分配 `max_seq_len` 大小的空间，避免动态扩展带来的内存碎片和复制开销。

**显存占用公式**（详细版）：

$$
\text{Memory} = 2 \times L \times B \times H \times S \times D \times \text{bytes_per_element}
$$

其中：
- $L$ = 层数
- $B$ = batch size
- $H$ = 头数（这里是 MHA/MQA/GQA 优化的关键！）
- $S$ = 序列长度
- $D$ = head_dim
- bytes_per_element = 2 (fp16) 或 4 (fp32)

---

## 3.4 Multi-Query Attention（MQA）

### 核心思想

MQA（Shazeer, 2019）提出了一个激进的方案：**所有头共享同一组 K 和 V**，只有 Q 保持多头。

用公式表示：

$$
\begin{aligned}
Q_i &= X W_Q^i \quad &\text{(每个头有独立的 Q 投影，} i = 1, \ldots, h \text{)} \\
K &= X W_K \quad &\text{(所有头共享同一个 K)} \\
V &= X W_V \quad &\text{(所有头共享同一个 V)} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V \quad &\text{(每个 Q}_i \text{ 都与同一个 K、V 计算注意力)}
\end{aligned}
$$

**参数统计**：
- Q 投影：$h$ 组独立的 $W_Q^i$
- K/V 投影：仅 1 组共享的 $W_K$, $W_V$
- KV Cache 需要存储：**1 组 K 和 1 组 V**（减少到 MHA 的 $1/h$）

### 效果

**优点**：KV Cache 减少到原来的 $1/h$。对于 32 头的模型，显存占用直接降低到 1/32。

**缺点**：所有头被迫使用相同的 K、V 进行注意力计算，**表达能力受限**，可能导致模型质量下降。

---

## 3.5 Grouped Query Attention（GQA）

### 核心思想

GQA（Ainslie et al., 2023）是 MHA 和 MQA 的**折中方案**：将 $h$ 个 Query 头分成 $g$ 组，每组共享一套 K、V。

用公式表示（假设每组有 $h/g$ 个 Q 头）：

$$
\begin{aligned}
Q_i &= X W_Q^i \quad &\text{(每个头有独立的 Q 投影，} i = 1, \ldots, h \text{)} \\
K_j &= X W_K^j \quad &\text{(第 } j \text{ 组的共享 K，} j = 1, \ldots, g \text{)} \\
V_j &= X W_V^j \quad &\text{(第 } j \text{ 组的共享 V，} j = 1, \ldots, g \text{)} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K_{g(i)}^T}{\sqrt{d_k}}\right) V_{g(i)} \quad &\text{(} Q_i \text{ 使用其所属组的 } K_{g(i)}, V_{g(i)} \text{)}
\end{aligned}
$$

其中 $g(i) = \lceil i \cdot g / h \rceil$ 表示第 $i$ 个 Q 头所属的组。

**参数统计**：
- Q 投影：$h$ 组独立的 $W_Q^i$
- K/V 投影：$g$ 组共享的 $W_K^j$, $W_V^j$
- KV Cache 需要存储：**$g$ 组 K 和 $g$ 组 V**（减少到 MHA 的 $g/h$）

### 特殊情况

- 当 $g = h$ 时，每个 Q 头都有自己的 KV → **退化为 MHA**
- 当 $g = 1$ 时，所有 Q 头共享一组 KV → **退化为 MQA**

### 优点

GQA 在**推理效率**和**模型质量**之间取得了良好的平衡：
- KV Cache 减少到 $g/h$（如 $g=4, h=32$ 时，减少到 1/8）
- 质量损失很小，接近原始 MHA

---

## 3.6 对比总结

| 方法 | Q 头数 | K/V 头数 | KV Cache 大小 | 模型质量 | 代表模型 |
|------|--------|----------|---------------|----------|----------|
| MHA  | $h$    | $h$      | 1×            | 最高     | GPT-3, BERT |
| MQA  | $h$    | 1        | $1/h$         | 可能下降 | PaLM |
| GQA  | $h$    | $g$      | $g/h$         | 接近 MHA | LLaMA-2, Mistral |

**一句话总结**：

> MHA 追求表达能力，MQA 追求极致效率，GQA 在两者之间找到了实用的平衡点——通过分组共享 KV，以较小的质量代价换取显著的推理加速。

---

## 补充知识：ResNet（残差网络）

在深度学习发展早期，研究者们发现一个反直觉的现象：**网络越深，效果反而可能越差**。

按理说，一个 56 层的网络应该比 20 层的网络更强大——最坏的情况下，多出来的 36 层什么都不做（学成恒等映射），也应该和 20 层一样好。但实验结果却显示，56 层网络的**训练误差**反而更高。

注意，这里说的是「训练误差」而非「测试误差」，所以问题不是过拟合，而是**优化困难**——网络根本就没学好。

这就引出了两个核心问题：
1. **函数拟合问题**：让网络学习「什么都不做」（恒等映射）其实很难
2. **梯度传播问题**：网络太深时，梯度在反向传播中逐渐消失

ResNet 的核心洞察是：与其让网络直接学习目标函数，不如让它学习「目标函数与输入的差值」——这就是「残差」的含义。

---

### 3.1 残差结构为什么在函数拟合上更容易？

#### 传统结构 vs 残差结构

假设我们希望某一层学习的目标函数是 $H(x)$：

| 结构 | 网络需要学习的内容 |
|------|-------------------|
| 传统结构 | 直接学习 $H(x)$ |
| 残差结构 | 学习 $F(x) = H(x) - x$，最后输出 $H(x) = F(x) + x$ |

#### 一个极端但重要的例子

假设某个深层的「完美答案」恰好是 $H(x) = x$（恒等映射）。

**传统结构的困境**：网络需要精心调整所有权重，使得输出恰好等于输入。这要求每一层的权重矩阵近似于单位矩阵，偏置接近零——对优化器来说，这是一个需要精确到达的「目标点」。

**残差结构的优势**：由于 $H(x) = F(x) + x$，要实现 $H(x) = x$，只需要 $F(x) = 0$。而「让输出趋近于零」对神经网络来说非常简单——只要把权重都推向零即可。

#### 类比理解

想象你要画一幅画：

- **传统方式**：从白纸开始，画出完整的作品
- **残差方式**：先把参考图打印在纸上，你只需要画出「需要修改的部分」

如果参考图已经很接近目标，残差方式只需要少量修改；如果差距很大，残差方式也不会更差。这就是残差学习的「下限保障」。

---

### 3.2 残差结构在梯度传播上的意义

现在来看核心公式。

#### 反向传播推导

设损失函数为 $L$，残差块的输出为 $H(x) = F(x) + x$。

根据链式法则：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial H}{\partial x}
$$

由于 $H(x) = F(x) + x$，对 $x$ 求导：

$$
\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + \frac{\partial x}{\partial x} = \frac{\partial F}{\partial x} + 1
$$

因此：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)
$$

#### 这个 "+1" 为什么重要？

在传统网络中，梯度完全依赖于 $\frac{\partial F}{\partial x}$。如果 $F$ 的梯度很小（比如 0.1），经过 50 层后：

$$
0.1^{50} \approx 10^{-50} \quad \text{（几乎为零）}
$$

这就是**梯度消失**：前面的层几乎收不到任何梯度信号，无法学习。

而在残差结构中，即使 $\frac{\partial F}{\partial x}$ 很小，梯度表达式中仍有 "+1" 这一项。展开多层来看：

$$
\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=0}^{n-1} \left( \frac{\partial F_i}{\partial x_i} + 1 \right)
$$

展开这个乘积，你会发现其中包含一条「直通路径」：

$$
\frac{\partial L}{\partial x_n} \cdot 1 \cdot 1 \cdot 1 \cdots = \frac{\partial L}{\partial x_n}
$$

这意味着**梯度可以无损地直接流到最前面的层**，这就是所谓的「梯度高速公路」。

#### 类比理解

想象一个传话游戏：
- **传统网络**：每个人只能把消息传给下一个人，传 50 次后消息严重失真
- **残差网络**：每个人除了传话，还有一条直通电话线到源头，确保关键信息不丢失

---

### 3.3 一句话总结

> **ResNet 通过「学习残差」而非「学习目标」，既降低了优化难度（恒等映射只需 F=0），又为梯度提供了直通路径（+1 项），从而使训练极深网络成为可能。**

---

## 3.7 代码实现

下面我们提供 GQA（分组查询注意力）的 PyTorch 实现代码。这段代码展示了如何实现关键组件：用于扩展 KV 头的 `repeat_kv` 函数，以及通过统一接口支持 MHA、MQA 和 GQA 的完整 Attention 类。

### 3.7.1 `repeat_kv` 函数

GQA 实现的核心在于 `repeat_kv` 函数。该函数负责扩展 K 和 V 张量，使每个 Query 头都有对应的 K 和 V 来计算注意力。

**解决的问题**

在 GQA 中，我们有：
- $h$ 个 Query 头（例如 32 个）
- $g$ 个 Key/Value 头（例如 4 个）

每个 KV 头需要服务 $h/g$ 个 Query 头（例如 8 个 Query 头共享 1 个 KV 头）。`repeat_kv` 函数通过将每个 KV 头重复 $n_rep = h/g$ 次来实现这一点。

**实现细节**

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 KV 头以匹配 Query 头的数量。

    参数:
        x: 输入张量，形状为 [batch, num_kv_heads, seq_len, head_dim]
        n_rep: 每个 KV 头重复的次数

    返回:
        形状为 [batch, num_kv_heads * n_rep, seq_len, head_dim] 的张量
    """
    if n_rep == 1:
        return x  # MHA 情况：无需重复

    batch, num_kv_heads, seq_len, head_dim = x.shape

    # 插入新维度并扩展
    # [batch, num_kv_heads, seq_len, head_dim]
    # -> [batch, num_kv_heads, 1, seq_len, head_dim]
    # -> [batch, num_kv_heads, n_rep, seq_len, head_dim]
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)

    # 重塑以合并重复的维度
    # -> [batch, num_kv_heads * n_rep, seq_len, head_dim]
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
```

**理解形状变换**

让我们用一个具体例子来追踪，其中 `batch=1`、`num_kv_heads=4`、`seq_len=10`、`head_dim=64`、`n_rep=8`：

1. 输入形状：`[1, 4, 10, 64]`
2. `unsqueeze` 后：`[1, 4, 1, 10, 64]`
3. `expand` 后：`[1, 4, 8, 10, 64]`
4. `reshape` 后：`[1, 32, 10, 64]`

现在我们有 32 个 KV「头」，可以与 32 个 Query 头配对。

### 3.7.2 Attention 类初始化

Attention 类使用 `num_key_value_heads` 来控制它是作为 MHA、MQA 还是 GQA 运行：

**关键配置参数**

```python
def __init__(self, args: ModelConfig):
    super().__init__()

    # Query 头的数量（始终是完整数量）
    self.n_heads = args.num_attention_heads

    # KV 头的数量（决定 MHA/MQA/GQA）
    # - num_kv_heads == n_heads: MHA
    # - num_kv_heads == 1: MQA
    # - 1 < num_kv_heads < n_heads: GQA
    self.num_kv_heads = args.num_key_value_heads or args.num_attention_heads

    # KV 头的重复因子
    self.n_rep = self.n_heads // self.num_kv_heads

    # 每个头的维度
    self.head_dim = args.hidden_size // args.num_attention_heads
```

**投影层维度**

与标准 MHA 的关键区别在于投影层的大小：

```python
# Q 投影：始终是完整大小 (n_heads * head_dim)
self.wq = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)

# K 和 V 投影：缩小的大小 (num_kv_heads * head_dim)
self.wk = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
self.wv = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

# 输出投影：回到 hidden_size
self.wo = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
```

这就是显存节省的来源 —— K 和 V 投影按 `n_heads / num_kv_heads` 的因子缩小。

### 3.7.3 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # 使用 4 个 KV 头的 GQA
    max_seq_len: int = 2048


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    为 GQA 重复 KV 头以匹配 Query 头的数量。

    参数:
        x: 输入张量，形状为 [batch, num_kv_heads, seq_len, head_dim]
        n_rep: 每个 KV 头重复的次数

    返回:
        形状为 [batch, num_kv_heads * n_rep, seq_len, head_dim] 的张量
    """
    if n_rep == 1:
        return x

    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """
    支持分组查询注意力（GQA）的多头注意力。

    通过调整 num_key_value_heads：
    - num_key_value_heads == num_attention_heads: 标准 MHA
    - num_key_value_heads == 1: 多查询注意力（MQA）
    - 1 < num_key_value_heads < num_attention_heads: 分组查询注意力（GQA）
    """

    def __init__(self, args: ModelConfig):
        super().__init__()

        # Query 头的数量
        self.n_heads = args.num_attention_heads

        # KV 头的数量（用于 GQA）
        self.num_kv_heads = args.num_key_value_heads or args.num_attention_heads

        # 确保 n_heads 能被 num_kv_heads 整除
        assert self.n_heads % self.num_kv_heads == 0, \
            f"num_attention_heads ({self.n_heads}) 必须能被 num_key_value_heads ({self.num_kv_heads}) 整除"

        # 每个 KV 头重复的次数
        self.n_rep = self.n_heads // self.num_kv_heads

        # 每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 线性投影
        # Q：完整大小，K/V：GQA 的缩小大小
        self.wq = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        带有可选 KV 缓存的注意力前向传播。

        参数:
            x: 输入张量，形状为 [batch, seq_len, hidden_size]
            freqs_cos, freqs_sin: RoPE 频率张量
            past_kv: 来自之前步骤的可选缓存 (K, V)
            use_cache: 是否返回更新的 KV 缓存

        返回:
            output: 注意力输出，形状为 [batch, seq_len, hidden_size]
            present_kv: 更新的 KV 缓存（如果 use_cache=True）
        """
        batch, seq_len, _ = x.shape

        # 线性投影
        q = self.wq(x)  # [batch, seq_len, n_heads * head_dim]
        k = self.wk(x)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.wv(x)  # [batch, seq_len, num_kv_heads * head_dim]

        # 重塑为多头格式
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 应用 RoPE（旋转位置编码）
        q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        # 转置用于注意力计算：[batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 处理自回归生成的 KV 缓存
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # 重复 KV 头以匹配 Q 头（GQA 的核心）
        k = repeat_kv(k, self.n_rep)  # [batch, n_heads, kv_seq_len, head_dim]
        v = repeat_kv(v, self.n_rep)  # [batch, n_heads, kv_seq_len, head_dim]

        # 缩放点积注意力
        # 使用 F.scaled_dot_product_attention 以提高效率（可用时使用 Flash Attention）
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,  # 自回归模型的因果掩码
        )

        # 重塑回：[batch, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # 输出投影
        output = self.wo(output)

        return output, present_kv
```

**使用示例**

```python
# GQA 配置（12 个 Q 头，4 个 KV 头）
config = ModelConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=4,  # 每个 KV 头服务 3 个 Q 头
)

# 创建注意力模块
attn = Attention(config)

# 输入张量
x = torch.randn(2, 128, 768)  # [batch=2, seq_len=128, hidden=768]

# RoPE 频率（简化）
freqs_cos = torch.ones(128, 64)
freqs_sin = torch.zeros(128, 64)

# 前向传播
output, kv_cache = attn(x, freqs_cos, freqs_sin, use_cache=True)
print(f"输出形状: {output.shape}")  # [2, 128, 768]
print(f"KV 缓存 K 形状: {kv_cache[0].shape}")  # [2, 4, 128, 64] - 只有 4 个 KV 头！
```

**显存对比**

对于上述配置，序列长度为 128 时：

| 方法 | KV 缓存形状 | 显存占用（每层，fp16） |
|------|------------|----------------------|
| MHA | `[2, 12, 128, 64]` | 384 KB |
| GQA (g=4) | `[2, 4, 128, 64]` | 128 KB |
| MQA | `[2, 1, 128, 64]` | 32 KB |

使用 4 个 KV 头的 GQA 将显存减少到 MHA 的 1/3，同时保持了大部分模型质量。
