---
layout: learning-post-layout
title: "2 位置编码 RoPE&YaRN"
date: 2025-12-26
lang: zh-CN
topic_url: /cn/learning/minimind.html
translate_url: /learning/minimind/part2.html
mathjax: true
---


本节我们来聊一聊大语言模型（LLM）中常用的几种位置编码方法，重点放在 **旋转位置编码（RoPE）** 以及它在长上下文场景下的一个重要扩展 —— **YaRN**。

我们会先从直觉出发，看看 RoPE 是如何通过在复平面中对向量进行旋转，把“相对位置”自然地融入到注意力计算中的；然后再进一步讨论，当序列长度超过模型训练时见过的范围时，原始 RoPE 会遇到哪些问题，以及 YaRN 是如何通过对不同频率进行动态调整，来让模型更稳定地处理超长上下文的。

## 2.1 RoPE (Rotary Positional Embedding)

假设有两个二维向量 $\mathbf{q}$ 与 $\mathbf{k}$，分别表示  **query**  向量与  **key**  向量：

$$
\mathbf{q} = [q_1, q_2], \qquad \mathbf{k} = [k_1, k_2]
$$

为了将向量旋转 $\theta$ 角度，我们引入二维旋转矩阵 $\mathbf{R}(\theta)$，定义为：

$$
\mathbf{R}(\theta) =

\begin{bmatrix}

\cos\theta & -\sin\theta \

\sin\theta & \cos\theta

\end{bmatrix}
$$

其中，$\cos\theta$ 与 $\sin\theta$ 分别是角度 $\theta$ 的余弦与正弦值。

将旋转矩阵与原向量相乘，即可得到旋转后的向量：

$$
\mathbf{q}' = \mathbf{R}(\theta)\mathbf{q}, \qquad

\mathbf{k}' = \mathbf{R}(\theta)\mathbf{k}
$$

其中，$\mathbf{q}'$ 与 $\mathbf{k}'$ 分别表示旋转 $\theta$ 角度后的  **query**  向量与  **key**  向量。

我们对 **query** 向量 $\mathbf{q}$ 施加一次旋转，旋转角度为 $m\theta$；对 **key** 向量 $\mathbf{k}$ 也施加旋转，旋转角度为 $n\theta$。其中，$m$ 与 $n$ 分别是 **query** 与 **key** 的位置索引：$m$ 表示该 **query** 向量 $\mathbf{q}$ 来自序列中的第 $m$ 个位置（第 $m$ 个 token），$n$ 表示该 **key** 向量 $\mathbf{k}$ 来自序列中的第 $n$ 个位置（第 $n$ 个 token）。旋转后的 **query** 与 **key** 向量分别为：

$$
\mathbf{q_m} = [q_1\cos (m\theta) - q_2\sin (m\theta), q_1\sin (m\theta) + q_2\cos (m\theta)] \\

\mathbf{k_n} = [k_1\cos (n\theta) - k_2\sin (n\theta), k_1\sin (n\theta) + k_2\cos (n\theta)]
$$

计算两者的点积，并将其进行逐步展开：

$$
\begin{aligned}

\mathbf{q_m}\cdot \mathbf{k_n}

&= (q_1\cos (m\theta) - q_2\sin (m\theta))(k_1\cos (n\theta) - k_2\sin (n\theta)) \\

&\quad + (q_1\sin (m\theta) + q_2\cos (m\theta))(k_1\sin (n\theta) + k_2\cos (n\theta)) \\

&= q_1k_1\cos (m\theta) \cos (n\theta) - q_1k_2\cos (m\theta) \sin (n\theta) - q_2k_1\sin (m\theta) \cos (n\theta) + q_2k_2\sin (m\theta) \sin (n\theta) \\

&\quad+ q_1k_1\sin (m\theta) \sin (n\theta) + q_1k_2\sin (m\theta) \cos (n\theta) + q_2k_1\cos (m\theta) \sin (n\theta) + q_2k_2\cos (m\theta) \cos (n\theta) \\

&=q_1k_1(\cos (m\theta) \cos (n\theta) + \sin (m\theta) \sin (n\theta)) + q_2k_2(\sin (m\theta) \sin (n\theta) + \cos (m\theta) \cos (n\theta)) \\

&\quad + q_1k_2(\sin (m\theta) \cos (n\theta) - \cos (m\theta) \sin (n\theta)) + q_2k_1(\cos (m\theta) \sin (n\theta) - \sin (m\theta) \cos (n\theta)) \\

&=(q_1k_1 + q_2k_2)\cos((m-n)\theta) + (q_1k_2 - q_2k_1)\sin((m-n)\theta)

\end{aligned}
$$

由于 $q_1$, $k_1$, $q_2$, $k_2$ 都是我们给定的值，且 $\theta$ 也是我们给定的值，也就是说整个式子中只和 $(m-n)$ 相关，也就是说最后的位置信息只由 $(m-n)$ 决定。  

RoPE 之后，用旋转后的 $q'_m, k'_n$ 来计算注意力分数：

$$\text{score}(m, n) = \frac{(q'_m)^\top (k'_n)}{\sqrt{d}}$$

因为 $q$ 和 $k$ 都带着各自位置的旋转角（$m\theta$、$n\theta$），因此点积会自然编码 相对位置信息（与 $m-n$ 强相关）。

## 2.2 RoPE 的实际应用

RoPE 不是直接旋转词向量 embedding，而是在注意力里对每个位置的 Query (Q)、Key (K) 做旋转（通常 Value (V) 不旋转）。

### 2.2.2 从隐藏状态到 Q/K/V

在某一层 Transformer 中，每个位置都会有一个隐藏状态向量：

$$
\mathbf{h}_m = [h_{m1}, h_{m2}, \ldots, h_{d}] \in \mathbb{R}^d
$$

通过线性投影得到注意力所需的 **Query**/**Key**/**Value**（以单头举例）：

$$
q_m = h_m W_q,\quad k_m = h_m W_k,\quad v_m = h_m W_v
$$

其中：

- $q_m, k_m, v_m \in \mathbb{R}^{d}$

- $d$ 是该 head 的维度（head_dim）

### 2.2.3 RoPE 对 Q/K 的“旋转”是怎么做的

将 head_dim 按 2 维一组拆开，假设 $d = 8$，则：

$$
q_m = [q_{m,0}, q_{m,1}, q_{m,2}, q_{m,3}, q_{m,4}, q_{m,5}, q_{m,6}, q_{m,7}]
$$

两两成对分组：

- 第 0 组：$(q_{m,0}, q_{m,1})$

- 第 1 组：$(q_{m,2}, q_{m,3})$

- 第 2 组：$(q_{m,4}, q_{m,5})$

- 第 3 组：$(q_{m,6}, q_{m,7})$

对 $k_m$ 同理。

每一组有自己的频率 $\theta_i$，角度随位置线性增长。对第 $i$ 组（一个 2D pair），旋转角度为：

$$\text{angle}_{m,i} = m \cdot \theta_i$$

其中：

$m$ 是 token 的位置索引（第 $m$ 个 token）

$\theta_i$ 是与维度组 $i$ 对应的频率（通常基于 $10000^{-2i/d}$ 生成）

用旋转矩阵对每个 pair 旋转，对某一组 2D 向量 $(x, y)$，旋转后为 $(x', y')$：

$$
\begin{bmatrix}

x' \\ y'

\end{bmatrix}=

\begin{bmatrix}

\cos(m\theta_i) & -\sin(m\theta_i) \\

\sin(m\theta_i) & \cos(m\theta_i)

\end{bmatrix}

\begin{bmatrix}

x \\ y

\end{bmatrix}
$$

### 2.2.4 一个例子（只演示某个 pair）

以一句话为例：

\> “我(0) 喜欢(1) 吃(2) 苹果(3)”

取 token “吃”，其位置为 $m=2$。

假设在某个 head 的第 0 个 pair 上（仅示意）：

- 原始 $q$ 的 pair：$(x, y) = (0.5, -1.0)$

- 原始 $k$ 的 pair：$(x, y) = (1.2, 0.3)$

- 该 pair 的频率：$\theta_0 = 0.1$

那么旋转角度为：

$$m\theta_0 = 2 \times 0.1 = 0.2 \text{ rad}$$

旋转后（四舍五入）：

- $q' \approx (0.6887, -0.8807)$

- $k' \approx (1.1165, 0.5324)$

**注意：这只是 Q/K 的一个 2D 片段；实际会对所有 pair、所有 head 都做同样操作，只是每个 pair 的 $\theta_i$ 不同。**

## 2.2 YaRN (Yet another RoPE extensioN method)

### 2.2.1 为什么要引入 YaRN 呢？

RoPE 中每一对维度都会对应一个固定的旋转频率，其定义为：

$$
\text{freqs}_i = \frac{1}{\text{rope_base}^{\frac{2i}{\text{dim}}}}
$$

从这个公式可以看出，随着维度索引 $i$ 的增大，旋转频率会逐渐减小：低维部分对应较高的旋转频率，高维部分对应较低的旋转频率。这里需要注意的是，RoPE 中的“维度”并不是随意的索引，而是对应着一组从高频到低频的多尺度旋转基，用于同时编码短距离与长距离的相对位置信息。

在序列长度不太长的情况（比如 4096）下，这样的频率分布通常是没有问题的。但当我们把上下文长度拉得很长（比如超过模型训练时见过的最大长度 4096， 如5000）时，就会出现问题。

直观来说，对于高频维度，当相对位置变得非常大时，不同距离对应的旋转角度可能会“转到同一个位置”上。也就是说，模型在远处计算得到的位置信息，在经过 RoPE 旋转之后，可能会和某些近处位置产生非常相似的表示。

一旦出现这种情况，模型在注意力计算中就很难区分「这是一个很远的 token」还是「这是一个比较近的 token」，远近关系被混淆，最终会影响模型在长上下文场景下的训练稳定性和效果。这也是为什么原始 RoPE 在长序列外推时容易出现性能下降的问题。

### 2.2.3 YaRN

YaRN 的核心思想并不是简单地整体缩放 RoPE 的频率，而是根据不同维度所对应的“波长”，对高频和低频部分采用不同的缩放策略。

在原始 RoPE 中，第 $i$ 个维度对应的旋转频率为 $\text{freqs}_i$。YaRN 的目标是对这些频率引入一个位置相关的缩放因子，使其在长上下文场景下更加稳定。

实际计算过程可以分为以下几步。



**第一步：确定需要调整的维度范围**

我们可以通过波长来判断某个维度是否已经“超出了模型原本能够理解的上下文范围”。由于 RoPE 中每个维度对应的旋转本质上是一个周期函数，不同维度对相对位置的“可区分范围”是不同的。如果某个维度在训练长度范围内已经完成了一次或多次完整旋转，那么该维度在更长序列下将不可避免地发生相位重复，从而导致不同距离映射到相同表示。

直观来看，在 RoPE 中，每一个维度实际上都在用一个**周期性的旋转**来编码位置信息。对于第 $i$ 个维度，位置 $m$ 对应的旋转角度可以写成：
$$
\theta_i(m) = m \cdot \text{freqs}_i
$$
由于旋转角度是以 $2\pi$ 为周期的，当相对位置变化使得旋转角度增加 $2\pi$ 时，该维度的表示会回到同一个相位。这意味着，一旦相对距离超过某个阈值，该维度将无法再区分更远的位置差异。

因此，我们可以用“转一整圈所对应的距离”来衡量该维度能够区分的最大相对位置范围，这个量就是波长。波长可以近似表示为：
$$
\lambda_i = \frac{2\pi}{\text{freqs}[i]}
$$

当某个维度对应的波长小于或接近模型在训练阶段见过的最大上下文长度（记为 $\text{original_max}$）时，说明在训练范围内，该维度已经开始出现相位重复；而在更长的序列中，这种重复会进一步加剧，从而导致远近位置信息发生混叠。

基于这一观察，YaRN 通过比较波长与训练时的最大上下文长度，来判断哪些维度在长序列场景下是“不安全的”。

因此，我们定义一个分界维度：

$$
\text{corr_dim} = \min \left\{ i \mid \frac{2\pi}{\text{freqs}[i]} > \text{original_max} \right\}
$$

这里的 $\text{corr_dim}$（correction dimension）表示第一个在训练上下文长度范围内仍未发生相位重复的维度索引，用于作为高频（不稳定）与低频（相对稳定）维度之间的分界点。

在此之前的维度被视为高频（短波长）部分，容易在长序列下产生混叠；而之后的维度则属于低频（长波长）部分，对长距离信息更加稳定。由于高频维度对应的波长较短，在训练阶段就已经开始出现相位重复，因此在长上下文场景下最容易引发位置混叠；相比之下，低频维度对长距离信息更加稳定。YaRN 的设计正是基于这一观察，对高频维度施加更强的校正，而尽量保持低频维度的原有结构。

**第二步：为不同维度计算插值权重**

为了在高频和低频之间平滑过渡，YaRN 为每个维度定义了一个归一化的权重：

$$
\text{power}_i = \frac{i}{\frac{\text{dim}}{2} - 1}
$$

该权重用于控制不同维度在后续缩放过程中的插值比例。



**第三步：计算每个维度对应的缩放系数**

YaRN 在高频和低频区域分别使用不同的缩放策略。  
首先，为每个维度定义一个插值后的 $\beta_i$：

$$
\beta_i = \beta_{\text{slow}} + (\beta_{\text{fast}} - \beta_{\text{slow}})\cdot \text{power}_i
$$

然后根据维度是否落在高频区域，定义最终的缩放因子：

$$
\text{scale}_i =
\begin{cases}
\displaystyle
\frac{\beta_i \cdot s - \beta_i + 1}{\beta_i + s},
& i < \text{corr_dim} \quad \text{(高频 / 短波长，复杂缩放)} \\[10pt]
\displaystyle
\frac{1}{s},
& i \ge \text{corr_dim} \quad \text{(低频 / 长波长，简单缩放)}
\end{cases}
$$

最终，新的旋转频率可以表示为：

$$
\text{freqs}_i^{\text{new}} = \text{freqs}_i \cdot \text{scale}_i
$$

通过这种方式，YaRN 在保证长距离位置信息可区分性的同时，尽量减少对原有 RoPE 表达能力的破坏。

## 2.3 代码实现

下面我们提供 RoPE 及其 YaRN 扩展的 PyTorch 实现代码。这段代码展示了如何预计算频率基以及如何将旋转位置编码应用于查询（Query）和键（Key）。

### 2.3.1 预计算频率（`precompute_freqs_cis`）

这个函数的主要目的是为了预先计算好所有位置所需的 $\sin$ 和 $\cos$ 矩阵，以便在后续的 `apply` 阶段直接查表使用。它可以分为两个阶段：标准 RoPE 的频率计算和 YaRN 的修正计算。

**1. 标准 RoPE 频率计算**
首先，根据 RoPE 的定义计算基础频率 $\theta_i$。代码中对应的是：
```python
freqs = 1.0 / rope_base ** torch.arange(0, dim, 2)[: dim // 2].float() / dim
```
这里有一个细节：RoPE 是两两一组维度的旋转，所以频率数量只有 `dim // 2` 个。生成的 `freqs` 对应了公式中的 $\theta_i$ 序列，从高频（低索引）到低频（高索引）排列。

**2. YaRN 修正 (如果需扩展上下文)**
这部分逻辑对应论文中的核心算法：
*   **寻找 `corr_dim`**：代码通过 `next(...)` 寻找第一个满足 `wavelength > original_max` 的维度索引。这个索引将维度划分为“高频不安全区”和“低频安全区”。
*   **计算平滑因子 `beta`**：利用 `power` 变量，生成一个从 `beta_slow` 到 `beta_fast` 过渡的系数，用于平滑高低频的边界。
*   **计算缩放因子 `scale`**：
    *   对于 **高频部分** (`i < corr_dim`)：使用插值公式 `(beta * factor - beta + 1) / (beta * factor)` 进行较复杂的缩放，以解决相位混叠。
    *   对于 **低频部分** (`i >= corr_dim`)：直接使用 `1.0 / factor` 进行简单的线性缩放，因为这些维度在长距离下依然相对稳定。
*   最后执行 `freqs = freqs * scale` 完成频率修正。

**3. 生成完整的位置编码表**
得到修正后的频率后，需要将其扩展到具体的序列长度 `end` 上。使用 `torch.outer` 将时间步 $t$ 与频率 $\theta$ 相乘，得到 $m\theta$ 矩阵。最后通过 `torch.cat` 拼接两份相同的 cos/sin，以匹配 `head_dim` 的形状（因为 RoPE 实现中通常将 $x, y$ 当作 $x, x, ..., y, y$ 或者交错处理，这里代码采用的是拼接方式匹配 hidden states）。

### 2.3.2 应用旋转（`apply_rotary_pos_emb`）

这个函数负责将预计算好的频率真正作用到 Query 和 Key 上。核心在于如何高效地实现二维向量的旋转操作。

**1. `rotate_half` 的作用**
根据复数旋转公式：
$$
(x + iy) e^{i\theta} = (x\cos\theta - y\sin\theta) + i(x\sin\theta + y\cos\theta)
$$
实部为 $x\cos\theta - y\sin\theta$，虚部为 $y\cos\theta + x\sin\theta$。
代码中的实现采用了实数矩阵运算的技巧。`q` 和 `k` 的最后一维包含成对的 $(x, y)$。
`rotate_half(x)` 函数实现了将向量 $(x, y)$ 变为 $(-y, x)$ 的操作：
```python
def rotate_half(x):
    # 将后半部分移到前面并取反，前半部分移到后面
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)
```
这对应着公式中与 $\sin\theta$ 相乘项的符号变换。

**2. 最终的旋转公式**
结合 `precompute_freqs_cis` 返回的 `cos` 和 `sin`，代码执行了如下运算：
```python
q_embed = (q * cos) + (rotate_half(q) * sin)
```
我们可以验证其正确性：
*   第一项 `q * cos` 提供了 $x\cos\theta$ 和 $y\cos\theta$。
*   第二项 `rotate_half(q) * sin` 提供了 $-y\sin\theta$ 和 $x\sin\theta$。
两者相加正好得到：
*   前一半维度：$x\cos\theta - y\sin\theta$ （旋转后的实部/x坐标）
*   后一半维度：$y\cos\theta + x\sin\theta$ （旋转后的虚部/y坐标）

这样就完成了一次完整的旋转位置编码注入。

### 2.3.3 完整实现

```python
def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
"""
预计算旋转位置编码（RoPE）的频率。

参数:
    dim (int): 维度。
    end (int, 可选): 推入的序列长度，默认值为 32768。
    rope_base (float, 可选): 基础频率，默认值为 1e6。
    rope_scaling (dict, 可选): 缩放参数，用于 YaRN 缩放，默认值为 None。

返回:
    freqs_cos (torch.Tensor): 余弦频率，形状为 (end, dim // 2)。
    freqs_sin (torch.Tensor): 正弦频率，形状为 (end, dim // 2)。
"""
    # 写出最初的 RoPE 式子
    """
    freqs_i = \frac{1}{rope_base^{\frac{2i}{dim}}}
    两两旋转： freqs 这里是 i，而指数那里是 2i，因为是两两一组进行旋转
    每个频率对应两个维度
    """
    freqs = 1.0 / rope_base ** torch.arange(0, dim, 2)[: dim // 2].float()/dim # 由于是两两一组进行旋转，所以以 2 为步长

    # YaRN
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        # 计算 corr_dim：从 0 开始，找第一个满足 波长 > 训练最大长度 的维度索引 i，也就是需要修正的维度，公式里面的 min 指的是最靠前的。为什么要找最靠前的？
        corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)

        # 计算 power
        power = torch.arange(0, dim//2, device=freqs.device).float() / max(dim//2 - 1, 1)

        # 计算 beta
        beta = beta_slow + (beta_fast - beta_slow) * power

        # 计算 scale
        scale = torch.where(
            torch.arange(0, dim//2, device=freqs.device).float() < corr_dim,
            (beta * factor - beta + 1)/beta * factor,
            1.0 / factor
        )

        # 应用 scale
        freqs = freqs * scale
    # 生成位置索引，与频率相乘
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float() # [end, dim//2] 的矩阵

    # 返回一个 cos 和 sin
    freqs_cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)
    freqs_sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):

    # 实数域上的旋转：[a, b] => [-b, a]
    def rotate_half(x):
        # x.shape[-1] 取最后一个维度的中点
        # [-x[..., x.shape[-1] //2:] 取后半部分
        # [x[..., :x.shape[-1] //2] 取前半部分
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    
    # 应用旋转位置编码
    # x_rotated = x  * cos + rotate_half(x) * sin
    # unsqueeze 用于后续的维度扩展
    # 计算 q_embed, k_embed
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed
```
