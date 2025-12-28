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

# 2.1 RoPE (Rotary Positional Embedding)

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

# 2.2 RoPE 的实际应用

RoPE 不是直接旋转词向量 embedding，而是在注意力里对每个位置的 Query (Q)、Key (K) 做旋转（通常 Value (V) 不旋转）。

## 2.2.2 从隐藏状态到 Q/K/V

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

## 2.2.3 RoPE 对 Q/K 的“旋转”是怎么做的

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

## 2.2.4 一个例子（只演示某个 pair）

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

# 2.2 YaRN (Yet another RoPE extensioN method)

## 2.2.1 为什么要引入 YaRN 呢？

RoPE 中每一对维度都会对应一个固定的旋转频率，其定义为：

$$
\text{freqs}_i = \frac{1}{\text{rope_base}^{\frac{2i}{\text{dim}}}}
$$

从这个公式可以看出，随着维度索引 $i$ 的增大，旋转频率会逐渐减小：低维部分对应较高的旋转频率，高维部分对应较低的旋转频率。

在序列长度不太长的情况（比如 4096）下，这样的频率分布通常是没有问题的。但当我们把上下文长度拉得很长（比如超过模型训练时见过的最大长度 4096， 如5000）时，就会出现问题。

直观来说，对于高频维度，当相对位置变得非常大时，不同距离对应的旋转角度可能会“转到同一个位置”上。也就是说，模型在远处计算得到的位置信息，在经过 RoPE 旋转之后，可能会和某些近处位置产生非常相似的表示。

一旦出现这种情况，模型在注意力计算中就很难区分「这是一个很远的 token」还是「这是一个比较近的 token」，远近关系被混淆，最终会影响模型在长上下文场景下的训练稳定性和效果。这也是为什么原始 RoPE 在长序列外推时容易出现性能下降的问题。

## 2.2.3 YaRN

YaRN 的核心思想并不是简单地整体缩放 RoPE 的频率，而是根据不同维度所对应的“波长”，对高频和低频部分采用不同的缩放策略。

在原始 RoPE 中，第 $i$ 个维度对应的旋转频率为 $\text{freqs}_i$。YaRN 的目标是对这些频率引入一个位置相关的缩放因子，使其在长上下文场景下更加稳定。

实际计算过程可以分为以下几步。



**第一步：确定需要调整的维度范围**

我们可以通过波长来判断某个维度是否已经“超出了模型原本能够理解的上下文范围”。

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

在此之前的维度被视为高频（短波长）部分，容易在长序列下产生混叠；而之后的维度则属于低频（长波长）部分，对长距离信息更加稳定。



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
