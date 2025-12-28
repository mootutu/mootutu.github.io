---
layout: learning-post-layout
title: "2 Positional Encoding RoPE&YaRN"
date: 2025-12-26
lang: en
topic_url: /learning/minimind.html
translate_url: /cn/learning/minimind/part2.html
mathjax: true
---

## 2.1 RoPE (Rotary Positional Embedding)

Assume there are two 2D vectors $\mathbf{q}$ and $\mathbf{k}$, representing the **query** vector and **key** vector respectively:

$$
\mathbf{q} = [q_1, q_2], \qquad \mathbf{k} = [k_1, k_2]
$$

To rotate a vector by an angle $\theta$, we introduce a 2D rotation matrix $\mathbf{R}(\theta)$, defined as:

$$
\mathbf{R}(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

where $\cos\theta$ and $\sin\theta$ are the cosine and sine values of the angle $\theta$.

Multiplying the rotation matrix with the original vector yields the rotated vector:

$$
\mathbf{q}' = \mathbf{R}(\theta)\mathbf{q}, \qquad
\mathbf{k}' = \mathbf{R}(\theta)\mathbf{k}
$$

where $\mathbf{q}'$ and $\mathbf{k}'$ represent the **query** and **key** vectors rotated by an angle $\theta$.

We apply a rotation to the **query** vector $\mathbf{q}$ with an angle $m\theta$, and to the **key** vector $\mathbf{k}$ with an angle $n\theta$. Here, $m$ and $n$ are the position indices of the **query** and **key**: $m$ indicates that the **query** vector $\mathbf{q}$ comes from the $m$-th position in the sequence (the $m$-th token), and $n$ indicates that the **key** vector $\mathbf{k}$ comes from the $n$-th position in the sequence (the $n$-th token). The rotated **query** and **key** vectors are:

$$
\mathbf{q_m} = [q_1\cos (m\theta) - q_2\sin (m\theta), q_1\sin (m\theta) + q_2\cos (m\theta)] \\
\mathbf{k_n} = [k_1\cos (n\theta) - k_2\sin (n\theta), k_1\sin (n\theta) + k_2\cos (n\theta)]
$$

Calculating the dot product of the two and expanding it step by step:

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

Since $q_1$, $k_1$, $q_2$, $k_2$ are given values, and $\theta$ is also a given value, the entire expression depends only on $(m-n)$. This means the final positional information is determined solely by $(m-n)$.

After RoPE, we use the rotated $q'_m, k'_n$ to calculate the attention score:

$$\text{score}(m, n) = \frac{(q'_m)^\top (k'_n)}{\sqrt{d}}$$

Because $q$ and $k$ both carry the rotation angles of their respective positions ($m\theta$, $n\theta$), the dot product naturally encodes relative positional information (strongly correlated with $m-n$).

## 2.2 Practical Application of RoPE

RoPE does not directly rotate the word vector embedding, but rotates the Query (Q) and Key (K) at each position in the attention mechanism (Value (V) is usually not rotated).

### 2.2.2 From Hidden State to Q/K/V

In a certain Transformer layer, each position has a hidden state vector:

$$
\mathbf{h}_m = [h_{m1}, h_{m2}, \ldots, h_{d}] \in \mathbb{R}^d
$$

The **Query**/**Key**/**Value** required for attention are obtained through linear projection (taking a single head as an example):

$$
q_m = h_m W_q,\quad k_m = h_m W_k,\quad v_m = h_m W_v
$$

Where:

- $q_m, k_m, v_m \in \mathbb{R}^{d}$
- $d$ is the dimension of the head (head_dim)

### 2.2.3 How RoPE "Rotates" Q/K

Split `head_dim` into groups of 2. Assuming $d = 8$:

$$
q_m = [q_{m,0}, q_{m,1}, q_{m,2}, q_{m,3}, q_{m,4}, q_{m,5}, q_{m,6}, q_{m,7}]
$$

Group them in pairs:

- Group 0: $(q_{m,0}, q_{m,1})$
- Group 1: $(q_{m,2}, q_{m,3})$
- Group 2: $(q_{m,4}, q_{m,5})$
- Group 3: $(q_{m,6}, q_{m,7})$

The same applies to $k_m$.

Each group has its own frequency $\theta_i$, and the angle increases linearly with position. For the $i$-th group (a 2D pair), the rotation angle is:

$$\text{angle}_{m,i} = m \cdot \theta_i$$

Where:

- $m$ is the position index of the token (the $m$-th token)
- $\theta_i$ is the frequency corresponding to dimension group $i$ (usually generated based on $10000^{-2i/d}$)

Rotate each pair using the rotation matrix. For a certain group of 2D vectors $(x, y)$, the rotated result $(x', y')$ is:

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

### 2.2.4 An Example (Demonstrating a Single Pair)

Take a sentence as an example:

> "I(0) like(1) eating(2) apples(3)"

Take the token "eating", its position is $m=2$.

Assume for the 0th pair of a certain head (for illustration only):

- Original $q$ pair: $(x, y) = (0.5, -1.0)$
- Original $k$ pair: $(x, y) = (1.2, 0.3)$
- Frequency of this pair: $\theta_0 = 0.1$

Then the rotation angle is:

$$m\theta_0 = 2 \times 0.1 = 0.2 \text{ rad}$$

After rotation (rounded):

- $q' \approx (0.6887, -0.8807)$
- $k' \approx (1.1165, 0.5324)$

**Note: This is just a 2D fragment of Q/K; in practice, the same operation is performed for all pairs and all heads, only the $\theta_i$ for each pair is different.**

## 2.2 YaRN (Yet another RoPE extensioN method)

### 2.2.1 Why Introduce YaRN?

In RoPE, each pair of dimensions corresponds to a fixed rotation frequency, defined as:

$$
\text{freqs}_i = \frac{1}{\text{rope\_base}^{\frac{2i}{\text{dim}}}}
$$

From this formula, it can be seen that as the dimension index $i$ increases, the rotation frequency gradually decreases: low-dimensional parts correspond to higher rotation frequencies, and high-dimensional parts correspond to lower rotation frequencies.

When the sequence length is not too long (e.g., 4096), such a frequency distribution is usually fine. But when we stretch the context length very long (e.g., exceeding the maximum length seen during model training, such as 5000), problems arise.

Intuitively, for high-frequency dimensions, when the relative position becomes very large, the rotation angles corresponding to different distances may "rotate to the same position". That is to say, the positional information obtained by the model at a distance may produce a very similar representation to some nearby positions after RoPE rotation.

Once this happens, it becomes difficult for the model to distinguish in the attention calculation whether "this is a very distant token" or "this is a relatively close token". The distance relationship becomes confused, which ultimately affects the training stability and performance of the model in long context scenarios. This is also why the original RoPE tends to suffer from performance degradation during long sequence extrapolation.

### 2.2.3 YaRN

The core idea of YaRN is not simply to scale the RoPE frequency as a whole, but to adopt different scaling strategies for high-frequency and low-frequency parts according to the "wavelength" corresponding to different dimensions.

In the original RoPE, the rotation frequency corresponding to the $i$-th dimension is $\text{freqs}_i$. YaRN's goal is to introduce a position-dependent scaling factor to these frequencies to make them more stable in long context scenarios.

The actual calculation process can be divided into the following steps.

**Step 1: Determine the Range of Dimensions to Adjust**

We can use the wavelength to judge whether a dimension has "exceeded the context range that the model can originally understand".

Intuitively, in RoPE, each dimension is actually using a **periodic rotation** to encode positional information. For the $i$-th dimension, the rotation angle corresponding to position $m$ can be written as:
$$
\theta_i(m) = m \cdot \text{freqs}_i
$$
Since the rotation angle is periodic with $2\pi$, when the relative position change causes the rotation angle to increase by $2\pi$, the representation of that dimension will return to the same phase. This means that once the relative distance exceeds a certain threshold, that dimension will no longer be able to distinguish further positional differences.

Therefore, we can use "the distance corresponding to a full rotation" to measure the maximum relative position range that the dimension can distinguish. This quantity is the wavelength. The wavelength can be approximated as:
$$
\lambda_i = \frac{2\pi}{\text{freqs}[i]}
$$

When the wavelength corresponding to a certain dimension is less than or close to the maximum context length seen during the training phase (denoted as $\text{original\_max}$), it implies that phase repetition has already started to occur for that dimension within the training range; and in longer sequences, this repetition will be further aggravated, causing aliasing of far and near positional information.

Based on this observation, YaRN determines which dimensions are "unsafe" in long sequence scenarios by comparing the wavelength with the maximum context length during training.

Therefore, we define a split dimension:

$$
\text{corr\_dim} = \min \left\{ i \mid \frac{2\pi}{\text{freqs}[i]} > \text{original\_max} \right\}
$$

Dimensions before this are considered high-frequency (short wavelength) parts, which are prone to aliasing in long sequences; while dimensions after this belong to low-frequency (long wavelength) parts, which are more stable for long-distance information.

**Step 2: Calculate Interpolation Weights for Different Dimensions**

To transition smoothly between high frequency and low frequency, YaRN defines a normalized weight for each dimension:

$$
\text{power}_i = \frac{i}{\frac{\text{dim}}{2} - 1}
$$

This weight is used to control the interpolation ratio of different dimensions in the subsequent scaling process.

**Step 3: Calculate the Scaling Factor for Each Dimension**

YaRN uses different scaling strategies in high-frequency and low-frequency regions respectively.
First, define an interpolated $\beta_i$ for each dimension:

$$
\beta_i = \beta_{\text{slow}} + (\beta_{\text{fast}} - \beta_{\text{slow}})\cdot \text{power}_i
$$

Then, define the final scaling factor based on whether the dimension falls in the high-frequency region:

$$
\text{scale}_i =
\begin{cases}
\displaystyle
\frac{\beta_i \cdot s - \beta_i + 1}{\beta_i + s},
& i < \text{corr\_dim} \quad \text{(High freq / Short wavelength, Complex scaling)} \\[10pt]
\displaystyle
\frac{1}{s},
& i \ge \text{corr\_dim} \quad \text{(Low freq / Long wavelength, Simple scaling)}
\end{cases}
$$

Finally, the new rotation frequency can be expressed as:

$$
\text{freqs}_i^{\text{new}} = \text{freqs}_i \cdot \text{scale}_i
$$

In this way, YaRN minimizes the damage to the original RoPE expressive ability while ensuring the distinguishability of long-distance positional information.
