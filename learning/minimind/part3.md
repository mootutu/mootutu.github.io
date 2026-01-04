---
layout: learning-post-layout
title: "3 Attention Mechanism"
date: 2025-12-27
lang: en
topic: minimind
order: 3
topic_url: /learning/minimind.html
translate_url: /cn/learning/minimind/part3.html
mathjax: true
---

In this chapter, we will dive deep into the attention mechanism and trace the evolution from **MHA (Multi-Head Attention)** to **MQA (Multi-Query Attention)** to **GQA (Grouped Query Attention)**. The core motivation behind this evolution is: **how to reduce memory usage and computational overhead during inference while maintaining model quality**.

## 3.1 The Essence of Attention: Query, Key, Value

### Intuitive Understanding

The essence of Attention is **relevance-based weighted aggregation**: given a query, we want to extract the most useful information from a set of data based on "relevance".

The standard Scaled Dot-Product Attention formula is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k$ is the dimension of the Key vector. Dividing by $\sqrt{d_k}$ prevents the dot product from becoming too large, which would cause softmax gradients to vanish.

### Professional Interpretation of Q, K, V

In Transformers, Q, K, V are all obtained through linear projections from the same input sequence (Self-Attention) or different sequences (Cross-Attention). Their roles can be understood from an **information retrieval** perspective:

| Symbol | Name | Role | Analogy |
|--------|------|------|---------|
| Q | Query | "Query vector": represents what information the current position wants to obtain | Search terms in a search engine |
| K | Key | "Key vector": represents what type of information each position can provide | Document index/tags |
| V | Value | "Value vector": represents the actual information content at each position | Document body content |

**Intuition behind the computation process**:
1. **Matching phase**: Compute dot products between Query and all Keys ($QK^T$) to get "similarity scores"
2. **Normalization phase**: Convert scores to "attention weights" (probability distribution) via softmax
3. **Aggregation phase**: Compute weighted sum of Values using attention weights to get the final output

**Why separate K and V?**

A natural question is: why not match Query directly with Value, instead of introducing Key as an intermediate layer?

The answer lies in **decoupling "matching" from "content"**:
- Key determines "who should be attended to" (relevance computation)
- Value determines "what to provide when attended to" (information content)

This separation allows the model to learn more flexible attention patterns. For example, two positions might be similar in "syntactic function" (similar Keys) but carry different "semantic content" (different Values).

### From a Linear Algebra Perspective

Let the input sequence be $X \in \mathbb{R}^{n \times d}$ ($n$ tokens, each $d$-dimensional), then:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$.

The attention output is:

$$
\text{head} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \in \mathbb{R}^{n \times d_v}
$$

The $(i, j)$-th element of matrix $QK^T \in \mathbb{R}^{n \times n}$ represents "the attention score from position $i$ to position $j$".

### Deep Dive: Why Divide by $\sqrt{d_k}$?

This scaling factor seems simple but is crucial for training stability. Let's analyze it from the perspective of variance.

Assume each element of $Q$ and $K$ is an i.i.d. random variable with mean 0 and variance 1. Then the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has the following statistical properties:

$$
\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
$$

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k
$$

**The problem**: When $d_k$ is large (e.g., 64 or 128), the variance of the dot product becomes large, causing some values to become extremely large or small.

When these values enter softmax:
- Extremely large positive values cause softmax outputs to approach 1
- Other values' outputs approach 0
- **Softmax enters saturation, gradients become nearly zero**

After dividing by $\sqrt{d_k}$, the variance of the scaled dot product becomes:

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

This ensures that regardless of how large $d_k$ is, the dot product distribution stays within a reasonable range, allowing softmax to work properly.

### Self-Attention vs Cross-Attention

Depending on the source of Q, K, V, Attention is divided into two types:

| Type | Q Source | K, V Source | Typical Application |
|------|----------|-------------|---------------------|
| Self-Attention | Input sequence X | Input sequence X | Encoder, Decoder self-attention layers |
| Cross-Attention | Target sequence Y | Source sequence X | Decoder attending to Encoder output |

**Self-Attention**: Tokens within the sequence attend to each other, capturing intra-sequence dependencies.

**Cross-Attention**: One sequence attends to another. The typical application is machine translation where the Decoder attends to the Encoder's output. The Decoder uses its own tokens as Query to find relevant information from the Encoder's output (Key and Value).

## 3.2 Multi-Head Attention (MHA)

### Why Multiple Heads?

Single-head attention can only learn **one attention pattern**. But in natural language, different types of relationships require different "attention modes":
- Syntactic relationships: subject attending to predicate
- Coreference relationships: pronouns attending to their referents
- Semantic relationships: associations between synonyms

Multi-Head Attention allows the model to **learn multiple attention patterns in parallel**.

### Structure

MHA projects the input into $h$ different subspaces, computes attention independently in each, then concatenates:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

where each head:

$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

**Key point**: Each head has **independent** $W_Q^i$, $W_K^i$, $W_V^i$ parameters.

The complete MHA computation can be expressed as:

$$
\begin{aligned}
Q_i &= X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i \quad &\text{(independent projection per head)} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \quad &\text{(compute attention)} \\
\text{Output} &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O \quad &\text{(concatenate and project)}
\end{aligned}
$$

where $i = 1, 2, \ldots, h$.

**Parameter summary**:
- Each head: $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d \times d_k}$
- Total: $h$ independent sets of Q, K, V projections
- KV Cache stores: $h$ sets of K and $h$ sets of V

### Implementation Details: Split Heads vs Separate Projections

In practice, MHA has two equivalent implementation approaches:

**Approach 1: Separate Projections**

Define independent projection matrices $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d \times d_k}$ for each head:

```python
# Conceptual implementation (less efficient)
heads = []
for i in range(n_heads):
    Q_i = X @ W_Q[i]  # [n, d_k]
    K_i = X @ W_K[i]  # [n, d_k]
    V_i = X @ W_V[i]  # [n, d_v]
    head_i = attention(Q_i, K_i, V_i)
    heads.append(head_i)
output = concat(heads) @ W_O
```

**Approach 2: Merged Projection + Split (Actually Used)**

Use one large projection matrix, then split the result into multiple heads:

```python
# Actual efficient implementation
W_Q = [d_model, n_heads * d_k]  # Merged Q projection for all heads
W_K = [d_model, n_heads * d_k]
W_V = [d_model, n_heads * d_v]

Q = X @ W_Q  # [n, n_heads * d_k]
K = X @ W_K
V = X @ W_V

# Reshape into multi-head format
Q = Q.reshape(n, n_heads, d_k).transpose(1, 2)  # [n_heads, n, d_k]
K = K.reshape(n, n_heads, d_k).transpose(1, 2)
V = V.reshape(n, n_heads, d_v).transpose(1, 2)

# Batch compute attention for all heads
output = batched_attention(Q, K, V)  # [n_heads, n, d_v]
output = output.transpose(1, 2).reshape(n, n_heads * d_v)  # [n, d_model]
output = output @ W_O
```

These two approaches are mathematically equivalent, but approach 2 leverages the parallelism of matrix operations for better efficiency.

### head_dim Design Considerations

In practice, we typically have:

$$
d_k = d_v = \text{head_dim} = \frac{d_{\text{model}}}{n_{\text{heads}}}
$$

For example, with $d_{\text{model}} = 768$, $n_{\text{heads}} = 12$:

$$
\text{head_dim} = \frac{768}{12} = 64
$$

**Why this design?**

1. **Constant computation**: Total parameters $n_{\text{heads}} \times d_{\text{model}} \times \text{head_dim} = d_{\text{model}}^2$, same as single head
2. **Parallel computation**: All heads can compute simultaneously
3. **Trade-off between expressiveness and efficiency**: More heads means more attention patterns, but lower dimension per head

### The Role of Output Projection $W_O$

After concatenating all heads' outputs, we apply a linear transformation $W_O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

**Why do we need $W_O$?**

1. **Dimension matching**: Ensures output dimension matches input for residual connection
2. **Information fusion**: Allows information from different heads to interact and combine
3. **Increased expressiveness**: Additional learnable parameters

Without $W_O$, the outputs of different heads are simply concatenated without any interaction.

### Patterns Learned by Different Heads

Research shows that different attention heads automatically learn different types of language patterns:

| Head Type | Attention Pattern | Example |
|-----------|------------------|---------|
| Positional head | Attends to fixed relative positions | Always attends to previous or next word |
| Syntactic head | Attends to syntactically related words | Verbs attend to their subjects/objects |
| Coreference head | Attends to coreference relations | Pronoun "it" attends to its referent noun |
| Rare word head | Attends to rare or important words | Focuses on proper nouns |
| Delimiter head | Attends to sentence boundaries | Attends to periods, commas, etc. |

This "division of labor" is automatically learned by the model, requiring no manual design.

### Advantages and Costs

**Advantages**: Strong expressive power; different heads can capture different types of dependencies.

**Costs**: Parameters and computation increase with the number of heads. More importantly, during **inference**, a serious memory bottleneck emerges — the KV Cache problem.

## 3.3 Inference Bottleneck: KV Cache

### Characteristics of Autoregressive Generation

Large language models generate text in an **autoregressive** manner: generating one token at a time, adding it to the input sequence, then continuing to generate the next.

```
Step 1: "The"           → generate "cat"
Step 2: "The cat"       → generate "sat"
Step 3: "The cat sat"   → generate "on"
...
```

### Problem: Redundant Computation

At step $t$, we need to compute attention from the current token to **all previous tokens**. If we recompute K and V for all positions every time, the computation is $O(t)$, making the overall complexity $O(n^2)$.

### Solution: KV Cache

Observation: K and V for previous positions **don't change** (they only depend on previous inputs). So we can **cache** the computed K and V:

```
Step 1: Compute K_1, V_1, cache
Step 2: Compute K_2, V_2, cache; reuse K_1, V_1
Step 3: Compute K_3, V_3, cache; reuse K_1, V_1, K_2, V_2
...
```

### New Problem: Memory Usage

KV Cache size is:

$$
\text{KV Cache Size} = 2 \times \text{layers} \times \text{heads} \times \text{seq_length} \times \text{head_dim}
$$

For a typical LLM (e.g., LLaMA-7B with 32 layers, 32 heads, head_dim=128, sequence length 4096):

$$
2 \times 32 \times 32 \times 4096 \times 128 \times 2\text{ bytes} \approx 2\text{ GB}
$$

This is just the cache for a **single request**. In high-concurrency scenarios, KV Cache becomes a memory bottleneck.

**This is the core motivation for optimizing attention mechanisms: reducing KV Cache size.**

### Deep Dive: Prefill Phase vs Decode Phase

LLM inference can be divided into two distinct phases:

**Prefill Phase (Initial Computation)**

Process the user's input prompt, compute K and V for all tokens at once and cache them.

Characteristics:
- **Compute-bound**: Needs to process the entire prompt
- Highly parallelizable
- Main bottleneck is computation speed

**Decode Phase (Token-by-Token Generation)**

Generate one token at a time, compute its K and V, append to cache.

Characteristics:
- **Memory-bound**: Only computes 1 token per step, but must access the entire KV Cache
- Difficult to parallelize (autoregressive dependency)
- Main bottleneck is memory bandwidth

This explains why KV Cache optimization is so important: Decode phase speed is mainly limited by the speed of reading KV Cache. Reducing Cache size can directly improve inference speed.

### Why Only Cache K and V, Not Q?

This is a great question. Let's analyze the structure of attention computation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

In autoregressive generation, when generating the $t$-th token:

| Variable | Dimension | Source | Need to Cache? |
|----------|-----------|--------|----------------|
| $Q_t$ | $[1, d_k]$ | Current token only | **No** |
| $K_{1:t}$ | $[t, d_k]$ | All generated tokens | **Yes** |
| $V_{1:t}$ | $[t, d_v]$ | All generated tokens | **Yes** |

**Key observations**:

1. **Q only needs current token**: When computing attention at step $t$, we only need $Q_t$ (the current token's Query), not the previous $Q_1, \ldots, Q_{t-1}$

2. **K and V need all history**: The current token needs to attend to all previous tokens, so we need $K_{1:t}$ and $V_{1:t}$

Understanding through code:

```python
# Step t generation
q_t = compute_query(x_t)      # Only compute current token's Q
k_t = compute_key(x_t)        # Compute current token's K
v_t = compute_value(x_t)      # Compute current token's V

# Update cache
k_cache.append(k_t)           # Cache K
v_cache.append(v_t)           # Cache V

# Compute attention
# Q: [1, d_k], only need current token
# K: [t, d_k], need all history (read from cache)
# V: [t, d_v], need all history (read from cache)
attn = softmax(q_t @ k_cache.T / sqrt(d_k)) @ v_cache
```

This is why KV Cache only stores K and V, not Q.

### KV Cache Memory Layout

In practice, KV Cache is typically organized as follows:

**Pre-allocation strategy**: Usually pre-allocate space for `max_seq_len` to avoid memory fragmentation and copy overhead from dynamic expansion.

**Detailed memory formula**:

$$
\text{Memory} = 2 \times L \times B \times H \times S \times D \times \text{bytes_per_element}
$$

Where:
- $L$ = number of layers
- $B$ = batch size
- $H$ = number of heads (this is the key for MHA/MQA/GQA optimization!)
- $S$ = sequence length
- $D$ = head_dim
- bytes_per_element = 2 (fp16) or 4 (fp32)

## 3.4 Multi-Query Attention (MQA)

### Core Idea

MQA (Shazeer, 2019) proposed a radical approach: **all heads share the same K and V**, only Q remains multi-headed.

Expressed in formulas:

$$
\begin{aligned}
Q_i &= X W_Q^i \quad &\text{(each head has independent Q projection, } i = 1, \ldots, h \text{)} \\
K &= X W_K \quad &\text{(all heads share the same K)} \\
V &= X W_V \quad &\text{(all heads share the same V)} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V \quad &\text{(each } Q_i \text{ computes attention with the same K, V)}
\end{aligned}
$$

**Parameter summary**:
- Q projections: $h$ independent $W_Q^i$
- K/V projections: only 1 shared $W_K$, $W_V$
- KV Cache stores: **1 set of K and 1 set of V** (reduced to $1/h$ of MHA)

### Effects

**Advantages**: KV Cache reduced to $1/h$ of original. For a 32-head model, memory usage drops to 1/32.

**Disadvantages**: All heads are forced to use the same K and V for attention computation, **limiting expressive power** and potentially degrading model quality.

## 3.5 Grouped Query Attention (GQA)

### Core Idea

GQA (Ainslie et al., 2023) is a **compromise between MHA and MQA**: divide $h$ Query heads into $g$ groups, with each group sharing one set of K and V.

Expressed in formulas (assuming each group has $h/g$ Q heads):

$$
\begin{aligned}
Q_i &= X W_Q^i \quad &\text{(each head has independent Q projection, } i = 1, \ldots, h \text{)} \\
K_j &= X W_K^j \quad &\text{(shared K for group } j \text{, } j = 1, \ldots, g \text{)} \\
V_j &= X W_V^j \quad &\text{(shared V for group } j \text{, } j = 1, \ldots, g \text{)} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K_{g(i)}^T}{\sqrt{d_k}}\right) V_{g(i)} \quad &\text{(} Q_i \text{ uses its group's } K_{g(i)}, V_{g(i)} \text{)}
\end{aligned}
$$

where $g(i) = \lceil i \cdot g / h \rceil$ denotes which group the $i$-th Q head belongs to.

**Parameter summary**:
- Q projections: $h$ independent $W_Q^i$
- K/V projections: $g$ shared $W_K^j$, $W_V^j$
- KV Cache stores: **$g$ sets of K and $g$ sets of V** (reduced to $g/h$ of MHA)

### Special Cases

- When $g = h$: each Q head has its own KV → **degenerates to MHA**
- When $g = 1$: all Q heads share one KV → **degenerates to MQA**

### Advantages

GQA achieves a good balance between **inference efficiency** and **model quality**:
- KV Cache reduced to $g/h$ (e.g., with $g=4, h=32$, reduced to 1/8)
- Quality loss is minimal, close to original MHA

## 3.6 Comparison Summary

| Method | Q Heads | K/V Heads | KV Cache Size | Model Quality | Representative Models |
|--------|---------|-----------|---------------|---------------|----------------------|
| MHA    | $h$     | $h$       | 1×            | Highest       | GPT-3, BERT |
| MQA    | $h$     | 1         | $1/h$         | May decrease  | PaLM |
| GQA    | $h$     | $g$       | $g/h$         | Close to MHA  | LLaMA-2, Mistral |

**One-sentence summary**:

> MHA prioritizes expressive power, MQA prioritizes maximum efficiency, and GQA finds a practical balance between the two — by sharing KV across groups, it trades a small quality cost for significant inference speedup.

## Supplementary: ResNet (Residual Network)

In the early days of deep learning, researchers observed a counterintuitive phenomenon: **deeper networks could actually perform worse**.

Logically, a 56-layer network should be at least as powerful as a 20-layer one — in the worst case, those extra 36 layers could simply learn identity mappings, and the network should perform just as well. However, experiments showed that the 56-layer network had a **higher training error**.

Note that we're talking about **training error**, not test error. This isn't overfitting — the network simply failed to learn effectively. This is an **optimization problem**.

This leads to two core challenges:
1. **Function fitting problem**: Making a network learn to "do nothing" (identity mapping) is actually difficult
2. **Gradient propagation problem**: In very deep networks, gradients tend to vanish during backpropagation

ResNet's key insight is: instead of learning the target function directly, let the network learn "the difference between the target function and the input" — this is what we call the **residual**.

### 3.1 Why Is Residual Learning Easier for Function Fitting?

#### Traditional vs. Residual Structure

Suppose we want a layer to learn a target function $H(x)$:

| Structure | What the network learns |
|-----------|------------------------|
| Traditional | Learn $H(x)$ directly |
| Residual | Learn $F(x) = H(x) - x$, then output $H(x) = F(x) + x$ |

#### An Extreme but Important Example

Suppose the "perfect answer" at some deep layer happens to be $H(x) = x$ (identity mapping).

**The traditional approach's dilemma**: The network needs to carefully adjust all weights so that the output exactly equals the input. This requires weight matrices close to identity matrices and biases near zero — a precise "target point" that the optimizer must reach.

**The residual approach's advantage**: Since $H(x) = F(x) + x$, achieving $H(x) = x$ only requires $F(x) = 0$. Making the output approach zero is very easy for neural networks — just push all the weights toward zero.

#### An Analogy

Imagine you need to paint a picture:

- **Traditional way**: Start from a blank canvas, paint the complete artwork
- **Residual way**: First print a reference image on the canvas, then only paint "the parts that need modification"

If the reference image is already close to the target, the residual approach requires minimal changes; if the gap is large, it won't be worse either. This is the "lower bound guarantee" of residual learning.

### 3.2 The Significance of Residual Structure for Gradient Propagation

Now let's look at the gradient formula.

#### Backpropagation Derivation

Let the loss function be $L$, and the residual block's output be $H(x) = F(x) + x$.

By the chain rule:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial H}{\partial x}
$$

Since $H(x) = F(x) + x$, taking the derivative with respect to $x$:

$$
\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + \frac{\partial x}{\partial x} = \frac{\partial F}{\partial x} + 1
$$

Therefore:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)
$$

#### Why Is This "+1" Important?

In traditional networks, gradients depend entirely on $\frac{\partial F}{\partial x}$. If $F$'s gradient is small (say 0.1), after 50 layers:

$$
0.1^{50} \approx 10^{-50} \quad \text{(essentially zero)}
$$

This is **gradient vanishing**: early layers receive almost no gradient signal and cannot learn.

In residual structures, even if $\frac{\partial F}{\partial x}$ is small, the "+1" term remains in the gradient expression. Expanding across multiple layers:

$$
\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_n} \cdot \prod_{i=0}^{n-1} \left( \frac{\partial F_i}{\partial x_i} + 1 \right)
$$

Expanding this product, you'll find it contains a "direct path":

$$
\frac{\partial L}{\partial x_n} \cdot 1 \cdot 1 \cdot 1 \cdots = \frac{\partial L}{\partial x_n}
$$

This means **gradients can flow directly to the earliest layers without loss** — this is the so-called "gradient highway".

#### An Analogy

Imagine a game of telephone:
- **Traditional network**: Each person can only pass the message to the next person; after 50 passes, the message is severely distorted
- **Residual network**: Besides passing messages, everyone has a direct phone line to the source, ensuring critical information isn't lost

### 3.3 Summary

> **ResNet learns "residuals" instead of "targets", which both reduces optimization difficulty (identity mapping only requires F=0) and provides a direct path for gradients (the +1 term), making it possible to train extremely deep networks.**

## 3.7 Code Implementation

Below, we provide a PyTorch implementation of GQA (Grouped Query Attention). This code demonstrates how to implement the key components: the `repeat_kv` function for expanding KV heads, and the complete Attention class that supports MHA, MQA, and GQA through a unified interface.

### 3.7.1 The `repeat_kv` Function

The core of GQA implementation lies in the `repeat_kv` function. This function is responsible for expanding the K and V tensors so that each Query head has corresponding K and V to compute attention with.

**The Problem It Solves**

In GQA, we have:
- $h$ Query heads (e.g., 32)
- $g$ Key/Value heads (e.g., 4)

Each KV head needs to serve $h/g$ Query heads (e.g., 8 Query heads share 1 KV head). The `repeat_kv` function achieves this by repeating each KV head $n_rep = h/g$ times.

**Implementation Details**

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats KV heads to match the number of Query heads.

    Args:
        x: Input tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Tensor of shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return x  # MHA case: no repetition needed

    batch, num_kv_heads, seq_len, head_dim = x.shape

    # Insert a new dimension and expand
    # [batch, num_kv_heads, seq_len, head_dim]
    # -> [batch, num_kv_heads, 1, seq_len, head_dim]
    # -> [batch, num_kv_heads, n_rep, seq_len, head_dim]
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)

    # Reshape to merge the repeated dimension
    # -> [batch, num_kv_heads * n_rep, seq_len, head_dim]
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
```

**Understanding the Shape Transformation**

Let's trace through with a concrete example where `batch=1`, `num_kv_heads=4`, `seq_len=10`, `head_dim=64`, `n_rep=8`:

1. Input shape: `[1, 4, 10, 64]`
2. After `unsqueeze`: `[1, 4, 1, 10, 64]`
3. After `expand`: `[1, 4, 8, 10, 64]`
4. After `reshape`: `[1, 32, 10, 64]`

Now we have 32 KV "heads" that can be paired with 32 Query heads.

### 3.7.2 Attention Class Initialization

The Attention class uses `num_key_value_heads` to control whether it operates as MHA, MQA, or GQA:

**Key Configuration Parameters**

```python
def __init__(self, args: ModelConfig):
    super().__init__()

    # Number of Query heads (always the full count)
    self.n_heads = args.num_attention_heads

    # Number of KV heads (determines MHA/MQA/GQA)
    # - num_kv_heads == n_heads: MHA
    # - num_kv_heads == 1: MQA
    # - 1 < num_kv_heads < n_heads: GQA
    self.num_kv_heads = args.num_key_value_heads or args.num_attention_heads

    # Repetition factor for KV heads
    self.n_rep = self.n_heads // self.num_kv_heads

    # Dimension per head
    self.head_dim = args.hidden_size // args.num_attention_heads
```

**Projection Layer Dimensions**

The key difference from standard MHA is in the projection layer sizes:

```python
# Q projection: always full size (n_heads * head_dim)
self.wq = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)

# K and V projections: reduced size (num_kv_heads * head_dim)
self.wk = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
self.wv = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

# Output projection: back to hidden_size
self.wo = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
```

This is where the memory savings come from — the K and V projections are smaller by a factor of `n_heads / num_kv_heads`.

### 3.7.3 Full Implementation

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
    num_key_value_heads: int = 4  # GQA with 4 KV heads
    max_seq_len: int = 2048


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats KV heads to match the number of Query heads for GQA.

    Args:
        x: Input tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Tensor of shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return x

    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):
    """
    Multi-Head Attention with support for Grouped Query Attention (GQA).

    By adjusting num_key_value_heads:
    - num_key_value_heads == num_attention_heads: Standard MHA
    - num_key_value_heads == 1: Multi-Query Attention (MQA)
    - 1 < num_key_value_heads < num_attention_heads: Grouped Query Attention (GQA)
    """

    def __init__(self, args: ModelConfig):
        super().__init__()

        # Number of Query heads
        self.n_heads = args.num_attention_heads

        # Number of KV heads (for GQA)
        self.num_kv_heads = args.num_key_value_heads or args.num_attention_heads

        # Ensure n_heads is divisible by num_kv_heads
        assert self.n_heads % self.num_kv_heads == 0, \
            f"num_attention_heads ({self.n_heads}) must be divisible by num_key_value_heads ({self.num_kv_heads})"

        # Number of times to repeat each KV head
        self.n_rep = self.n_heads // self.num_kv_heads

        # Dimension per head
        self.head_dim = args.hidden_size // args.num_attention_heads

        # Linear projections
        # Q: full size, K/V: reduced size for GQA
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
        Forward pass for attention with optional KV caching.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
            freqs_cos, freqs_sin: RoPE frequency tensors
            past_kv: Optional cached (K, V) from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            output: Attention output of shape [batch, seq_len, hidden_size]
            present_kv: Updated KV cache (if use_cache=True)
        """
        batch, seq_len, _ = x.shape

        # Linear projections
        q = self.wq(x)  # [batch, seq_len, n_heads * head_dim]
        k = self.wk(x)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.wv(x)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape to multi-head format
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE (rotary positional embedding)
        q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle KV cache for autoregressive generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Repeat KV heads to match Q heads (core of GQA)
        k = repeat_kv(k, self.n_rep)  # [batch, n_heads, kv_seq_len, head_dim]
        v = repeat_kv(v, self.n_rep)  # [batch, n_heads, kv_seq_len, head_dim]

        # Scaled dot-product attention
        # Using F.scaled_dot_product_attention for efficiency (Flash Attention when available)
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,  # Causal mask for autoregressive models
        )

        # Reshape back: [batch, seq_len, n_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # Output projection
        output = self.wo(output)

        return output, present_kv
```

**Usage Example**

```python
# Configuration for GQA (12 Q heads, 4 KV heads)
config = ModelConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=4,  # Each KV head serves 3 Q heads
)

# Create attention module
attn = Attention(config)

# Input tensor
x = torch.randn(2, 128, 768)  # [batch=2, seq_len=128, hidden=768]

# RoPE frequencies (simplified)
freqs_cos = torch.ones(128, 64)
freqs_sin = torch.zeros(128, 64)

# Forward pass
output, kv_cache = attn(x, freqs_cos, freqs_sin, use_cache=True)
print(f"Output shape: {output.shape}")  # [2, 128, 768]
print(f"KV cache K shape: {kv_cache[0].shape}")  # [2, 4, 128, 64] - only 4 KV heads!
```

**Memory Comparison**

For the configuration above with sequence length 128:

| Method | KV Cache Shape | Memory (per layer, fp16) |
|--------|---------------|-------------------------|
| MHA | `[2, 12, 128, 64]` | 384 KB |
| GQA (g=4) | `[2, 4, 128, 64]` | 128 KB |
| MQA | `[2, 1, 128, 64]` | 32 KB |

GQA with 4 KV heads reduces memory to 1/3 of MHA while maintaining most of the model quality.
