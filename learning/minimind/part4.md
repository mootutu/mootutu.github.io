---
layout: learning-post-layout
title: "4 FFN Sub-layer (SwiGLU)"
date: 2025-12-27
lang: en
topic: minimind
order: 4
topic_url: /learning/minimind.html
translate_url: /cn/learning/minimind/part4.html
mathjax: true
---

The FFN (Feed-Forward Network) sub-layer follows a simple core idea: **expand dimension -> nonlinearity -> compress dimension**. It first unfolds features in a higher-dimensional space, applies gated nonlinearity, then projects back to the original dimension for the residual path.

## 4.1 Structure Overview / Architecture Diagram

The diagram shows a complete **FFN sub-layer (with normalization and residual)**. The overall structure is **Pre-Norm**:

$$
Output = x + \text{Dropout}(\text{FFN}(\text{RMSNorm}(x)))
$$

![FFN sub-layer overview](/assets/images/minimind/ffn-swiglu.png)

## 4.2 Module Breakdown: From Input to Output

### 4.2.1 Input and Normalization (RMSNorm)

- **Location**: the gray block at the bottom of the diagram.
- **Role**: normalize the input before FFN. Pre-Norm tends to stabilize training and improve gradient flow compared to Post-Norm.

### 4.2.2 Expansion Layer (Two Parallel Linear Projections)

- **Location**: two blue `Linear` blocks after RMSNorm.
- **Structure**: SwiGLU uses two parallel branches.
  - Left `Linear` (Up Projection): produces candidate features.
  - Right `Linear` (Gate Projection): produces gating signals.
- **Dimension change**: from $d_{model}$ to $d_{ff}$ (often $8/3$ or $4$ times).

### 4.2.3 Nonlinearity and Gating (SiLU + ⊙)

- **Location**: the yellow `SiLU` block and the `⊙` in the middle.
- **Key idea**: only the gate branch goes through the activation, then multiplies the candidate branch element-wise.

$$
SwiGLU(x) = (xW_g \cdot \text{SiLU}) \odot (xW_{up})
$$

- `⊙` is element-wise multiplication. The gate controls how much of the candidate features pass through.

### 4.2.4 Compression Layer (Down Projection)

- **Location**: the blue `Linear` above the multiplication.
- **Dimension change**: from $d_{ff}$ back to $d_{model}$.
- **Purpose**: compress rich features learned in high dimension back to the model dimension for residual addition.

### 4.2.5 Regularization and Residual Connection

- **Dropout**: randomly zeros activations during training to reduce overfitting.
- **Residual connection**:

$$
\,x_{out} = x_{in} + FFN(x_{in})
$$

The residual path lets the network learn only the change, easing gradient issues and enabling deeper stacks.

## 4.3 Why FFN Improves Expressiveness

### 4.3.1 A Single Linear Layer Is Not Enough

A single linear transform $y = Wx$ is limited to linear mappings. Stacking linear layers is still linear:

$$
W_2(W_1 x) = (W_2 W_1) x = W' x
$$

Nonlinearity is necessary to model complex relationships.

### 4.3.2 Why Expand Then Compress

| Direct transform | Expand -> compress |
|---------|----------|
| Nonlinearity in low dimension | Nonlinearity in high dimension |
| Limited feature mixing | Features can fully unfold and recombine |
| Like playing ping-pong in a small room | Like playing ping-pong in a stadium |

High-dimensional nonlinear transforms are more expressive. A simple analogy: expansion is like unfolding a folded sheet, making it easier to draw, then folding it back.

### 4.3.3 Why SwiGLU Over Vanilla FFN

| Vanilla FFN | SwiGLU |
|---------|--------|
| $\text{ReLU}(xW_1)W_2$ | $(\text{SiLU}(xW_g) \odot xW_{up})W_{down}$ |
| Single expansion path | Two expansion paths + gating |
| Hard truncation of negatives | Smooth gating |

Gating lets the model adaptively select features. In practice, SwiGLU performs better at similar parameter counts (as used in the LLaMA series).

## 4.4 Code Implementation

Implementation notes: follow the two-branch expansion with gate activation, then multiply and project back down. The gate branch is activated, the up branch is not.

```python
class FeedForward(nn.Module):
    def __init__(self, args: MootutuConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # 升维：两条并行路径
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        
        # 降维
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]  # 默认 SiLU

    def forward(self, x):
        # SwiGLU: gate 过激活函数，up 不过，然后相乘
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
```

Common pitfall: the gate and up branches must have the same dimension; otherwise the element-wise multiplication will fail.

## 4.5 Key Takeaways

| Component | Dimension change | Role |
|------|---------|------|
| `gate_proj` | hidden -> intermediate | generate gating signals |
| `up_proj` | hidden -> intermediate | generate candidate features |
| SiLU + multiplication | intermediate -> intermediate | nonlinearity + feature selection |
| `down_proj` | intermediate -> hidden | compress back to model dimension |

One-line summary: FFN uses **expand -> gated nonlinearity -> compress** to learn rich nonlinear features in high-dimensional space, then fold them back into the original dimension for residual integration.
