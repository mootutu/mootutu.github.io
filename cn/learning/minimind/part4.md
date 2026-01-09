---
layout: learning-post-layout
title: "4 FFN 子层（SwiGLU）"
date: 2025-12-27
lang: zh-CN
topic: minimind
order: 4
topic_url: /cn/learning/minimind.html
translate_url: /learning/minimind/part4.html
mathjax: true
---
FFN（Feed-Forward Network）子层的核心思路可以概括为：**升维 → 非线性 → 降维**。它让模型先在高维空间展开特征、通过门控非线性进行筛选，再压缩回原维度以便残差连接。

## 4.1 结构总览/架构图

图中展示了完整的 **FFN Sub-layer（含归一化与残差）**。整体采用 **Pre-Norm** 结构：

$$
Output = x + \text{Dropout}(\text{FFN}(\text{RMSNorm}(x)))
$$

![FFN Sub-layer 结构总览](/assets/images/minimind/ffn-swiglu.png)

## 4.2 模块分解：从输入到输出

### 4.2.1 输入与归一化（RMSNorm）

- **位置**：图中最下方的灰色模块。
- **作用**：进入 FFN 前先进行标准化。Pre-Norm 相比 Post-Norm 更利于梯度传播，训练更稳定。

### 4.2.2 升维层（双路 Linear）

- **位置**：RMSNorm 之后的两个并列蓝色模块。
- **结构**：SwiGLU 采用双路设计。
  - 左路 `Linear`（Up Projection）：生成候选特征。
  - 右路 `Linear`（Gate Projection）：生成门控信号。
- **维度变化**：从 $d_{model}$ 到 $d_{ff}$（常见为 $8/3$ 倍或 $4$ 倍）。

### 4.2.3 非线性激活与门控（SiLU + ⊙）

- **位置**：黄色的 `SiLU` 与中间的 `⊙`。
- **要点**：仅对门控分支做激活，随后与候选特征逐元素相乘。

$$
SwiGLU(x) = (xW_g \cdot \text{SiLU}) \odot (xW_{up})
$$

- `⊙` 表示逐元素乘法（Element-wise Multiplication），门控信号决定候选特征的通过/抑制程度。

### 4.2.4 降维层（Down Projection）

- **位置**：乘法操作上方的蓝色 `Linear`。
- **维度变化**：从 $d_{ff}$ 回到 $d_{model}$。
- **目的**：将高维中抽取到的复杂特征压回原维度，便于与残差相加。

### 4.2.5 正则化与残差连接

- **Dropout**：训练时随机置零部分神经元，降低过拟合风险。
- **Residual Connection**：

$$
\,x_{out} = x_{in} + FFN(x_{in})
$$

残差让网络学习“增量变化”，缓解梯度消失，训练更深层结构时更稳定。

## 4.3 为什么 FFN 能提升表达能力

### 4.3.1 线性层不足以建模非线性

单层线性变换 $y = Wx$ 只能做线性映射，多层线性叠加依然等价于一个线性变换：

$$
W_2(W_1 x) = (W_2 W_1) x = W' x
$$

要打破这个限制，必须引入非线性激活。

### 4.3.2 为什么要先升维再降维

| 直接变换 | 升维→降维 |
|---------|----------|
| 在低维空间做非线性 | 在高维空间做非线性 |
| 特征混合受限 | 特征可以充分展开、重组 |
| 类似在小房间里打乒乓球 | 类似在大体育馆里打乒乓球 |

高维空间里的非线性变换更有表达力。直观类比：升维像“展开”折叠的纸，展开后更容易画图，再折回去。

### 4.3.3 SwiGLU 相比原版 FFN 的优势

| 原版 FFN | SwiGLU |
|---------|--------|
| $\text{ReLU}(xW_1)W_2$ | $(\text{SiLU}(xW_g) \odot xW_{up})W_{down}$ |
| 一条升维路径 | 两条升维路径 + 门控 |
| 硬截断负值 | 平滑门控 |

门控机制允许模型自适应选择要激活的特征，实践中在同等参数量下效果更好（如 LLaMA 系列采用）。

## 4.4 代码实现

实现要点：按 SwiGLU 的双路升维与门控相乘，再做降维；gate 经过激活，up 不经过。

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

常见坑：门控分支和候选分支的维度必须一致，否则逐元素乘法会出错。

## 4.5 关键要点总结

| 组件 | 维度变化 | 作用 |
|------|---------|------|
| `gate_proj` | hidden → intermediate | 生成门控信号 |
| `up_proj` | hidden → intermediate | 生成候选特征 |
| SiLU + 乘法 | intermediate → intermediate | 非线性 + 特征选择 |
| `down_proj` | intermediate → hidden | 信息压缩回原维度 |

一句话总结：FFN 通过“升维→门控激活→降维”的结构，让模型在高维空间学习复杂的非线性特征，再压回原维度并与残差融合，从而提升 Transformer 的表达能力。
