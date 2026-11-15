# 学习内容维护

## 4.1 概述

学习区分两层：

- **Learning Topic（学习主题）**：一个主题一个页面，比如 Minimind、Python Guidelines、Paper Reading。页面会自动列出属于这个主题的全部笔记。
- **Learning Post（学习笔记）**：属于某个主题的单篇笔记，通常用 Markdown 写。

`learning.html` 是学习区首页，自动把所有主题显示成编号卡片，并显示每个主题的笔记数量。

论文阅读库（Paper Reading）是一个特殊主题，笔记字段更多，页面带搜索和筛选，单独在 [4.5](#45-论文阅读笔记) 说明。

---

## 4.2 目录结构

```
learning.html                       # 学习区首页（主题卡片）
learning/
├── minimind.html                   # 主题页：topic: minimind
├── minimind/                       # 该主题的笔记
│   ├── day1.html
│   ├── part2.md
│   ├── part3.md
│   └── part4.md
├── python.html                     # 主题页：topic: python
├── python_guidelines/
│   ├── python_language_conventions.md
│   └── python_style_conventions.md
├── paper-reading.html              # 论文阅读库：topic: paper-reading
└── paper-reading/
    ├── README.md                   # 目录说明（不是笔记）
    └── 2603.25450-cross-model-disagreement.md
```

笔记归属哪个主题只看 Front Matter 里的 `topic`，跟所在目录名无关（例如 `python_guidelines/` 下的笔记写的是 `topic: python`）。不过为了好找，请把笔记放在对应主题的目录里。

---

## 4.3 学习主题

### 主题页结构

```html
---
layout: learning-topic-layout
title: "Minimind Learning"
description: "Recording the process of building a large model (minimind) from scratch."
topic: minimind
order: 1
lang: en
---
{% include learning_entries.html %}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `layout` | 是 | 固定为 `learning-topic-layout` |
| `title` | 是 | 页头大标题，也是学习区首页卡片标题 |
| `description` | 是 | 页头引言，也是卡片描述 |
| `topic` | 是 | 主题键，笔记通过它归属到这个主题 |
| `order` | 是 | 在学习区首页的排序（从小到大） |
| `lang` | 是 | 固定为 `en` |
| `eyebrow` | 否 | 标题上方的小字，缺省为 `Learning topic` |
| `illo` | 否 | 页头插画，缺省为 `learning`（见 [10.8](./10-features.md#108-插画)） |
| `hide_back` | 否 | 设为 `true` 时不显示“All topics”返回链接 |

页头（返回链接、小字、标题、引言、插画）由布局自动生成，主题页正文只需要 `{% include learning_entries.html %}`。标题里不要加 emoji，以保持风格统一。

### 新增学习主题

1. 新建主题页 `learning/your-topic.html`，按上面的结构填写
2. 新建笔记目录 `learning/your-topic/`
3. 本地打开 `/learning.html`，确认新卡片出现、笔记数量正确

---

## 4.4 新增学习笔记

### Front Matter

```yaml
---
layout: learning-post-layout
title: "2 Positional Encoding RoPE&YaRN"
date: 2025-12-26
lang: en
topic: minimind
order: 2
topic_url: /learning/minimind.html
mathjax: true
---
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `layout` | 是 | 固定为 `learning-post-layout` |
| `title` | 是 | 笔记标题 |
| `date` | 是 | 显示在笔记页和主题页列表中 |
| `lang` | 是 | 固定为 `en`（主题页只列出 `lang` 相同的笔记） |
| `topic` | 是 | 与主题页的 `topic` 一致 |
| `order` | 是 | 在主题页中的顺序 |
| `topic_url` | 是 | 返回链接指向的主题页；不填会回到 `/learning.html` |
| `mathjax` | 否 | 用到公式时设为 `true` |
| `category` | 否 | 标题上方的小字，缺省为 `Notes` |
| `excerpt` | 否 | 标题下方的引言 |

### 步骤

1. 复制模板

   ```bash
   cp _templates/learning_post_template.md learning/minimind/part5.md
   ```

2. 填写 Front Matter，用 Markdown 写正文
3. 本地打开对应主题页，确认新笔记出现在正确的位置，再打开笔记检查目录、公式、代码块

### 修改和删除

- 修改：直接编辑文件；改 `order` 可以调整在主题页中的顺序
- 删除：`git rm` 对应文件即可，主题页和学习区首页的数量会自动更新

---

## 4.5 论文阅读笔记

论文阅读库 `learning/paper-reading.html` 会列出所有 `topic: paper-reading` 的笔记（按 `date` 倒序），并提供：

- 搜索框：匹配标题、作者、标签、方法、数据集、模型、一句话总结和正文
- 筛选：阅读状态 `status`、相关度 `relevance`

### 新增论文笔记

```bash
cp _templates/paper_reading_template.md learning/paper-reading/2603.12345-short-title.md
```

文件名建议用 `arXiv编号-简短标题.md`，没有 arXiv 编号时用 `年份-第一作者-关键词.md`。

### 专用字段

| 字段 | 说明 | 可选值 / 示例 |
|------|------|--------------|
| `paper_year` | 论文年份 | `2026` |
| `authors` | 作者列表 | YAML 列表 |
| `arxiv` | arXiv 编号 | `"2603.25450"` |
| `doi` | DOI | 可留空 |
| `paper_url` | 论文链接，笔记页会显示 “Paper · arXiv:…” 链接 | URL |
| `status` | 阅读状态 | `unread` / `skimming` / `read` / `reproduced` / `integrated` |
| `relevance` | 与自己研究的相关度；`high` 时卡片上会多一个橙色标记 | `high` / `medium` / `low` |
| `tags` | 标签（尽量复用已有标签） | YAML 列表 |
| `methods` / `datasets` / `models` | 方法、数据集、模型，参与搜索 | YAML 列表 |
| `one_line_summary` | 一句话总结，显示在卡片和笔记页标题下 | 字符串 |

`status` 和 `relevance` 必须使用上表中的小写值，否则筛选不到。

论文阅读库页面底部还有一段 “Maintenance workflow”，是给自己看的维护步骤，修改时直接编辑 `learning/paper-reading.html`。

---

## 4.6 Markdown 写作约定

- 用 `##` 分大节、`###` 分小节，右侧目录基于这两级生成；`####` 不进目录
- 代码块用三个反引号并写语言名，例如 ` ```python `
- 提示、引用用 `>`，会显示为左侧橙色竖线的浅色块
- 表格会在窄屏上横向滚动
- 站内链接写 `.html` 后缀，例如 `[Part 3](/learning/minimind/part3.html)`

---

## 4.7 数学公式

在 Front Matter 里加 `mathjax: true` 后即可使用：

```markdown
行内公式：$\cos\theta$
独立公式：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
```

过宽的公式会在自己的范围内横向滚动，不会撑宽整页。

---

## 4.8 现有主题

| 主题 | 主题页 | `topic` | 笔记目录 |
|------|--------|---------|---------|
| Minimind Learning | `learning/minimind.html` | `minimind` | `learning/minimind/` |
| Python Guidelines | `learning/python.html` | `python` | `learning/python_guidelines/` |
| Paper Reading Library | `learning/paper-reading.html` | `paper-reading` | `learning/paper-reading/` |

---

<div class="doc-nav">
  <a href="./03-blog-maintenance.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 博客文章维护</span>
  </a>
  <a href="./05-bilingual.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">站点语言与界面文字 →</span>
  </a>
</div>
