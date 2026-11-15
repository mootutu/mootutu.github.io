# 模板文件

新建内容时优先复制 `_templates/` 里的模板文件。下面列出每种模板的要点，便于不打开文件也能写对。

---

## 博客文章

复制：`_templates/blog_post_template.html` → `blog/posts/your-post.html`

```html
---
layout: blog-post-layout
title: "Your Post Title"
date: 2026-01-01
author: "Weiqin Wang"
category: "Technical Tutorial"
excerpt: "One sentence shown on the blog card and under the title."
tags: ["Tag1", "Tag2"]
reading_time: "6 minutes"
cover_image: /assets/images/covers/your-post.png
lang: en
---

<p>Introduction paragraph.</p>

<h2>1. First Section</h2>
<p>Content.</p>

<pre><code class="language-python">print("hello")
</code></pre>

<img src="/assets/images/posts/your-post/image.png" alt="Description">

<h2>2. Second Section</h2>
<p>Content.</p>

<div class="blog-tags">
  <a href="#" class="blog-tag">Tag1</a>
  <a href="#" class="blog-tag">Tag2</a>
</div>
```

模板文件里的 `<!-- How to add a new Blog Post … -->` 注释可以保留，不会显示在页面上。

---

## 学习笔记（Markdown）

复制：`_templates/learning_post_template.md` → `learning/{主题目录}/your-note.md`

~~~~markdown
---
layout: learning-post-layout
title: "Learning Post Title"
date: 2026-01-01
lang: en
topic: minimind
order: 5
topic_url: /learning/minimind.html
mathjax: true
---

## Section 1

Text with inline math $\cos\theta$.

$$
y = \mathrm{softmax}(Wx + b)
$$

## Section 2

```python
def f(x):
    return x * 2
```
~~~~

---

## 论文阅读笔记

复制：`_templates/paper_reading_template.md` → `learning/paper-reading/2603.12345-short-title.md`

```markdown
---
layout: learning-post-layout
title: "Paper Title"
date: 2026-01-01
lang: en
topic: paper-reading
order: 999
topic_url: /learning/paper-reading.html
mathjax: true
paper_year: 2026
authors:
  - Author Name
arxiv: ""
doi: ""
paper_url: ""
status: unread          # unread / skimming / read / reproduced / integrated
relevance: medium       # high / medium / low
tags:
  - reasoning
methods: []
datasets: []
models: []
one_line_summary: ""
---
```

正文按模板中的小节填写：One-sentence summary、Research question、Method、Key findings（每条写 Claim / Evidence / Condition / Confidence）、My insights 等。

---

## 学习主题页

新建：`learning/your-topic.html`

```html
---
layout: learning-topic-layout
title: "Your Topic"
description: "One or two sentences describing what this topic collects."
topic: your-topic
order: 4
lang: en
---
{% include learning_entries.html %}
```

---

## 包装布局（新增栏目时）

新建：`_layouts/your-section-layout.html`

```html
---
layout: page          # 列表/介绍页用 page，文章页用 post
eyebrow: Section
illo: learning        # 对应 _includes/illo/learning.html，可省略
back_url: /
back_label: Home
---
{{ content }}
```

---

## 首页论文条目

加到 `_data/publications.yml` 最上方：

```yaml
- id: wang2026example
  title: "Paper Title."
  authors: "<b>Weiqin Wang</b>, Coauthor, and Hui Huang."
  venue: "In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (<b>EMNLP</b>), 2026"
  links:
    - label: pdf
      url: https://aclanthology.org/xxxx.pdf
    - label: code
      url: https://github.com/szu-tera/xxxx
  bib: /assets/bibtex/wang2026example.bib
```

预印本：`venue: "arXiv Preprint, 2026"`（没有 `<b>`，标签显示为 Preprint）。

---

## 首页经历 / 奖项行

```html
<li class="row" data-reveal>
  <span class="row__aside">2026</span>
  <div>
    <div class="row__title">Title</div>
    <div class="row__detail">Detail</div>
  </div>
</li>
```

---

<div class="doc-nav">
  <a href="../14-faq.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 常见问题FAQ</span>
  </a>
  <a href="./cheatsheet.md" class="doc-nav-card next">
    <span class="doc-nav-label">附录</span>
    <span class="doc-nav-title">速查表 →</span>
  </a>
</div>
