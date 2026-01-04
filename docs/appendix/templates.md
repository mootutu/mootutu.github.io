# 模板文件

本文档提供各类内容的完整模板，可直接复制使用。

---

## 博客文章模板

### 文件位置

复制到：`blog/posts/your-article-name.html`

### 完整模板

```html
---
layout: blog-post-layout
title: "Your Article Title Here"
date: 2025-12-31
author: "Weiqin Wang"
category: "Technical Tutorial"
excerpt: "Short summary shown on the blog index."
tags: ["Tag1", "Tag2"]
reading_time: "6 minutes"
cover_image: /assets/images/covers/your-article-name.png
lang: en
translate_url: /cn/blog/posts/your-article-name.html
---

<div class="blog-content">
    <p>
        文章引言段落，简要介绍本文主题和内容概要。
    </p>

    <h2 id="section-1">1. 第一部分</h2>
    <p>
        第一部分的正文内容...
    </p>

    <h3 id="section-1-1">1.1 子标题</h3>
    <p>
        子标题下的内容...
    </p>

    <!-- 代码块示例 -->
    <pre><code class="language-python">def hello():
    print("Hello, World!")
</code></pre>

    <!-- 图片示例 -->
    <figure class="image">
        <img src="/assets/images/posts/your-article-name/image.png" alt="图片描述">
        <figcaption>图片说明文字</figcaption>
    </figure>

    <h2 id="section-2">2. 第二部分</h2>
    <p>
        第二部分的内容...
    </p>

    <!-- 列表示例 -->
    <ul>
        <li>列表项 1</li>
        <li>列表项 2</li>
        <li>列表项 3</li>
    </ul>

    <h2 id="conclusion">总结</h2>
    <p>
        文章总结内容...
    </p>
</div>
```

---

## 学习文章模板（Markdown）

### 文件位置

复制到：`learning/topic-name/your-article.md`

### 完整模板

```markdown
---
layout: learning-post-layout
title: "Your Article Title Here"
date: 2025-12-31
lang: en
topic: topic-name
order: 1
topic_url: /learning/topic-name.html
translate_url: /cn/learning/topic-name/your-article.html
mathjax: false
---

## 引言

本文介绍...

## 第一部分

正文内容...

### 1.1 子标题

子标题内容...

```python
# 代码示例
def hello():
    print("Hello!")
```

## 第二部分

更多内容...

> 💡 **提示**：这是一个提示框

## 总结

总结内容...
```

---

## 学习主题索引页模板

### 文件位置

复制到：`learning/your-topic.html`

### 完整模板

```html
---
layout: learning-topic-layout
title: "Your Topic Name 🔥"
description: "主题描述文字，说明这个学习主题包含什么内容。"
topic: your-topic
order: 3
lang: en
translate_url: /cn/learning/your-topic.html
---
```

---

## 博客索引卡片模板

### 说明

博客索引页自动生成，无需手动添加卡片。请在文章 Front Matter 中补全卡片字段：

```yaml
excerpt: "文章摘要，1-2 句话概括文章内容。"
tags: ["Tag1", "Tag2", "Tag3"]
reading_time: "8 minutes"
cover_image: /assets/images/covers/your-article-name.png
```

---

## 学习主题卡片模板

### 说明

学习主题卡片由主题页 Front Matter 自动生成，无需手动添加。

---

## 出版物条目模板

### 文件位置

添加到：`_data/publications.yml`

### 完整模板

```yaml
- id: wang2025paper
  title:
    en: "Paper Title: A Comprehensive Study"
    zh: "Paper Title: A Comprehensive Study"
  authors:
    en: "<b>Weiqin Wang</b>, Coauthor One, Coauthor Two"
    zh: "<b>王伟钦</b>, 合作者一, 合作者二"
  venue:
    en: "Conference Name (CONF), 2025"
    zh: "Conference Name (CONF), 2025"
  links:
    - label: pdf
      url: https://arxiv.org/pdf/xxxx.xxxxx.pdf
  bib: /assets/bibtex/wang2025paper.bib
```

---

## 使用说明

1. 复制对应模板
2. 替换占位文本（`your-article-name`、`Your Article Title` 等）
3. 根据实际内容修改
4. 保存并预览

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
