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
lang: en
translate_url: /cn/learning/your-topic.html
---

<div class="learning-header">
    <h1 class="learning-title">Your Topic Name 🔥</h1>
    <p class="learning-subtitle">
        主题描述文字，说明这个学习主题包含什么内容。
    </p>
</div>

<div class="learning-entries">
    <a href="/learning/your-topic/article-1.html" class="entry-link">Article 1 Title</a>
    <a href="/learning/your-topic/article-2.html" class="entry-link">Article 2 Title</a>
    <a href="/learning/your-topic/article-3.html" class="entry-link">Article 3 Title</a>
</div>
```

---

## 博客索引卡片模板

### 文件位置

添加到：`blog.html` 的 `<div class="blog-posts">` 内

### 完整模板

```html
<!-- 新文章 - 添加到列表最前面 -->
<article class="blog-post">
  <div class="blog-post-content">
    <h2 class="blog-post-title">
      <a href="/blog/posts/your-article-name.html">Your Article Title</a>
    </h2>
    <div class="blog-post-meta">
      <span>Published: December 31, 2025</span>
      <span>Reading Time: 8 minutes</span>
    </div>
    <p class="blog-post-excerpt">
      文章摘要，1-2 句话概括文章内容。这段文字会显示在博客列表页面。
    </p>
    <div class="blog-post-tags">
      <a href="#" class="blog-tag">Tag1</a>
      <a href="#" class="blog-tag">Tag2</a>
      <a href="#" class="blog-tag">Tag3</a>
    </div>
    <a href="/blog/posts/your-article-name.html" class="read-more">Read More</a>
  </div>
  <div class="blog-post-image">
    <img src="/assets/images/covers/your-article-name.png" alt="Your Article Title">
  </div>
</article>
```

---

## 学习主题卡片模板

### 文件位置

添加到：`learning.html` 的 `<div class="learning-topics">` 内

### 完整模板

```html
<a href="/learning/your-topic.html" class="topic-card">
    <div class="topic-title">Your Topic Name 🔥</div>
    <div class="topic-desc">主题描述，说明这个主题包含什么内容。</div>
</a>
```

---

## 出版物条目模板

### 文件位置

添加到：`index.html` 的 Publications 部分

### 完整模板

```html
<!-- 新出版物 -->
<div class="publication">
    <div class="pub-title">
        <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
            Paper Title: A Comprehensive Study
        </a>
    </div>
    <div class="pub-authors">
        <strong>Weiqin Wang</strong>, Coauthor One, Coauthor Two
    </div>
    <div class="pub-venue">
        Conference Name (CONF), 2025
    </div>
    <div class="pub-links">
        <a href="https://arxiv.org/pdf/xxxx.xxxxx.pdf" target="_blank">[PDF]</a>
        <a href="javascript:void(0)" onclick="showBibtex('wang2025paper')">[bib]</a>
    </div>
</div>

<!-- BibTeX 内容（添加到页面底部的 bibtex 区域） -->
<div id="wang2025paper" class="bibtex-content" style="display:none;">
    <pre>@inproceedings{wang2025paper,
  title={Paper Title: A Comprehensive Study},
  author={Wang, Weiqin and One, Coauthor and Two, Coauthor},
  booktitle={Conference Name},
  year={2025}
}</pre>
</div>
```

---

## 使用说明

1. 复制对应模板
2. 替换占位文本（`your-article-name`、`Your Article Title` 等）
3. 根据实际内容修改
4. 保存并预览

---

[返回目录](../README.md) | [速查表 →](./cheatsheet.md)
