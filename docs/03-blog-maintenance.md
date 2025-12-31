# 博客文章维护

## 3.1 概述与使用场景

### 什么是 Blog Post

Blog Post 是博客文章，用于记录：
- 工具使用教程
- 技术学习心得
- 项目实践经验
- 独立的技术话题

### 与 Learning Post 的区别

| 特性 | Blog Post | Learning Post |
|------|-----------|---------------|
| 组织方式 | 扁平列表，按时间排序 | 按主题分组 |
| 适用场景 | 独立话题、教程 | 系统化学习笔记 |
| 文件格式 | HTML | Markdown |
| 需要封面图 | 是 | 否 |

---

## 3.2 目录结构与存放位置

```
blog/
└── posts/                    # 英文博客文章
    ├── modernizing-python.html
    ├── automating-tasks-python.html
    ├── git-basics.html
    └── markdown-guide.html

cn/blog/
└── posts/                    # 中文博客文章
    ├── modernizing-python.html
    ├── automating-tasks-python.html
    ├── git-basics.html
    └── markdown-guide.html
```

---

## 3.3 文件命名规范

### 规则

- 使用**小写字母**
- 单词之间用**连字符 `-`** 分隔
- 扩展名为 `.html`
- 名称应简洁、有描述性

### 示例

| 正确 | 错误 |
|------|------|
| `git-basics.html` | `Git_Basics.html` |
| `modernizing-python.html` | `modernizing python.html` |
| `my-first-post.html` | `MyFirstPost.HTML` |

---

## 3.4 Front Matter 字段说明

每篇博客文章的开头必须包含 Front Matter：

```yaml
---
layout: blog-post-layout
title: "文章标题"
date: 2025-12-31
author: "Weiqin Wang"
category: "Technical Tutorial"
lang: en
translate_url: /cn/blog/posts/article-name.html
---
```

### 字段详解

| 字段 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `layout` | 是 | 必须为 `blog-post-layout` | `blog-post-layout` |
| `title` | 是 | 文章标题，显示在页面顶部 | `"Git Basics for Developers"` |
| `date` | 是 | 发布日期，格式 `YYYY-MM-DD` | `2025-12-31` |
| `author` | 是 | 作者名称 | `"Weiqin Wang"` |
| `category` | 是 | 文章分类 | `"Technical Tutorial"` |
| `lang` | 是 | 语言代码：`en` 或 `zh-CN` | `en` |
| `translate_url` | 是 | 对应翻译版本的 URL | `/cn/blog/posts/git-basics.html` |

### 常用 Category 值

- `Technical Tutorial` / `技术教程`
- `Tool Usage` / `工具使用`
- `Development` / `开发实践`
- `Learning Notes` / `学习笔记`

---

## 3.5 新增博客文章完整流程

### 步骤 1：创建文章文件

```bash
# 复制模板
cp _templates/blog_post_template.html blog/posts/your-article-name.html
```

### 步骤 2：编辑 Front Matter

打开文件，修改头部的 Front Matter：

```yaml
---
layout: blog-post-layout
title: "Your Article Title"
date: 2025-12-31
author: "Weiqin Wang"
category: "Technical Tutorial"
lang: en
translate_url: /cn/blog/posts/your-article-name.html
---
```

### 步骤 3：编写文章内容

在 `<div class="blog-content">` 内编写 HTML 内容：

```html
<div class="blog-content">
    <p>文章引言段落...</p>

    <h2 id="section-1">1. 第一部分</h2>
    <p>正文内容...</p>

    <h3 id="section-1-1">1.1 子标题</h3>
    <p>更多内容...</p>

    <h2 id="section-2">2. 第二部分</h2>
    <p>继续...</p>
</div>
```

**注意事项：**
- 使用 `<h2>` 和 `<h3>` 创建标题（会自动生成侧边目录）
- 每个标题必须有唯一的 `id` 属性
- 代码块使用 `<pre><code class="language-xxx">` 格式

### 步骤 4：添加代码块（如需要）

```html
<pre><code class="language-python">def hello():
    print("Hello, World!")
</code></pre>
```

支持的语言：`python`、`bash`、`javascript`、`json`、`toml`、`yaml`、`html`、`css`

### 步骤 5：添加图片（如需要）

```html
<!-- 行内图片 -->
<figure class="image">
    <img src="/assets/images/posts/your-article-name/image.png" alt="描述">
    <figcaption>图片说明文字</figcaption>
</figure>
```

图片存放位置：`assets/images/posts/your-article-name/`

### 步骤 6：添加封面图

准备一张封面图片：
- 推荐尺寸：正方形（如 400x400 或 600x600）
- 格式：PNG 或 JPG
- 保存位置：`assets/images/covers/your-article-name.png`

**命名必须与文章文件名一致！**

### 步骤 7：更新博客索引页（重要！）

打开 `blog.html`，在 `<div class="blog-posts">` 下面添加文章卡片：

```html
<!-- 新文章 -->
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
      文章简短描述，1-2 句话概括文章内容...
    </p>
    <div class="blog-post-tags">
      <a href="#" class="blog-tag">Tag1</a>
      <a href="#" class="blog-tag">Tag2</a>
    </div>
    <a href="/blog/posts/your-article-name.html" class="read-more">Read More</a>
  </div>
  <div class="blog-post-image">
    <img src="/assets/images/covers/your-article-name.png" alt="Your Article Title">
  </div>
</article>
```

> ⚠️ **注意**：新文章应添加到列表的**最前面**（最新的在上面）

### 步骤 8：创建中文版本

重复步骤 1-7，但：
- 文件放入 `cn/blog/posts/` 目录
- 修改 `lang: zh-CN`
- 修改 `translate_url` 指向英文版本
- 更新 `cn/blog.html`

### 步骤 9：本地预览

```bash
jekyll serve
# 访问 http://localhost:4000/blog/posts/your-article-name.html
```

### 步骤 10：提交发布

```bash
git add .
git commit -m "feat: Add blog post 'Your Article Title'"
git push
```

---

## 3.6 修改已有博客文章

### 修改文章内容

1. 直接编辑 `blog/posts/xxx.html` 文件
2. 同步修改中文版本 `cn/blog/posts/xxx.html`（如有）
3. 本地预览确认
4. 提交推送

### 修改文章元信息

如需修改标题、日期、分类等：

1. 修改文章的 Front Matter
2. 同步修改 `blog.html` 中的卡片信息
3. 同步修改中文版本

### 更新文章图片

1. 替换 `assets/images/posts/xxx/` 目录下的图片
2. 确保新图片名称与引用路径一致

---

## 3.7 删除博客文章

1. 删除文章文件
   ```bash
   rm blog/posts/article-to-delete.html
   rm cn/blog/posts/article-to-delete.html
   ```

2. 删除封面图
   ```bash
   rm assets/images/covers/article-to-delete.png
   ```

3. 删除文章内图片（如有）
   ```bash
   rm -rf assets/images/posts/article-to-delete/
   ```

4. 从 `blog.html` 和 `cn/blog.html` 中删除对应的卡片

5. 提交更改
   ```bash
   git add .
   git commit -m "chore: Remove blog post 'xxx'"
   git push
   ```

---

## 3.8 博客索引页维护（blog.html）

### 文件位置

- 英文：`/blog.html`
- 中文：`/cn/blog.html`

### 结构说明

```html
<div class="blog-posts">
  <!-- 文章卡片从这里开始，最新的在最上面 -->
  <article class="blog-post">...</article>
  <article class="blog-post">...</article>
  <article class="blog-post">...</article>
</div>
```

### 卡片模板

完整的文章卡片模板见 [附录-模板文件](./appendix/模板文件.md#博客索引卡片模板)

---

## 3.9 封面图规范

### 存放位置

```
assets/images/covers/
├── modernizing-python.png
├── git-basics.png
├── automating-tasks-python.png
└── markdown-guide.png
```

### 命名规则

封面图文件名必须与文章文件名一致（不含扩展名）：

| 文章文件 | 封面图文件 |
|----------|-----------|
| `git-basics.html` | `git-basics.png` |
| `my-new-post.html` | `my-new-post.png` |

### 尺寸建议

- 推荐比例：1:1（正方形）
- 推荐尺寸：400x400 ~ 800x800 像素
- 格式：PNG 或 JPG

---

## 3.10 文章内图片规范

### 存放位置

```
assets/images/posts/
└── your-article-name/      # 与文章同名的目录
    ├── image.png           # 第一张图
    ├── image-1.png         # 第二张图
    ├── image-2.png         # 第三张图
    └── ...
```

### 命名规则

- 默认命名：`image.png`、`image-1.png`、`image-2.png`...
- 也可使用描述性名称：`vscode-config.png`、`terminal-output.png`

### 引用方式

```html
<img src="/assets/images/posts/your-article-name/image.png" alt="描述">
```

---

[← 上一篇：快速开始](./02-快速开始.md) | [返回目录](./README.md) | [下一篇：学习内容维护 →](./04-学习内容维护.md)
