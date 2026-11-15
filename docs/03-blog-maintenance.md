# 博客文章维护

## 3.1 概述

博客文章（Blog Post）用来记录工具使用、技术教程、实践经验这类独立话题。

| 特性 | Blog Post | Learning Post |
|------|-----------|---------------|
| 组织方式 | 按日期倒序的卡片列表 | 按主题分组、按 `order` 排序 |
| 适合 | 独立话题、教程 | 系列化学习笔记 |
| 格式 | HTML | Markdown（推荐） |
| 封面图 | 需要 | 不需要 |

---

## 3.2 存放位置

```
blog/posts/
├── automating-tasks-python.html
├── config-my-vscode.html
├── git-basics.html
├── markdown-guide.html
└── modernizing-python.html

assets/images/covers/      # 封面图，文件名与文章同名
assets/images/posts/       # 文章内配图，每篇一个子目录
```

文章的线上地址是 `/blog/posts/文件名.html`。

---

## 3.3 文件命名

- 全部小写，单词之间用连字符 `-`
- 扩展名 `.html`
- 简短、能看出主题

| 正确 | 错误 |
|------|------|
| `git-basics.html` | `Git_Basics.html` |
| `my-first-post.html` | `MyFirstPost.HTML` |

---

## 3.4 Front Matter 字段

```yaml
---
layout: blog-post-layout
title: "Git Basics for Developers"
date: 2025-11-28
author: "Weiqin Wang"
category: "Technical Tutorial"
excerpt: "Git helps developers manage code changes and collaborate effectively."
tags: ["Git", "Version Control", "Development"]
reading_time: "8 minutes"
cover_image: /assets/images/covers/git-basics.png
lang: en
---
```

| 字段 | 必填 | 用在哪里 |
|------|------|---------|
| `layout` | 是 | 固定为 `blog-post-layout` |
| `title` | 是 | 文章大标题、博客卡片标题、浏览器标签页、搜索 |
| `date` | 是 | 排序依据；卡片和文章页显示，格式 `YYYY-MM-DD` |
| `author` | 否 | 文章页作者，缺省为 `Weiqin Wang` |
| `category` | 是 | 文章页标题上方的小字、卡片元信息 |
| `excerpt` | 是 | 卡片摘要、文章页标题下的引言、页面 description |
| `tags` | 是 | 卡片底部最多显示 3 个；搜索会匹配 |
| `reading_time` | 是 | 只用于博客卡片（`minutes` 会显示成 `min`） |
| `cover_image` | 是 | 博客卡片封面 |
| `lang` | 是 | 固定为 `en` |

文章页里显示的阅读时长是按正文字数自动计算的（200 词/分钟），和 `reading_time` 无关。

常用 `category`：`Technical Tutorial`、`Technical Writing`、`Tool Usage`、`Learning Notes`。

---

## 3.5 新增博客文章

1. **复制模板**

   ```bash
   cp _templates/blog_post_template.html blog/posts/your-post.html
   ```

2. **填写 Front Matter**（见上一节）

3. **写正文**：在第二个 `---` 下面写 HTML。

   ```html
   <p>引言段落……</p>

   <h2>1. First Section</h2>
   <p>正文……</p>

   <h3>1.1 Subsection</h3>
   <p>更多内容……</p>
   ```

   - 用 `<h2>`、`<h3>` 分节，右侧目录会自动生成（至少两个标题才显示）
   - 标题不需要手写 `id`，脚本会自动补；目录里会去掉开头的 `1.`、`1.1` 这类编号
   - 不要再写自定义 `<style>`，正文样式统一由 `.prose` 控制

4. **代码块**

   ```html
   <pre><code class="language-python">def hello():
       print("Hello, World!")
   </code></pre>
   ```

   已加载的语言：`python`、`bash`、`toml`、`json`、`yaml`，以及 Prism 自带的 `markup`（HTML）、`css`、`javascript`。需要其他语言见 [10-特殊功能](./10-features.md#105-代码高亮)。

5. **配图**

   ```html
   <img src="/assets/images/posts/your-post/image.png" alt="描述">
   ```

   图片会自动居中、圆角、不超过正文宽度。

6. **文末标签（可选）**：现有文章在正文末尾放了一组标签，样式已经适配：

   ```html
   <div class="blog-tags">
     <a href="#" class="blog-tag">Git</a>
     <a href="#" class="blog-tag">Version Control</a>
   </div>
   ```

7. **封面图**：放到 `assets/images/covers/your-post.png`，要求见 [3.8](#38-封面图规范)。

8. **本地预览**：打开 `http://localhost:4000/blog.html` 确认卡片出现，再点进文章检查排版、目录、代码高亮，并切换一次深色模式。

9. **提交**

   ```bash
   git add blog/posts/your-post.html assets/images/covers/your-post.png
   git commit -m "feat: add blog post about X"
   git push
   ```

---

## 3.6 修改已有文章

- 改正文：直接编辑 `blog/posts/xxx.html`
- 改标题、日期、摘要、封面：改 Front Matter；博客列表会自动更新
- 换配图：替换 `assets/images/posts/xxx/` 下的文件，保持文件名不变就不用改正文

---

## 3.7 删除文章

```bash
git rm blog/posts/old-post.html
git rm assets/images/covers/old-post.png
git rm -r assets/images/posts/old-post/   # 如果有
git commit -m "chore: remove blog post 'old post'"
```

删除后博客列表和搜索索引都会自动更新。

---

## 3.8 封面图规范

- 位置：`assets/images/covers/`，文件名与文章同名（`git-basics.html` → `git-basics.png`）
- 显示比例：**16:9**，卡片会按 16:9 裁切（`object-fit: cover`），主体请放在画面中间
- 推荐尺寸：1600×900 或 1200×675
- 格式：PNG 或 JPG，尽量压缩到 500 KB 以内

---

## 3.9 文章内图片规范

```
assets/images/posts/
└── your-post/
    ├── image.png
    ├── image-1.png
    └── terminal-output.png
```

- 每篇文章一个与文件同名的目录
- 默认命名 `image.png`、`image-1.png`……，也可以用描述性名称
- 一律使用以 `/` 开头的绝对路径引用

---

<div class="doc-nav">
  <a href="./02-quickstart.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 快速开始</span>
  </a>
  <a href="./04-learning-maintenance.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">学习内容维护 →</span>
  </a>
</div>
