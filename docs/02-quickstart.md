# 快速开始

## 2.1 本地开发环境搭建

### 前置要求

- **Ruby** 2.5.0 或更高版本
- **Bundler**（Ruby 包管理器）
- **Git**

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/mootutu/mootutu.github.io.git
cd mootutu.github.io

# 2. 安装 Ruby 依赖（如果有 Gemfile）
bundle install

# 3. 启动本地服务器
bundle exec jekyll serve

# 或者直接使用（如果没有 Gemfile）
jekyll serve
```

### 访问地址

本地服务器启动后，访问：

```
http://localhost:4000
```

### 常用开发命令

| 命令 | 说明 |
|------|------|
| `jekyll serve` | 启动本地服务器（自动刷新） |
| `jekyll serve --drafts` | 启动服务器并显示草稿 |
| `jekyll build` | 构建站点到 `_site/` 目录 |
| `jekyll clean` | 清理构建缓存 |

---

## 2.2 常用操作速查

### 新增博客文章（5 步）

```bash
# 1. 复制模板
cp _templates/blog_post_template.html blog/posts/my-new-post.html

# 2. 编辑文章内容
# - 修改 Front Matter（title, date, author, category, excerpt, tags, reading_time, cover_image, lang, translate_url）
# - 编写 HTML 正文

# 3. 添加封面图
# 将图片放入 assets/images/covers/my-new-post.png

# 4. 索引自动更新（无需手动改 blog.html）

# 5. 创建中文版本（如需要）
# 重复以上步骤，文件放入 cn/ 目录
```

### 新增学习文章（4 步）

```bash
# 1. 复制模板
cp _templates/learning_post_template.md learning/python_guidelines/my-topic.md

# 2. 编辑文章内容
# - 修改 Front Matter
# - 使用 Markdown 编写正文

# 3. 设置 topic/order（用于自动生成索引）

# 4. 创建中文版本（如需要）
```

### 更新个人简介

```bash
# 编辑以下文件：
_includes/bio.html      # 英文简介
_includes/bio-cn.html   # 中文简介
```

### 添加出版物

```bash
# 1. 添加 BibTeX 文件
assets/bibtex/author2025title.bib

# 2. 更新数据文件
# 编辑 _data/publications.yml 添加新条目
```

### 提交更改

```bash
# 1. 查看更改
git status
git diff

# 2. 添加并提交
git add .
git commit -m "feat: Add new blog post about XXX"

# 3. 推送到远程
git push origin main
```

---

## 2.3 10 分钟上手指南

### 场景一：我想发布一篇新博客

**耗时：约 5-10 分钟**

1. **创建文章文件**
   ```bash
   # 复制模板并重命名
   cp _templates/blog_post_template.html blog/posts/my-first-post.html
   ```

2. **编辑 Front Matter**
   ```yaml
   ---
   layout: blog-post-layout
   title: "我的第一篇博客"
   date: 2025-12-31
   author: "Weiqin Wang"
   category: "Technical Tutorial"
   excerpt: "Short summary shown on the blog index."
   tags: ["Tag1", "Tag2"]
   reading_time: "6 minutes"
   cover_image: /assets/images/covers/my-first-post.png
   lang: en
   translate_url: /cn/blog/posts/my-first-post.html
   ---
   ```

3. **编写内容**
   - 在 `<div class="blog-content">` 内编写 HTML
   - 使用 `<h2>` `<h3>` 创建标题（会自动生成目录）
   - 代码块使用 `<pre><code class="language-python">` 格式

4. **添加封面图**
   - 准备一张正方形图片
   - 保存为 `assets/images/covers/my-first-post.png`

5. **完善 Front Matter**（摘要/标签/封面图等）

6. **本地预览**
   ```bash
   jekyll serve
   # 访问 http://localhost:4000/blog/posts/my-first-post.html
   ```

7. **发布**
   ```bash
   git add .
   git commit -m "feat: Add blog post 'My First Post'"
   git push
   ```

### 场景二：我想修改一篇已有文章

1. 直接编辑对应的文件（如 `blog/posts/git-basics.html`）
2. 本地预览确认无误
3. 提交并推送

### 场景三：我想更新首页的个人简介

1. 编辑 `_includes/bio.html`（英文）
2. 编辑 `_includes/bio-cn.html`（中文）
3. 本地预览确认
4. 提交并推送

---

## 2.4 必须补全的信息清单

> ⚠️ 以下信息不会自动补全，每次新增内容时必须手动填写！

| 操作 | 需要更新的文件 |
|------|----------------|
| 新增博客文章 | Front Matter（摘要/标签/封面图） |
| 新增学习主题 | 主题页 Front Matter（title/description/topic/order） |
| 新增学习文章 | Front Matter（topic/order） |
| 新增中文版本 | 英文页面的 `translate_url` 字段 |

---

## 2.5 文件模板位置

| 模板 | 路径 | 用途 |
|------|------|------|
| 博客文章模板 | `_templates/blog_post_template.html` | 创建新博客 |
| 学习文章模板 | `_templates/learning_post_template.md` | 创建新学习文章 |

---

---

<div class="doc-nav">
  <a href="./01-overview.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 项目概述</span>
  </a>
  <a href="./03-blog-maintenance.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">博客文章维护 →</span>
  </a>
</div>
