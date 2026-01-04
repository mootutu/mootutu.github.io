# 速查表

## Front Matter 速查

### 博客文章

```yaml
---
layout: blog-post-layout
title: "文章标题"
date: 2025-12-31
author: "Weiqin Wang"
category: "Technical Tutorial"
lang: en
translate_url: /cn/blog/posts/xxx.html
---
```

| 字段 | 必填 | 值 |
|------|------|-----|
| layout | 是 | `blog-post-layout` |
| title | 是 | 字符串 |
| date | 是 | YYYY-MM-DD |
| author | 是 | 字符串 |
| category | 是 | 字符串 |
| lang | 是 | `en` 或 `zh-CN` |
| translate_url | 是 | 绝对路径 |

---

### 学习文章

```yaml
---
layout: learning-post-layout
title: "文章标题"
date: 2025-12-31
lang: en
topic_url: /learning/topic.html
translate_url: /cn/learning/topic/xxx.html
mathjax: false
---
```

| 字段 | 必填 | 值 |
|------|------|-----|
| layout | 是 | `learning-post-layout` |
| title | 是 | 字符串 |
| date | 是 | YYYY-MM-DD |
| lang | 是 | `en` 或 `zh-CN` |
| topic_url | 是 | 绝对路径 |
| translate_url | 是 | 绝对路径 |
| mathjax | 否 | `true` 或 `false` |

---

### 学习主题

```yaml
---
layout: learning-topic-layout
lang: en
translate_url: /cn/learning/topic.html
---
```

---

## 路径速查

### 内容路径

| 内容 | 英文路径 | 中文路径 |
|------|----------|----------|
| 博客文章 | `/blog/posts/xxx.html` | `/cn/blog/posts/xxx.html` |
| 学习主题 | `/learning/xxx.html` | `/cn/learning/xxx.html` |
| 学习文章 | `/learning/topic/xxx.html` | `/cn/learning/topic/xxx.html` |

### 资源路径

| 资源 | 路径 |
|------|------|
| 封面图 | `/assets/images/covers/xxx.png` |
| 文章图片 | `/assets/images/posts/article/xxx.png` |
| BibTeX | `/assets/bibtex/xxx.bib` |
| CSS | `/assets/css/xxx.css` |

---

## 命令速查

### Jekyll 命令

| 命令 | 说明 |
|------|------|
| `jekyll serve` | 启动本地服务器 |
| `jekyll serve --drafts` | 显示草稿 |
| `jekyll serve --livereload` | 实时重载 |
| `jekyll build` | 构建站点 |
| `jekyll clean` | 清理缓存 |

### Git 命令

| 命令 | 说明 |
|------|------|
| `git status` | 查看状态 |
| `git add .` | 添加所有文件 |
| `git commit -m "msg"` | 提交 |
| `git push` | 推送 |
| `git pull` | 拉取 |
| `git log --oneline` | 查看历史 |

---

## 代码高亮语言

| 语言 | 标识符 |
|------|--------|
| Python | `python` |
| Bash/Shell | `bash` |
| JavaScript | `javascript` |
| JSON | `json` |
| YAML | `yaml` |
| TOML | `toml` |
| HTML | `html` |
| CSS | `css` |

**使用方式（HTML）**：
```html
<pre><code class="language-python">code here</code></pre>
```

**使用方式（Markdown）**：
````markdown
```python
code here
```
````

---

## 布局对照

| 布局 | 用途 |
|------|------|
| `default` | 首页、关于页 |
| `blog-layout` | 博客列表 |
| `blog-post-layout` | 博客文章 |
| `learning-layout` | 学习主题列表 |
| `learning-topic-layout` | 单个主题 |
| `learning-post-layout` | 学习文章 |

---

## 文件命名规则

| 类型 | 规则 | 示例 |
|------|------|------|
| 博客文章 | `kebab-case.html` | `git-basics.html` |
| 学习文章 | `snake_case.md` | `python_style.md` |
| 封面图 | 与文章同名 | `git-basics.png` |
| 文章图片 | 序号或描述 | `image-1.png` |

---

## 常用 Category

| 英文 | 中文 |
|------|------|
| Technical Tutorial | 技术教程 |
| Tool Usage | 工具使用 |
| Development | 开发实践 |
| Learning Notes | 学习笔记 |

---

## Git 提交类型

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat: Add blog post` |
| fix | 修复 | `fix: Correct typo` |
| docs | 文档 | `docs: Update README` |
| style | 样式 | `style: Adjust spacing` |
| refactor | 重构 | `refactor: Simplify logic` |
| chore | 杂项 | `chore: Update deps` |

---

## 暗黑模式颜色

| 元素 | 亮色 | 暗色 |
|------|------|------|
| 背景 | `#ffffff` | `#1a1a2e` |
| 文字 | `#333333` | `#e2e8f0` |
| 次要文字 | `#666666` | `#cbd5e1` |
| 链接 | `#0066cc` | `#60a5fa` |
| 边框 | `#e9ecef` | `#334155` |
| 代码背景 | `#f8f9fa` | `#1e293b` |

---

## 需手动更新的内容

| 操作 | 需更新 |
|------|--------|
| 新增博客 | Front Matter（摘要/标签/封面图） |
| 新增主题 | 主题页 Front Matter（title/description/topic/order） |
| 新增学习文章 | Front Matter（topic/order） |
| 新增中文版本 | 英文页的 `translate_url` |

---

## 常用链接

| 链接 | 地址 |
|------|------|
| 线上网站 | https://mootutu.github.io |
| 本地预览 | http://localhost:4000 |
| GitHub 仓库 | https://github.com/mootutu/mootutu.github.io |
| Jekyll 文档 | https://jekyllrb.com/docs/ |

---

<div class="doc-nav">
  <a href="./templates.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 模板文件</span>
  </a>
  <div class="doc-nav-placeholder"></div>
</div>
