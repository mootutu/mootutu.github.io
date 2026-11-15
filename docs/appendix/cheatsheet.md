# 速查表

## Front Matter

### 博客文章

```yaml
layout: blog-post-layout
title: "Title"
date: 2026-01-01
author: "Weiqin Wang"
category: "Technical Tutorial"
excerpt: "One sentence."
tags: ["Tag1", "Tag2"]
reading_time: "6 minutes"
cover_image: /assets/images/covers/file-name.png
lang: en
```

### 学习笔记

```yaml
layout: learning-post-layout
title: "Title"
date: 2026-01-01
lang: en
topic: minimind
order: 1
topic_url: /learning/minimind.html
mathjax: true          # 可选
```

### 论文笔记（在学习笔记基础上增加）

```yaml
topic: paper-reading
topic_url: /learning/paper-reading.html
paper_year: 2026
authors: [Author A, Author B]
arxiv: "2603.12345"
paper_url: "https://arxiv.org/abs/2603.12345"
status: read           # unread / skimming / read / reproduced / integrated
relevance: high        # high / medium / low
tags: [reasoning]
one_line_summary: "One sentence."
```

### 学习主题页

```yaml
layout: learning-topic-layout
title: "Topic Title"
description: "What this topic is about."
topic: topic-key
order: 1
lang: en
```

### 页面可选字段（列表页 / About / 主题页）

| 字段 | 作用 |
|------|------|
| `heading` | 页头显示的标题（覆盖 `title`） |
| `eyebrow` | 标题上方小字 |
| `illo` | 页头插画：`blog` / `learning` / `reading` / `about` |
| `hide_back` | `true` 时隐藏返回链接 |
| `description` | 引言 + meta description |

---

## 路径

| 内容 | 位置 | 线上地址 |
|------|------|---------|
| 博客文章 | `blog/posts/x.html` | `/blog/posts/x.html` |
| 学习笔记 | `learning/minimind/x.md` | `/learning/minimind/x.html` |
| 论文笔记 | `learning/paper-reading/x.md` | `/learning/paper-reading/x.html` |
| 博客封面 | `assets/images/covers/x.png` | `/assets/images/covers/x.png` |
| 文章配图 | `assets/images/posts/x/` | `/assets/images/posts/x/…` |
| BibTeX | `assets/bibtex/x.bib` | `/assets/bibtex/x.bib` |
| 英文简历 | `cv/my_cv.pdf` | `/cv/my_cv.pdf` |
| 中文简历（站内未链接） | `cv_zh/cv_zh.pdf` | `/cv_zh/cv_zh.pdf` |

---

## 布局对照

| 内容 | `layout` | 实际结构 |
|------|----------|---------|
| 首页 | `base` | `base` |
| 博客列表 | `blog-layout` | `page` → `base` |
| 博客文章 | `blog-post-layout` | `post` → `base` |
| 学习区首页 | `learning-layout` | `page` → `base` |
| 学习主题页 | `learning-topic-layout` | `page` → `base` |
| 学习 / 论文笔记 | `learning-post-layout` | `post` → `base` |
| About | `about-layout` | `page` → `base` |

---

## 命令

### Jekyll

| 命令 | 说明 |
|------|------|
| `bundle exec jekyll serve --livereload` | 本地预览并自动刷新 |
| `bundle exec jekyll build` | 构建到 `_site/` |
| `bundle exec jekyll clean` | 清理缓存 |

### 简历

| 命令 | 说明 |
|------|------|
| `cd cv && latexmk my_cv.tex` | 编译英文简历 |
| `cd cv_zh && latexmk cv_zh.tex` | 编译中文简历（XeLaTeX） |

### Git

| 命令 | 说明 |
|------|------|
| `git status` / `git diff` | 查看改动 |
| `git add <file>` | 添加文件 |
| `git commit -m "feat: …"` | 提交 |
| `git push origin main` | 发布 |
| `git revert <commit>` | 撤销某次提交 |

---

## 代码高亮语言

已加载：`python`、`bash`、`toml`、`json`、`yaml`、`markup`（HTML）、`css`、`javascript`。

---

## 提交类型

`feat`（新内容）· `fix`（修正）· `docs`（文档）· `style`（样式）· `refactor`（重构）· `chore`（杂项）

---

## 设计变量（浅色 / 深色）

| 变量 | 浅色 | 深色 |
|------|------|------|
| `--bg` | `#f4f1ea` | `#1a1917` |
| `--surface` | `#faf8f3` | `#22211e` |
| `--text` | `#1f1e1b` | `#eeebe3` |
| `--muted` | `#6e6a62` | `#a29e94` |
| `--accent` | `#c4633f` | `#e08a63` |
| `--illo-clay` | `#d98a64` | `#b8694a` |
| `--illo-olive` | `#9aa67a` | `#6f7a55` |
| `--illo-sky` | `#9dbad6` | `#5d7c99` |

完整列表见 `assets/css/site.css` 顶部。

---

## 需要手动更新的内容

| 内容 | 位置 |
|------|------|
| 首页介绍、经历、奖项、服务 | `index.html` |
| 论文 | `_data/publications.yml` + `assets/bibtex/` |
| 简历 | `cv/`、`cv_zh/` |
| 联系按钮（Email、Scholar、LinkedIn、CV） | `_includes/contact-buttons.html` |
| 导航文字 | `_data/ui_text.yml` |

---

## 常用链接

| 名称 | 地址 |
|------|------|
| 线上网站 | https://mootutu.github.io |
| 本地预览 | http://localhost:4000 |
| 维护文档页 | https://mootutu.github.io/maintenance.html |
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
