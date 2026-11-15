# 项目概述

## 1.1 项目简介

这是一个基于 Jekyll 的个人学术主页，托管在 GitHub Pages 上，线上地址是 `https://mootutu.github.io`。

站点包含：

- **首页**：个人介绍、研究方向、论文、教育经历、奖项、学术服务
- **博客**：工具使用、技术教程类文章
- **学习区**：按主题组织的学习笔记（Minimind、Python Guidelines 等）
- **论文阅读库**：可搜索、可筛选的论文阅读笔记
- **About**：简短自我介绍与联系方式
- **简历**：英文、中文两份 LaTeX 简历（PDF）

站点只有英文版，没有中文镜像，也没有语言切换。

---

## 1.2 设计风格

2026-11 起全站使用统一的新设计：

- 暖色米白底色，深色模式是暖黑底，陶土橙作为点缀色
- 标题用衬线体 Source Serif 4，正文用 Inter，代码用 JetBrains Mono
- 首页和各列表页的页头配有手绘风 SVG 插画
- 动效克制：内容渐入、插画线条描绘、卡片悬停上浮；系统开启“减少动态效果”时自动关闭

颜色、字体、间距都定义在 `assets/css/site.css` 顶部的 CSS 变量里，详见 [06-静态资源管理](./06-assets.md) 和 [10-特殊功能](./10-features.md)。

---

## 1.3 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Jekyll | 3.10（由 `github-pages` gem 决定） | 静态站点生成 |
| Liquid | 4 | 模板语言 |
| Kramdown | — | Markdown 解析 |
| Prism.js | 1.29.0（cdnjs） | 代码高亮 |
| MathJax | 3.2.2（cdnjs） | 数学公式 |
| Google Fonts | — | Source Serif 4、Inter、JetBrains Mono |
| GitHub Pages | — | 自动构建与托管 |

站点没有使用任何前端框架，交互都由 `assets/javascripts/site.js` 和 `theme-toggle.js` 两个原生 JS 文件实现。

---

## 1.4 内容类型

| 类型 | 说明 | 格式 | 位置 | 布局 |
|------|------|------|------|------|
| Blog Post | 博客文章 | HTML | `blog/posts/` | `blog-post-layout` |
| Learning Topic | 学习主题页 | HTML | `learning/*.html` | `learning-topic-layout` |
| Learning Post | 学习笔记 | Markdown（推荐）或 HTML | `learning/{目录}/` | `learning-post-layout` |
| Paper Note | 论文阅读笔记（学习笔记的一种，`topic: paper-reading`） | Markdown | `learning/paper-reading/` | `learning-post-layout` |

博客列表、学习主题列表、主题下的笔记列表、论文阅读库都会根据 Front Matter **自动生成**，不需要手动维护索引。

---

## 1.5 目录结构

```
mootutu.github.io/
├── _config.yml
├── _data/
│   ├── publications.yml        # 首页论文列表
│   └── ui_text.yml             # 导航等界面文字
├── _layouts/
│   ├── base.html               # 所有页面的骨架
│   ├── page.html               # 列表页 / 介绍页
│   ├── post.html               # 文章页
│   ├── blog-layout.html        # ↓ 以下都是套用 page / post 的包装层
│   ├── blog-post-layout.html
│   ├── learning-layout.html
│   ├── learning-topic-layout.html
│   ├── learning-post-layout.html
│   └── about-layout.html
├── _includes/
│   ├── site-header.html        # 导航栏（首页各区块的锚点）
│   ├── site-footer.html        # 页脚
│   ├── theme-init.html         # 首屏前设置深色/浅色
│   ├── publication-list.html   # 首页论文列表 + BibTeX 弹窗
│   ├── blog_list.html          # 博客卡片列表
│   ├── learning_topics.html    # 学习主题卡片
│   ├── learning_entries.html   # 主题下的笔记列表
│   └── illo/                   # 页头插画（blog / learning / reading / about）
├── _templates/
│   ├── blog_post_template.html
│   ├── learning_post_template.md
│   └── paper_reading_template.md
├── assets/
│   ├── css/site.css            # 全站唯一的样式表
│   ├── javascripts/
│   │   ├── site.js             # 全站交互
│   │   └── theme-toggle.js     # 深色模式切换
│   ├── images/                 # 封面、文章配图、图标等
│   ├── bibtex/                 # 论文 BibTeX
│   └── fontawesome/            # 旧图标库（目前没有页面引用）
├── blog/posts/
├── learning/
│   ├── minimind.html  minimind/
│   ├── python.html    python_guidelines/
│   └── paper-reading.html  paper-reading/
├── cv/                         # 英文简历 my_cv.tex → my_cv.pdf
├── cv_zh/                      # 中文简历 cv_zh.tex → cv_zh.pdf
├── docs/                       # 本维护文档（不参与 Jekyll 构建）
├── index.html  blog.html  learning.html  about.html
├── search.html                 # 生成 /search.json
└── maintenance.html            # 在线阅读 docs/ 的页面
```

### 目录职责速查

| 目录 / 文件 | 职责 | 多久改一次 |
|-------------|------|-----------|
| `index.html` | 首页文字、研究方向、教育、奖项、服务 | 有新经历时 |
| `_data/publications.yml` | 论文列表 | 有新论文时 |
| `blog/posts/` | 博客文章 | 写新文章时 |
| `learning/` | 学习主题与笔记 | 写新笔记时 |
| `cv/`、`cv_zh/` | 简历 | 有新经历时 |
| `_layouts/`、`_includes/` | 页面结构 | 很少 |
| `assets/css/site.css` | 全站样式 | 调整设计时 |

---

## 1.6 页面与布局对照

| 页面 | 文件 | 布局链 |
|------|------|--------|
| 首页 | `index.html` | `base` |
| 博客列表 | `blog.html` | `blog-layout` → `page` → `base` |
| 博客文章 | `blog/posts/*.html` | `blog-post-layout` → `post` → `base` |
| 学习主题列表 | `learning.html` | `learning-layout` → `page` → `base` |
| 学习主题页 | `learning/*.html` | `learning-topic-layout` → `page` → `base` |
| 学习笔记 / 论文笔记 | `learning/*/*.md` | `learning-post-layout` → `post` → `base` |
| About | `about.html` | `about-layout` → `page` → `base` |
| 维护文档页 | `maintenance.html` | 无（独立页面） |

---

<div class="doc-nav">
  <div class="doc-nav-placeholder"></div>
  <a href="./02-quickstart.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">快速开始 →</span>
  </a>
</div>
