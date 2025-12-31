# 项目概述

## 1.1 项目简介

这是一个基于 Jekyll 的个人学术/技术网站，托管在 GitHub Pages 上。网站包含以下主要功能：

- **个人首页**：展示个人简介、出版物、获奖情况
- **博客系统**：记录工具使用、技术教程、学习心得
- **学习笔记**：按主题组织的系统化学习内容
- **双语支持**：支持中英文切换

**线上地址**：`https://mootutu.github.io`

---

## 1.2 技术栈

| 技术 | 版本/说明 | 用途 |
|------|----------|------|
| **Jekyll** | - | 静态站点生成器，将 Markdown/HTML 转换为静态网页 |
| **Liquid** | - | Jekyll 的模板引擎，用于动态内容渲染 |
| **Kramdown** | - | Markdown 解析器，支持扩展语法 |
| **GitHub Pages** | - | 自动构建和托管服务 |
| **Prism.js** | v1.29.0 | 代码语法高亮（支持 Python、Bash、JSON 等） |
| **MathJax** | v3 | 数学公式渲染（LaTeX 语法） |
| **FontAwesome** | - | 图标库 |
| **Fira Code** | - | 代码专用等宽字体 |

---

## 1.3 核心概念

### 内容类型

本站有三种主要内容类型：

| 类型 | 说明 | 文件格式 | 存放位置 |
|------|------|----------|----------|
| **Blog Post** | 博客文章，按时间排序 | HTML | `/blog/posts/` |
| **Learning Topic** | 学习主题索引页 | HTML | `/learning/` |
| **Learning Post** | 学习文章，属于某个主题 | Markdown/HTML | `/learning/{topic}/` |

### 双语架构

网站采用目录镜像方式实现双语：

```
/                        # 英文版本
├── index.html
├── blog.html
├── blog/posts/
└── learning/

/cn/                     # 中文版本（镜像结构）
├── index.html
├── blog.html
├── blog/posts/
└── learning/
```

每个页面通过 `translate_url` 字段链接到对应的翻译版本。

### 布局系统

Jekyll 使用布局（Layout）来定义页面结构：

```
_layouts/
├── default.html           # 基础布局（首页、关于页）
├── blog-layout.html       # 博客列表页布局
├── blog-post-layout.html  # 博客文章布局
├── learning-layout.html   # 学习主题列表布局
├── learning-topic-layout.html  # 单个学习主题布局
└── learning-post-layout.html   # 学习文章布局
```

---

## 1.4 目录结构

### 完整目录树

```
mootutu.github.io/
│
├── _config.yml              # Jekyll 主配置文件
├── _data/                   # 站点数据
│   └── ui_text.yml          # 多语言 UI 文本
│
├── _layouts/                # 页面布局模板
│   ├── default.html
│   ├── blog-layout.html
│   ├── blog-post-layout.html
│   ├── learning-layout.html
│   ├── learning-topic-layout.html
│   └── learning-post-layout.html
│
├── _includes/               # 可复用组件
│   ├── navigation.html      # 导航栏
│   ├── bio.html             # 英文个人简介
│   └── bio-cn.html          # 中文个人简介
│
├── _templates/              # 内容创建模板
│   ├── blog_post_template.html
│   └── learning_post_template.md
│
├── assets/                  # 静态资源
│   ├── css/                 # 样式表
│   ├── javascripts/         # JavaScript
│   ├── images/              # 图片资源
│   │   ├── covers/          # 博客封面图
│   │   └── posts/           # 文章内图片
│   ├── fontawesome/         # 图标库
│   └── bibtex/              # 论文引用文件
│
├── blog/                    # 英文博客
│   └── posts/               # 博客文章
│       ├── modernizing-python.html
│       ├── git-basics.html
│       └── ...
│
├── learning/                # 英文学习内容
│   ├── python.html          # Python 主题索引页
│   ├── python_guidelines/   # Python 文章目录
│   │   ├── python_style_conventions.md
│   │   └── python_language_conventions.md
│   ├── minimind.html        # Minimind 主题索引页
│   └── minimind/            # Minimind 文章目录
│
├── cn/                      # 中文版本
│   ├── index.html
│   ├── blog.html
│   ├── blog/posts/
│   ├── learning.html
│   └── learning/
│
├── docs/                    # 维护文档（本目录）
│
├── index.html               # 英文首页
├── blog.html                # 英文博客列表
├── learning.html            # 英文学习主题列表
├── about.html               # 关于页面
└── search.html              # 搜索功能（生成 search.json）
```

### 目录职责速查

| 目录 | 职责 | 需要手动维护 |
|------|------|-------------|
| `_layouts/` | 页面结构模板 | 很少修改 |
| `_includes/` | 可复用组件 | 更新个人简介时 |
| `_data/` | 配置数据 | 添加新 UI 文本时 |
| `_templates/` | 创建内容的模板 | 几乎不需要 |
| `assets/images/covers/` | 博客封面图 | 每次新增博客 |
| `assets/images/posts/` | 文章内图片 | 需要时 |
| `blog/posts/` | 博客文章 | 每次新增/修改博客 |
| `learning/` | 学习内容 | 每次新增/修改学习内容 |
| `cn/` | 中文版本 | 每次新增内容时同步 |

---

## 1.5 关键文件说明

### 配置文件

| 文件 | 说明 |
|------|------|
| `_config.yml` | Jekyll 主配置，定义站点元数据、插件、集合 |
| `_data/ui_text.yml` | 多语言 UI 文本（导航栏、按钮、标签等） |

### 索引页面（需手动维护）

| 文件 | 说明 |
|------|------|
| `blog.html` | 英文博客列表，新增博客后必须更新 |
| `cn/blog.html` | 中文博客列表 |
| `learning.html` | 英文学习主题列表，新增主题后必须更新 |
| `cn/learning.html` | 中文学习主题列表 |
| `learning/python.html` | Python 主题索引，新增文章后必须更新 |

### 组件文件

| 文件 | 说明 |
|------|------|
| `_includes/navigation.html` | 导航栏（含搜索功能） |
| `_includes/bio.html` | 英文个人简介 |
| `_includes/bio-cn.html` | 中文个人简介 |

---

## 1.6 内容类型对比

| 特性 | Blog Post | Learning Topic | Learning Post |
|------|-----------|----------------|---------------|
| **用途** | 独立文章，技术教程 | 主题索引/导航页 | 系统化学习笔记 |
| **文件格式** | HTML | HTML | Markdown（推荐） |
| **布局** | `blog-post-layout` | `learning-topic-layout` | `learning-post-layout` |
| **存放位置** | `/blog/posts/` | `/learning/` | `/learning/{topic}/` |
| **需要封面图** | 是 | 否 | 否 |
| **Front Matter** | title, date, author, category, lang, translate_url | lang, translate_url | title, date, lang, topic_url, translate_url, mathjax |
| **索引维护** | 更新 `blog.html` | 更新 `learning.html` | 更新主题页面 |

---

---

<div class="doc-nav">
  <div class="doc-nav-placeholder"></div>
  <a href="./02-quickstart.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">快速开始 →</span>
  </a>
</div>
