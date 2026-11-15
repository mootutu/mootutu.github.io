# 命名规范汇总

## 12.1 文件命名

| 类型 | 规则 | 示例 |
|------|------|------|
| 博客文章 | 小写 + 连字符，`.html` | `blog/posts/git-basics.html` |
| 学习笔记 | 小写，连字符或下划线，`.md`（推荐） | `learning/minimind/part2.md` |
| 论文笔记 | `arXiv编号-简短标题.md` | `learning/paper-reading/2603.25450-cross-model-disagreement.md` |
| 学习主题页 | 小写，`.html` | `learning/minimind.html` |
| 布局 | 小写 + 连字符 | `_layouts/blog-post-layout.html` |
| 组件 | 小写，新组件用连字符 | `_includes/site-header.html` |
| 插画 | 小写单词 | `_includes/illo/reading.html` |
| BibTeX | `姓年份关键词.bib`，全小写 | `wang2025ranked.bib` |

已有的 `blog_list.html`、`learning_entries.html`、`python_guidelines/` 等使用下划线，属于历史命名，保持不变即可；新文件统一用连字符。

---

## 12.2 目录命名

| 目录 | 规则 | 示例 |
|------|------|------|
| 学习笔记目录 | 与主题对应，小写 | `learning/minimind/` |
| 文章配图目录 | 与文章文件名相同 | `assets/images/posts/modernizing-python/` |
| 笔记配图目录 | 与主题对应 | `assets/images/minimind/` |

---

## 12.3 图片命名

| 类型 | 规则 | 示例 |
|------|------|------|
| 博客封面 | 与文章文件名相同 | `covers/git-basics.png` |
| 文章配图 | `image.png`、`image-1.png`……或描述性名称 | `posts/git-basics/branch-graph.png` |

不要使用空格、大写字母和中文。

---

## 12.4 Front Matter 字段

### 博客文章

`layout`、`title`、`date`、`author`、`category`、`excerpt`、`tags`、`reading_time`、`cover_image`、`lang`

### 学习笔记

`layout`、`title`、`date`、`lang`、`topic`、`order`、`topic_url`，可选 `mathjax`、`category`、`excerpt`

### 论文笔记

学习笔记字段 + `paper_year`、`authors`、`arxiv`、`doi`、`paper_url`、`status`、`relevance`、`tags`、`methods`、`datasets`、`models`、`one_line_summary`

### 学习主题页

`layout`、`title`、`description`、`topic`、`order`、`lang`，可选 `eyebrow`、`illo`、`hide_back`

### 列表页 / About

`layout`、`title`、`description`、`lang`，可选 `heading`、`eyebrow`、`illo`

### 取值约定

| 字段 | 约定 |
|------|------|
| `date` | `YYYY-MM-DD` |
| `lang` | 固定 `en` |
| `topic` | 小写，多个单词用连字符（`paper-reading`） |
| `order` | 正整数，从 1 开始 |
| `status` | `unread` / `skimming` / `read` / `reproduced` / `integrated` |
| `relevance` | `high` / `medium` / `low` |
| 路径类字段 | 以 `/` 开头的绝对路径 |

---

## 12.5 URL

| 内容 | 地址 |
|------|------|
| 博客文章 | `/blog/posts/{文件名}.html` |
| 学习主题 | `/learning/{主题}.html` |
| 学习笔记 | `/learning/{目录}/{文件名}.html`（`.md` 发布后变成 `.html`） |
| 资源 | `/assets/...` |

---

## 12.6 CSS 类名

新的样式采用 BEM 风格：`块__元素--修饰`。

```
.post-card            块
.post-card__title     元素
.venue-tag--muted     修饰
.is-visible / .is-open / .is-active   由 JS 切换的状态类
```

常用类名：

| 类名 | 用途 |
|------|------|
| `.container` | 居中、限制最大宽度、左右留白 |
| `.prose` | 正文排版（文章、About） |
| `.button` / `.button--primary` | 圆角按钮 |
| `.chip` | 小号描边按钮（论文链接） |
| `.tag` | 灰底小标签 |
| `.venue-tag` | 橙色/灰色标签（会议、阅读状态） |
| `.text-link` | 带下划线过渡的文字链接 |
| `.eyebrow` | 标题上方的大写小字 |
| `.illo` | 插画 SVG |

---

## 12.7 Git 提交信息

格式 `类型: 简短说明`，类型为 `feat`、`fix`、`docs`、`style`、`refactor`、`chore`，详见 [11.2](./11-deployment.md#112-提交信息规范)。

---

<div class="doc-nav">
  <a href="./11-deployment.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 部署与发布</span>
  </a>
  <a href="./13-checklist.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">维护检查清单 →</span>
  </a>
</div>
