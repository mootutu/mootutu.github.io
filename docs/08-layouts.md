# 布局与组件

## 8.1 布局体系

所有页面最终都套用 `base.html`。中间有两种“页面类型”布局，再往外是一层很薄的包装布局，用来给不同栏目设置默认值：

```
base.html                     骨架：<head>、导航、页脚、脚本
├── page.html                 列表页 / 介绍页：页头 + 正文
│   ├── blog-layout.html          博客列表
│   ├── learning-layout.html      学习区首页
│   ├── learning-topic-layout.html 学习主题页
│   └── about-layout.html         About
├── post.html                 文章页：页头 + 正文 + 目录
│   ├── blog-post-layout.html     博客文章
│   └── learning-post-layout.html 学习笔记 / 论文笔记
└── （index.html 直接使用 base）
```

包装布局只有 Front Matter 和一行 `{{ content }}`，例如：

```yaml
# _layouts/blog-post-layout.html
---
layout: post
back_url: /blog.html
back_label: All posts
eyebrow: Blog
---
{{ content }}
```

保留这些旧布局名，是因为博客列表、主题页、搜索都靠 `where: "layout", "blog-post-layout"` 这类条件筛选内容；内容文件的 `layout` 字段因此不用改。

### 布局变量的合并

Jekyll 会把子布局的 Front Matter 合并进父布局的 `layout` 变量，所以在 `page.html`、`post.html`、`base.html` 里都能读到包装布局设置的值。取值优先级是：

```liquid
{% assign illo = page.illo | default: layout.illo %}
```

即**页面自己的 Front Matter 优先**，其次才是布局的默认值。

---

## 8.2 `base.html`

负责所有页面共用的部分：

- `<head>`：标题（`页面标题 · Weiqin Wang`）、description、favicon、`theme-init.html`、Google Fonts、`site.css`
- 隐藏的 SVG 滤镜 `#hand-drawn`，给插画线条加手绘抖动
- `site-header.html`、`<main>`、`site-footer.html`
- 回到顶部按钮
- `theme-toggle.js`、`site.js`

可由布局开启的选项：

| 布局变量 | 作用 | 谁开启 |
|---------|------|--------|
| `code: true` | 加载 Prism.js 主题与语言包 | `post.html` |
| `progress: true` | 页面顶部显示阅读进度条 | `post.html` |

可由页面开启的选项：

| 页面变量 | 作用 |
|---------|------|
| `mathjax: true` | 加载 MathJax |
| `description` | 写入 `<meta name="description">`；缺省时依次使用 `excerpt`、`one_line_summary` |

---

## 8.3 `page.html`

用于列表页和 About。页头包含：返回链接、小字、标题、引言、右侧插画；下面是 `<div class="container page-body">{{ content }}</div>`。

| 变量 | 来源 | 说明 |
|------|------|------|
| `title` | 页面 | 标题（同时用于浏览器标签） |
| `heading` | 页面 | 页头显示的标题，填了就覆盖 `title`（About 用它显示 “Hi, I'm Weiqin.”） |
| `description` | 页面 | 标题下的引言 |
| `eyebrow` | 页面 / 布局 | 标题上方的小字 |
| `illo` | 页面 / 布局 | 插画名，对应 `_includes/illo/{名字}.html`；不填则没有插画、标题占满宽度 |
| `back_url` / `back_label` | 页面 / 布局 | 返回链接 |
| `hide_back` | 页面 | 设为 `true` 隐藏返回链接 |

---

## 8.4 `post.html`

用于博客文章和学习笔记：

- **页头**：返回链接（`topic_url` 优先，否则用布局的 `back_url`）、小字（`category`，否则布局的 `eyebrow`）、标题、引言（`one_line_summary` 或 `excerpt`）、元信息
- **元信息**：作者（论文笔记显示 `authors` 和 `paper_year`）、日期、自动计算的阅读时长、`status`、`paper_url` 链接
- **正文**：`<div class="prose" id="post-content">`
- **目录**：右侧 `<aside class="toc">`，由 `site.js` 根据正文 `h2`/`h3` 生成，宽度小于 1080px 时隐藏
- **文末**：返回按钮

---

## 8.5 组件（`_includes/`）

| 文件 | 用途 | 使用位置 |
|------|------|---------|
| `site-header.html` | 导航栏、Light / Auto / Dark 主题切换、隐藏入口面板 | `base.html` |
| `site-footer.html` | 版权、最后更新时间 | `base.html` |
| `theme-init.html` | 首屏前设置深色/浅色，视图过渡样式 | `base.html` |
| `publication-list.html` | 论文列表与 BibTeX 弹窗 | `index.html` |
| `blog_list.html` | 博客卡片网格 | `blog.html` |
| `learning_topics.html` | 学习主题卡片 | `learning.html` |
| `learning_entries.html` | 主题下的笔记列表 | `learning/minimind.html`、`learning/python.html` |
| `illo/blog.html` 等 | 页头插画 | `page.html`（按 `illo` 变量引入） |

导航栏依次是 Home、Publications、Experience、Awards、Service，都指向首页：Home 是 `/`，其余是 `/#publications` 这样的区块锚点。每个链接带有 `data-section="区块 id"`；在首页滚动时，`site.js` 会高亮当前所在区块对应的链接（页面顶部时高亮 Home）。新增首页区块并想放进导航时，给 `<section>` 一个 `id`，再在 `site-header.html` 加一个带同名 `data-section` 的链接，并在 `ui_text.yml` 加文字。

### 隐藏页面

博客、学习区、论文阅读库、About 目前**不在导航中**，首页也没有链接指向它们，但页面照常构建。

**隐藏入口**：在任意页面上，连续快速点击左上角名字前的橙色圆点 5 次（1.5 秒内），或在页面上直接键入 `mootutu`，会弹出一个 “Back room” 面板，列出下面这些页面。面板在 `_includes/site-header.html` 的 `#secret-dialog`，触发逻辑在 `site.js` 的 “Hidden entrance” 部分；单击圆点不会跳转首页（点名字文字才会）。

也可以直接输入地址访问：

| 页面 | 地址 |
|------|------|
| 博客列表 | `/blog.html` |
| 学习区首页 | `/learning.html` |
| 论文阅读库 | `/learning/paper-reading.html` |
| About | `/about.html` |
| 维护文档 | `/maintenance.html` |

这些页面之间的互相链接（例如文章页的返回链接）都还在。要重新公开某个页面，在 `site-header.html` 里加回导航链接即可。

---

## 8.6 修改布局的注意事项

1. 改 `base.html`、`site-header.html`、`site.css` 会影响所有页面，改完至少检查：首页、博客列表、一篇博客、学习区首页、一个主题页、一篇带公式的笔记、论文阅读库、About
2. 每个页面都在浅色和深色模式下各看一次
3. 用浏览器开发者工具切到 375px 宽检查手机布局，确认没有横向滚动条
4. 新样式用 `site.css` 里的变量，不要写死颜色
5. 新加的动画要在 `prefers-reduced-motion: reduce` 下能关掉（全局规则已覆盖大多数情况）

---

## 8.7 新增一种页面

大多数情况不需要新布局，用 `page` 或 `post` 加一个包装布局即可：

```yaml
# _layouts/talks-layout.html
---
layout: page
eyebrow: Talks
illo: about
back_url: /
back_label: Home
---
{{ content }}
```

页面里写 `layout: talks-layout`。只有页面结构确实和现有两种都不同时，才新建一个继承 `base` 的布局。

---

## 8.8 Liquid 常用写法

```liquid
{% assign posts = site.pages | where: "layout", "blog-post-layout" | sort: "date" | reverse %}
{% for post in posts %}{{ post.title }}{% endfor %}

{{ page.date | date: "%b %-d, %Y" }}
{{ page.title | default: "Untitled" }}
{% include illo/{{ illo }}.html %}
```

常用变量：`site.pages`、`site.data.publications`、`site.data.ui_text.en`、`site.time`、`page.url`、`page.title`、`layout.xxx`。

---

<div class="doc-nav">
  <a href="./07-personal-info.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 个人信息维护</span>
  </a>
  <a href="./09-config.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">配置文件 →</span>
  </a>
</div>
