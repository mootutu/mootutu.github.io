# 特殊功能

## 10.1 搜索

### 使用方式

> **目前未启用。** 搜索只覆盖博客和论文阅读笔记，这些页面已从导航隐藏，所以导航栏里的搜索按钮和搜索面板也一起去掉了。`/search.json` 仍会生成，`site.js` 里的搜索代码也保留着：在 `site-header.html` 里加回 `#search-open` 按钮和 `#search-dialog` 面板（可从 git 历史中找回）即可恢复，恢复后按钮或 **⌘K**（Windows/Linux 为 **Ctrl+K**）打开面板。

### 索引来源

`search.html` 在构建时生成 `/search.json`，收录：

- 所有博客文章（`layout: blog-post-layout`）
- 所有论文阅读笔记（`topic: paper-reading`）
- 论文阅读库页面

Minimind、Python 等普通学习笔记**目前不在搜索范围内**。如需加入，在 `search.html` 中把对应条件加入 `pages_to_index`。

每条记录包含 `title`、`category`、`tags`、`url`、`date`、`content`。`content` 优先使用 `description`，其次 `intro`，否则取正文前 50 个词。

### 匹配规则

- 标题、标签：输入 2 个字符以上就匹配
- 正文摘要：输入 4 个字符以上才匹配（避免短词误中）
- 最多显示 8 条结果

实现位于 `assets/javascripts/site.js` 的 “Search palette” 部分，面板结构在 `_includes/site-header.html`。

---

## 10.2 深色模式

### 原理

- `<html>` 上有 `dark-mode` 类时为深色
- `site.css` 在 `html.dark-mode` 下重新定义所有颜色变量，组件只用变量，所以不需要为每个组件单独写深色样式
- `_includes/theme-init.html` 在页面渲染前读取 `localStorage` 中的 `theme_preference`；没有保存过时跟随系统设置
- 导航栏右侧是三段选择 `#theme-switch`：**Light / Auto / Dark**。选 Light 或 Dark 会把选择存进 `theme_preference`；选 Auto 会删除这个记录，改为跟随系统设置
- 切换时整页颜色淡入淡出（浏览器支持 View Transitions 时；系统开启“减少动态效果”时直接切换）
- 处于 Auto 时，系统主题变化（例如晚上系统自动切到深色）会实时跟随
- 宽度不足 1100px 时只显示三个图标，不显示文字
- 逻辑在 `assets/javascripts/theme-toggle.js`，样式在 `site.css` 的 “Theme switch” 部分

### 给新组件加深色支持

只要用变量写颜色即可：

```css
.my-card {
  background: var(--surface);
  color: var(--text-soft);
  border: 1px solid var(--line);
}
```

确实需要特殊处理时，用 `html.dark-mode .my-card { … }`。

---

## 10.3 文章目录（TOC）与阅读进度

只在文章页（`post.html`）生效：

- 目录根据正文中的 `h2`、`h3` 生成，至少 2 个标题才显示
- 没有 `id` 的标题会自动补 `section-1`、`section-2`……
- 目录文字会去掉开头的编号（如 `1.`、`2.3`）
- 滚动时高亮当前所在小节
- 屏幕宽度小于 1080px 时隐藏目录
- 页面顶部有一条橙色阅读进度条

实现位于 `site.js` 的 “Article: reading progress + table of contents” 部分。

---

## 10.4 阅读时长

文章页的阅读时长在 `post.html` 中计算：

```liquid
{% assign word_count = content | strip_html | number_of_words %}
{% assign read_time = word_count | divided_by: 200 %}
```

不足 1 分钟按 1 分钟显示。博客列表卡片上显示的是 Front Matter 里手写的 `reading_time`，两者可能不同。

---

## 10.5 代码高亮

- 使用 Prism.js 1.29.0（`prism-tomorrow` 主题），只在文章页加载
- 已加载语言：`python`、`bash`、`toml`、`json`、`yaml`，以及 Prism 核心自带的 `markup`、`css`、`clike`、`javascript`
- 代码块背景固定为暖黑色，浅色、深色模式下都一样

HTML 写法：

```html
<pre><code class="language-python">print("hello")</code></pre>
```

Markdown 写法：三个反引号后写语言名。

### 增加语言

在 `_layouts/base.html` 的 `{% if layout.code %}` 区块里加一行：

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-rust.min.js" defer></script>
```

---

## 10.6 数学公式

- Front Matter 写 `mathjax: true` 后，`base.html` 会加载 MathJax 3.2.2
- 行内公式用 `$…$` 或 `\(…\)`，独立公式用 `$$…$$`
- 任何页面都可以开启，不限于学习笔记
- 过宽的公式在自身范围内横向滚动（`site.css` 末尾的 MathJax 规则）

---

## 10.7 动效

| 效果 | 用法 |
|------|------|
| 滚动渐入 | 给元素加 `data-reveal` 属性；用 `style="--reveal-delay: 0.08s"` 设置延迟 |
| 首屏依次浮现 | 父元素加 `class="intro"`，前 5 个子元素按顺序延迟出现 |
| 卡片悬停上浮 | `.focus-card`、`.post-card`、`.topic-card`、`.paper-card` 自带 |
| 链接下划线过渡 | 给链接加 `class="text-link"` |

滚动渐入用的是 `translate` + `opacity` 动画，不会和卡片悬停用的 `transform` 冲突。

系统开启“减少动态效果”（`prefers-reduced-motion: reduce`）时，`site.css` 会把所有动画和过渡时长压到接近 0，内容直接显示。

---

## 10.8 插画

### 现有插画

| 插画 | 位置 | 内容 |
|------|------|------|
| 首页主图 | `index.html` 的 `.hero__art` | 书里长出推理树，指向打勾的气泡 |
| 研究方向卡片 ×3 | `index.html` 的 `.focus-card__art` | 投票路径、重叠的语义圆、卡片与锚 |
| 区块小图 ×4 | `_includes/illo/section-*.html` | 论文：纸和笔；经历：学士帽与小旗；奖项：奖杯；服务：清单（窄屏隐藏） |
| `illo/blog.html` | 博客列表页头 | 笔记本、铅笔、咖啡杯 |
| `illo/learning.html` | 学习区、主题页页头 | 书堆和小苗 |
| `illo/reading.html` | 论文阅读库页头 | 放大镜和论文 |
| `illo/about.html` | About 页头 | “hi!” 对话气泡 |

### 画法约定

插画都是内联 SVG，外层加 `class="illo"`：

| 类名 | 作用 |
|------|------|
| `ink` / `ink-thin` | 黑色线条（深色模式下变浅色），粗 2.4 / 1.6 |
| `hand` | 加在 `<g>` 上，给线条加手绘抖动（滤镜 `#hand-drawn` 定义在 `base.html`） |
| `f-clay` `f-olive` `f-sky` `f-sand` `f-paper` `f-ink` | 填充色，对应 `--illo-*` 变量 |
| `blob` | 背景色块：缩放出现后缓慢漂浮 |
| `draw` | 线条描绘动画，用 `--len`（线长上限）和 `--d`（延迟）控制 |
| `pop` | 弹出出现，`--d` 控制延迟 |
| `float` | 弹出后上下轻微浮动 |
| `twinkle` | 小星星闪烁 |

页头动画只在 `.hero__art` 和 `.page-hero__art` 内生效。

### 新增一幅页头插画

1. 新建 `_includes/illo/my-illo.html`，`viewBox="0 0 300 240"`，参照现有文件的结构
2. 在页面或包装布局的 Front Matter 写 `illo: my-illo`
3. 浅色、深色模式各检查一次

---

## 10.9 BibTeX 弹窗

首页论文的 “bibtex” 按钮带有 `data-bib="/assets/bibtex/xxx.bib"`。点击后 `site.js` 用 `fetch` 读取文件内容，显示在 `<dialog id="bib-dialog">` 中，可以一键复制。弹窗结构在 `_includes/publication-list.html`。

---

## 10.10 回到顶部

页面向下滚动超过 600px 时，右下角出现回到顶部按钮。按钮在 `base.html`，逻辑在 `site.js`。

---

## 10.11 论文库筛选

`learning/paper-reading.html` 中每张论文卡片带有 `data-search`、`data-status`、`data-relevance` 属性，`site.js` 根据搜索框和两个下拉框实时隐藏不匹配的卡片，并更新数量。

---

## 10.12 维护文档页

维护文档页的地址是 `/maintenance.html`，站内没有链接指向它，需要时直接输入地址访问。这个页面是独立的 HTML（`layout: null`），会从 GitHub 读取 `main` 分支上的 `docs/*.md` 并渲染，所以**文档要推送到 `main` 后才会在这个页面上更新**。新增或改名文档时，需要同步修改 `maintenance.html` 里的卡片和 `docTitles` 映射。

---

## 10.13 暗房 Darkroom

`/darkroom/` 是一个纯前端的拍立得相纸工具：上传照片（可批量），套进 Anthropic 风格的相纸，加插画、装饰物件、胶片效果、涂鸦和文字，再导出 PNG/JPG、ZIP、拼图、A4 打印 PDF 或显影动画视频。照片只在浏览器本地处理，不会上传。

**入口**：和其他隐藏页面一样放在 Back room 面板里（连点橙色圆点 5 次，或键入 `mootutu`），导航和站内搜索都没有它；页面 front matter 里的 `noindex: true` 会让 `base.html` 输出 `<meta name="robots" content="noindex, nofollow">`。

| 文件 | 作用 |
|------|------|
| `darkroom/index.html` | 页面骨架（`layout: base`，带站点页眉、页脚和主题切换） |
| `assets/css/site.css` 末尾的 “Darkroom” 段 | 页面样式，全部限定在 `.dr` 容器内，用站点的颜色变量，深色模式自动适配 |
| `assets/javascripts/darkroom/illos.js` | 插画库（100×100 坐标的路径数据） |
| `assets/javascripts/darkroom/app.js` | 全部功能逻辑（绘制、导出、拼图、PDF、视频、模板、撤销） |

- 工具脚本较大（约 100 KB），只在这个页面加载，所以没有并进 `site.js`。
- 相纸上用到的字体（Newsreader、Caveat、Noto Serif SC 等）只在这个页面以非阻塞方式加载；3.5 秒内 Google Fonts 加载不到时会改用 `fonts.loli.net` 镜像。
- 自己保存的模板存在浏览器的 `localStorage`（键名 `instant-frame-templates-v1`），换设备不会同步，可以用「导出 .json」备份。
- 新增插画：在 `illos.js` 的 `ILLOS` 数组里加一项（`p` 是若干路径，`f` 是填充色键 C/K/O/S/M/W/H/I），再把 id 写进文件底部 `ILLO_CATS` 的对应分类。

---

<div class="doc-nav">
  <a href="./09-config.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 配置文件</span>
  </a>
  <a href="./11-deployment.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">部署与发布 →</span>
  </a>
</div>
