# 个人信息维护

## 7.1 首页文字

首页内容全部在 `index.html` 里，从上到下是：

| 区块 | 标记 | 内容 |
|------|------|------|
| 首屏 | `<section class="hero">` | 小字 `eyebrow`、英文名 + 中文名、一句研究方向（`hero__lede`）、学校与导师（`hero__meta`）、联系按钮（`contact-buttons.html`）、右侧主插画 |
| 研究方向卡片 | `<div class="focus">` | 三张卡片，每张有一幅小插画、标题和一句说明 |
| 01 Publications | `<section id="publications">` | 自动从 `_data/publications.yml` 生成，见 [7.2](#72-添加论文) |
| 02 Experience | `<section id="experience">` | 分 “Education” 和 “Research Internships” 两组，每段经历一个 `<li class="row">` |
| 03 Awards | `<section id="awards">` | 每个奖项一个 `<li class="row">` |
| 04 Service & Teaching | `<section id="service">` | 审稿、助教等 |

每个编号区块左侧标题下方都有一幅小插画，分别是 `_includes/illo/section-papers.html`、`section-experience.html`、`section-awards.html`、`section-service.html`（窄屏下隐藏）。

页脚的“Last updated”是构建时间，自动更新，不用手改。

### 修改研究方向一句话

```html
<p class="hero__lede">
  I study how large language models
  <span class="scribble">reason<svg …><path d="M2 7 C 25 2, 55 2, 98 5"/></svg></span>,
  how they represent meaning, and when we can trust what they say.
</p>
```

`<span class="scribble">` 包住的那个词下面会画出一条手写下划线。换词时保留 `<svg>` 部分不动即可。

### 增加一行经历 / 奖项

```html
<li class="row" data-reveal>
  <span class="row__aside">Nov 2026 – Present</span>  <!-- 左侧：起止年月，或机构简称 -->
  <div>
    <div class="row__title">Best Paper Award</div>
    <div class="row__detail">Some Conference</div>  <!-- 可选：补充说明 -->
    <span class="row__highlight">Top 1%</span>      <!-- 可选：灰底高亮 -->
  </div>
</li>
```

`data-reveal` 让这一行在滚动到时渐入，保留即可。

Experience 区块里的日期统一写成 `Mon YYYY – Mon YYYY`（英文月份缩写、en dash `–`），正在进行的写 `Present`。要新增一组（例如 “Industry Experience”），在 `<ul class="rows">` 前加：

```html
<h3 class="rows__group">Industry Experience</h3>
```

### 研究方向卡片

每张卡片是一个 `<article class="focus-card">`，上半部分是 `focus-card__art` 里的插画，下半部分是 `focus-card__body` 里的标题和说明。只改文字时不需要动插画；插画画法见 [10.8](./10-features.md#108-插画)。

---

## 7.2 添加论文

### 步骤 1：BibTeX

把文件放到 `assets/bibtex/`，命名规则见 [6.7](./06-assets.md#67-bibtex)。

### 步骤 2：`_data/publications.yml`

按时间从新到旧排列，新论文加在最上面：

```yaml
- id: wang2026example
  illo: example
  title: "Paper Title Ending With a Period."
  authors: "<b>Weiqin Wang</b>, Coauthor A, and Hui Huang."
  venue: "In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (<b>ACL</b>), 2026"
  links:
    - label: paper
      url: https://aclanthology.org/2026.acl-long.1.pdf
    - label: code
      url: https://github.com/szu-tera/Example
  bib: /assets/bibtex/wang2026example.bib
```

| 字段 | 说明 |
|------|------|
| `id` | 与 BibTeX 文件名一致 |
| `illo` | 论文配图名，对应 `_includes/illo/pub/{名字}.html`；不填则用通用的 `default.html`（一张纸） |
| `title` | 标题；末尾的句号会在页面上自动去掉 |
| `authors` | 作者，自己的名字用 `<b>…</b>` 加粗；共同一作加 `*`，通讯作者加 `#` |
| `venue` | 完整出处。**`<b>…</b>` 里的文字会成为插画左上角的会议标签**（如 ACL、EMNLP、ACL Findings）；没有 `<b>` 时标签显示为灰色的 “Preprint” |
| `venue` 的最后 4 个字符 | 会被当作年份显示在插画右下角，所以必须以年份结尾 |
| `links` | 链接按钮；第一个链接同时作为标题的跳转地址。论文链接（PDF 或 arXiv）统一用 `paper`，代码用 `code` |
| `bib` | BibTeX 路径；有这个字段才会显示 “bibtex” 按钮 |

### 步骤 3（可选）：论文配图

每篇论文左侧有一张小插画，放在 `_includes/illo/pub/`，现有 6 张：

| 文件 | 论文 | 画面 |
|------|------|------|
| `probability.html` | Probability is Not Enough | token 链分叉成多个候选，旁边计数 |
| `dysem.html` | DySem | 两段文本之间的动态语义分量柱状图 |
| `sempa.html` | SemPA | 向量空间里偏好 / 非偏好两个方向 |
| `beyond.html` | Beyond Majority Voting | 箭靶，表示更精细的奖励信号 |
| `anchormem.html` | AnchorMem | 船锚牵着几张事实卡片 |
| `ranked.html` | Ranked Voting | 1/2/3 名次领奖台 |
| `default.html` | （通用） | 一张纸 |

新论文画新图时，复制一个现有文件改：

- `viewBox="0 0 160 120"`，第一个 `<path>` 是背景色块（`f-sand`、`f-olive`、`f-sky` 轮换使用）
- 线条用 `class="ink"`（细线加 `ink-thin`），填色用 `style="fill:var(--illo-clay)"` 这类变量，深色模式会自动适配
- 线条放在 `<g class="hand">` 里，获得手绘抖动效果
- 想要鼠标悬停时动一下的部分，包在 `<g class="wiggle wiggle--up">` 里（还有 `--left`、`--right`、`--spin`）

然后在 `publications.yml` 的条目里写 `illo: 文件名`（不带 `.html`）。

### 步骤 4：检查

本地打开首页，确认：配图显示正常、会议标签和年份正确、自己的名字加粗、各链接可打开、BibTeX 弹窗能显示内容。

同时记得更新简历（[7.4](#74-更新简历)）。

---

## 7.3 About 页面

`about.html`：

- Front Matter 的 `heading` 是页头大标题（“Hi, I'm Weiqin.”），`description` 是标题下的引言
- 正文写在 `<div class="prose about-prose">` 里
- 底部是联系按钮，与首页共用 `_includes/contact-buttons.html`

---

## 7.4 更新简历

| 简历 | 源文件 | 生成文件 | 编译引擎 |
|------|--------|---------|---------|
| 英文 | `cv/my_cv.tex` | `cv/my_cv.pdf` | pdfLaTeX |
| 中文 | `cv_zh/cv_zh.tex` | `cv_zh/cv_zh.pdf` | XeLaTeX（ctex） |

两个目录里都有 `latexmkrc`，直接在目录中运行即可：

```bash
cd cv && latexmk my_cv.tex
cd ../cv_zh && latexmk cv_zh.tex
```

编译产生的 `.aux`、`.log` 等文件已在 `.gitignore` 中忽略。提交时把 `.tex` 和 `.pdf` 一起提交。

简历在这些地方被链接：

| 位置 | 链接 |
|------|------|
| 首页、About 页的 CV 按钮 | `/cv/my_cv.pdf` |

中文简历 `cv_zh.pdf` 目前不在站内链接，需要时可以直接把 `/cv_zh/cv_zh.pdf` 发给别人。

根目录的 `resume.pdf` 是旧文件，目前没有页面链接到它。

---

## 7.5 联系方式

首页首屏和 About 页底部的按钮都来自同一个文件 `_includes/contact-buttons.html`，按顺序是：

| 按钮 | 链接 |
|------|------|
| Email | `mailto:here1swqw@gmail.com` |
| Google Scholar | `https://scholar.google.com/citations?user=JonSUasAAAAJ` |
| LinkedIn | `https://www.linkedin.com/in/weiqin-wang-aa8800392/` |
| CV | `/cv/my_cv.pdf` |

改链接、调整顺序或新增按钮（例如 GitHub）都只改这一个文件。每个按钮里是一个 16×16 的线条图标 + 文字：

```html
<a class="button" href="https://github.com/xxx" target="_blank" rel="noopener">
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">…</svg>
  GitHub
</a>
```

第一个按钮带 `button--primary`（深色实心），其余是描边按钮。

---

## 7.6 网站图标

网站图标是首页推理树插画的缩小版：米白圆角底、黑色树枝、一个空心节点和一个陶土橙节点。共三个文件：

| 文件 | 用途 |
|------|------|
| `assets/images/favicon.svg` | 现代浏览器使用的矢量图标（源文件，改图标先改它） |
| `favicon.ico`（仓库根目录） | 老浏览器和直接请求 `/favicon.ico` 时使用，内含 16/32/48 像素 |
| `assets/images/apple-touch-icon.png` | iPhone/iPad “添加到主屏幕”时的 180×180 图标（方形满铺，系统自动切圆角） |

在 `_layouts/base.html` 和 `maintenance.html` 的 `<head>` 中引用：

```html
<link rel="icon" href="/favicon.ico" sizes="32x32">
<link rel="icon" href="/assets/images/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/assets/images/apple-touch-icon.png">
```

`.ico` 和 `.png` 是由 SVG 的同一套图形用 Python（Pillow）画出来的。修改图标时，先改 SVG，再按同样的图形重新生成另外两个文件，三者保持一致。浏览器会缓存图标，替换后可能要清缓存或过一段时间才看到新图标。

---

## 7.7 检查清单

- [ ] 首页文字、经历、奖项已更新
- [ ] 新论文的会议标签、年份、加粗、链接、BibTeX 都正确
- [ ] 中英文简历已重新编译并提交 PDF
- [ ] `contact-buttons.html` 中的链接都能打开

---

<div class="doc-nav">
  <a href="./06-assets.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 静态资源管理</span>
  </a>
  <a href="./08-layouts.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">布局与组件 →</span>
  </a>
</div>
