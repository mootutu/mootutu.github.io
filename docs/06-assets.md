# 静态资源管理

## 6.1 目录结构

```
assets/
├── css/
│   └── site.css              # 全站唯一的样式表
├── javascripts/
│   ├── site.js               # 全站交互
│   └── theme-toggle.js       # 深色模式切换
├── images/
│   ├── covers/               # 博客封面（与文章同名）
│   ├── posts/{文章名}/        # 博客文章配图
│   ├── minimind/             # Minimind 笔记配图
│   ├── favicon.svg           # 网站图标（矢量源文件）
│   ├── apple-touch-icon.png  # iOS 主屏幕图标
│   ├── mootutu.png           # 旧头像，目前没有页面使用
│   ├── weiqin.png            # 旧照片，目前没有页面使用
│   └── nailong.ico / .jpg    # 目前没有页面使用
├── bibtex/                   # 论文 BibTeX
└── fontawesome/              # 旧图标库，目前没有页面引用
```

标注“没有页面使用”的文件可以在确认不需要后删除，以减小仓库体积。

---

## 6.2 图片

### 分类与位置

| 类型 | 位置 | 说明 |
|------|------|------|
| 博客封面 | `assets/images/covers/文章名.png` | 16:9 显示，见 [3.8](./03-blog-maintenance.md#38-封面图规范) |
| 博客配图 | `assets/images/posts/文章名/` | 每篇一个目录 |
| 学习笔记配图 | `assets/images/{主题}/` | 如 `assets/images/minimind/` |
| 网站图标 | `assets/images/favicon.svg`、根目录 `favicon.ico`、`assets/images/apple-touch-icon.png` | 见 [7.6](./07-personal-info.md#76-网站图标) |

### 要求

- 文件名小写、用连字符；不要有空格和中文
- 照片用 JPG，截图和带透明的图用 PNG
- 单张尽量控制在 500 KB 以内
- 引用一律用 `/assets/...` 开头的绝对路径

---

## 6.3 CSS：`assets/css/site.css`

全站只有这一个样式表，由 `_layouts/base.html` 引入。页面和布局里**不要**再写 `<style>`。

文件按区块组织，搜索注释标题即可定位：

| 区块 | 内容 |
|------|------|
| `:root` / `html.dark-mode` | 设计变量：颜色、插画配色、字体、圆角、缓动曲线 |
| Base | 基础排版、链接、`.container` |
| Header | 顶部导航、图标按钮、移动端菜单 |
| Hero / Focus cards / Sections | 首页首屏、研究方向卡片、编号区块 |
| Publications / Generic rows | 论文列表、教育/奖项/服务行 |
| Footer / Dialog / Search palette | 页脚、BibTeX 弹窗、搜索面板 |
| Motion / Illustrations | 渐入动效、插画样式与动画 |
| Inner pages：Page hero / Post / Prose | 列表页页头、文章页、正文排版 |
| Blog index / Learning / Paper library | 博客卡片、主题卡片、笔记列表、论文库 |
| Back to top / MathJax | 回到顶部按钮、公式溢出处理 |

### 设计变量

改颜色、字体时只改变量，不要在各处写死颜色值：

| 变量 | 浅色 | 用途 |
|------|------|------|
| `--bg` | `#f4f1ea` | 页面底色 |
| `--surface` | `#faf8f3` | 卡片底色 |
| `--surface-strong` | `#ebe6da` | 标签、强调底色 |
| `--text` | `#1f1e1b` | 标题、主要文字 |
| `--text-soft` | `#3d3b36` | 正文 |
| `--muted` | `#6e6a62` | 次要文字 |
| `--line` | 12% 黑 | 分隔线、边框 |
| `--accent` | `#c4633f` | 陶土橙点缀色 |
| `--illo-*` | — | 插画配色：`clay`、`olive`、`sky`、`sand`、`paper`、`ink`、`panel` |
| `--font-serif` / `--font-sans` / `--font-mono` | — | 三种字体 |

`html.dark-mode` 下同名变量会被重新定义成深色值，所以只要用变量，深色模式就自动生效。

---

## 6.4 JavaScript

| 文件 | 内容 |
|------|------|
| `assets/javascripts/site.js` | 导航栏滚动边框、移动端菜单、首页导航高亮、滚动渐入、搜索面板（目前未启用）、BibTeX 弹窗、回到顶部、阅读进度条、文章目录、论文库筛选 |
| `assets/javascripts/theme-toggle.js` | 深色/浅色切换按钮（带圆形展开过渡） |
| `_includes/theme-init.html` | 内联脚本，在页面渲染前根据偏好加上 `dark-mode` 类，避免闪烁 |

`site.js` 里的每个功能都会先检查页面上有没有对应元素，没有就跳过，所以所有页面共用一个文件即可。

### 外部库（CDN）

| 库 | 地址 | 何时加载 |
|----|------|---------|
| Prism.js 1.29.0 | cdnjs | 文章页（布局设置了 `code: true`） |
| MathJax 3.2.2 | cdnjs | 页面 Front Matter 有 `mathjax: true` |
| Google Fonts | fonts.googleapis.com | 所有页面 |

### 缓存

`base.html` 引用 `site.css` 和 `site.js` 时带了 `?v=构建时间`，每次部署都会让浏览器拿到新文件，不需要手动改版本号。

---

## 6.5 字体

| 字体 | 变量 | 用途 |
|------|------|------|
| Source Serif 4 | `--font-serif` | 标题、引言、插画旁的大字 |
| Inter | `--font-sans` | 正文、导航、按钮 |
| JetBrains Mono | `--font-mono` | 代码、编号、日期 |

三种字体都从 Google Fonts 加载（`_layouts/base.html` 的 `<head>`）。后备字体中包含了中文字体（Songti SC、PingFang SC）。

---

## 6.6 图标与插画

- 界面图标（搜索、月亮/太阳、箭头等）都是直接写在模板里的内联 SVG，不依赖图标库
- 页头和首页插画是内联 SVG，放在 `_includes/illo/` 和 `index.html` 中，画法见 [10.8](./10-features.md#108-插画)

---

## 6.7 BibTeX

- 位置：`assets/bibtex/`
- 命名：`第一作者姓 + 年份 + 标题首个关键词.bib`，全小写，如 `wang2025ranked.bib`
- 与 `_data/publications.yml` 中的 `id` 和 `bib` 字段保持一致
- 首页点击 “bibtex” 会在弹窗中显示文件内容，并可一键复制

---

## 6.8 路径规范

站点部署在域名根目录，所有站内资源和链接都用以 `/` 开头的绝对路径：

```html
<img src="/assets/images/posts/git-basics/image.png" alt="">
<a href="/learning/minimind/part3.html">Part 3</a>
```

不要写 `../assets/...` 这样的相对路径，否则在不同深度的页面中会失效。

---

<div class="doc-nav">
  <a href="./05-bilingual.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 站点语言与界面文字</span>
  </a>
  <a href="./07-personal-info.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">个人信息维护 →</span>
  </a>
</div>
