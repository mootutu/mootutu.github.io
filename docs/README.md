# 仓库维护文档

> mootutu.github.io 个人主页的维护手册，写给第一次接手这个仓库的人。
>
> 站点只有英文版；2026-07-16 起已移除 `/cn/` 中文镜像和语言切换。2026-11 起全站换成了新的设计（暖色米白配色、衬线标题、手绘风插画）。

---

## 文档导航

### 基础入门

| 文档 | 说明 |
|------|------|
| [01-项目概述](./01-overview.md) | 站点包含什么、技术栈、目录结构 |
| [02-快速开始](./02-quickstart.md) | 本地预览、最常用的几个操作 |

### 内容维护

| 文档 | 说明 |
|------|------|
| [03-博客文章维护](./03-blog-maintenance.md) | 新增、修改、删除博客文章 |
| [04-学习内容维护](./04-learning-maintenance.md) | 学习主题、学习笔记、论文阅读笔记 |
| [05-站点语言与界面文字](./05-bilingual.md) | 英文单语约定、`ui_text.yml` |

### 资源与信息

| 文档 | 说明 |
|------|------|
| [06-静态资源管理](./06-assets.md) | 图片、CSS、JS、字体、插画 |
| [07-个人信息维护](./07-personal-info.md) | 首页文字、论文、奖项、简历、图标 |

### 技术参考

| 文档 | 说明 |
|------|------|
| [08-布局与组件](./08-layouts.md) | `_layouts/` 与 `_includes/` 的结构 |
| [09-配置文件](./09-config.md) | `_config.yml`、`_data/` |
| [10-特殊功能](./10-features.md) | 搜索、深色模式、目录、动效、插画等 |

### 部署与规范

| 文档 | 说明 |
|------|------|
| [11-部署与发布](./11-deployment.md) | GitHub Pages 发布流程、提交信息规范 |
| [12-命名规范汇总](./12-naming.md) | 文件、目录、Front Matter、CSS 类名 |
| [13-维护检查清单](./13-checklist.md) | 新增内容、提交、发布前后的检查项 |
| [14-常见问题FAQ](./14-faq.md) | 常见问题与解决方法 |

### 附录

| 文档 | 说明 |
|------|------|
| [附录-模板文件](./appendix/templates.md) | 博客、学习笔记、论文笔记、论文条目模板 |
| [附录-速查表](./appendix/cheatsheet.md) | Front Matter、路径、命令速查 |

---

## 最常用的操作

- 新增博客文章 → [03-博客文章维护](./03-blog-maintenance.md#35-新增博客文章)
- 新增学习笔记 → [04-学习内容维护](./04-learning-maintenance.md#44-新增学习笔记)
- 新增论文阅读笔记 → [04-学习内容维护](./04-learning-maintenance.md#45-论文阅读笔记)
- 添加一篇论文 → [07-个人信息维护](./07-personal-info.md#72-添加论文)
- 修改首页介绍 → [07-个人信息维护](./07-personal-info.md#71-首页文字)

---

## 技术栈速览

| 技术 | 用途 |
|------|------|
| Jekyll 3.10（GitHub Pages 内置） | 静态站点生成 |
| Liquid | 模板语言 |
| Kramdown | Markdown 解析 |
| Prism.js 1.29 | 代码高亮（仅文章页加载） |
| MathJax 3.2 | 数学公式（文章设置 `mathjax: true` 时加载） |
| Google Fonts | Source Serif 4、Inter、JetBrains Mono |
| GitHub Pages | 推送 `main` 后自动构建发布 |

---

## 仓库结构速览

```
mootutu.github.io/
├── _layouts/          # 页面骨架：base / page / post 及其包装层
├── _includes/         # 页头、页脚、列表组件、插画
├── _data/             # 论文列表、界面文字
├── _templates/        # 新建内容时复制的模板
├── assets/            # CSS、JS、图片、BibTeX
├── blog/posts/        # 博客文章
├── learning/          # 学习主题与笔记
├── cv/  cv_zh/        # 英文 / 中文 LaTeX 简历
├── docs/              # 本维护文档
├── index.html         # 首页
├── blog.html          # 博客列表
├── learning.html      # 学习主题列表
├── about.html         # About
├── search.html        # 生成 /search.json
└── maintenance.html   # 维护文档阅读页（站内无入口，直接访问）
```

---

*最后更新：2026-11-15*
