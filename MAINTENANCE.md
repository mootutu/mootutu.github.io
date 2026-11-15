# 个人主页维护指南

完整的维护文档在 [`docs/`](docs/README.md)，这里只列最常用的入口。

| 想做的事 | 去改哪里 | 文档 |
|---------|---------|------|
| 修改首页介绍、教育、奖项、服务 | `index.html` | [07 个人信息维护](docs/07-personal-info.md) |
| 添加论文 | `_data/publications.yml` + `assets/bibtex/` | [7.2 添加论文](docs/07-personal-info.md#72-添加论文) |
| 更新简历 | `cv/my_cv.tex`、`cv_zh/cv_zh.tex` | [7.4 更新简历](docs/07-personal-info.md#74-更新简历) |
| 写博客 | `blog/posts/` | [03 博客文章维护](docs/03-blog-maintenance.md) |
| 写学习笔记 / 论文笔记 | `learning/` | [04 学习内容维护](docs/04-learning-maintenance.md) |
| 改样式、颜色、字体 | `assets/css/site.css` | [06 静态资源管理](docs/06-assets.md) |
| 改页面结构 | `_layouts/`、`_includes/` | [08 布局与组件](docs/08-layouts.md) |

本地预览：

```bash
bundle exec jekyll serve --livereload
```

然后访问 `http://localhost:4000`。
