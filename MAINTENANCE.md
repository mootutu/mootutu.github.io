# 个人主页维护指南

本指南说明了如何维护和更新个人主页。

## 项目结构

仓库已经过重组，以保持根目录整洁。所有静态资源现在都位于 `assets/` 目录下：

- `assets/css/`: 样式表
- `assets/javascripts/`: JavaScript 文件
- `assets/images/`: 图片（包括作者照片）
- `assets/fontawesome/`: FontAwesome 图标
- `assets/bibtex/`: 论文的 BibTeX 文件

## 如何更新

### 1. 更新个人简介 (Biography)
个人简介部分分为两个包含文件：
- **英文版**: 编辑 `_includes/bio.html`
- **中文版**: 编辑 `_includes/bio-cn.html`

### 2. 添加论文 (Publications)
要添加新的论文：
1.  **添加 BibTeX**: 将 `.bib` 文件放入 `assets/bibtex/` 目录。
2.  **更新 HTML**: 在 `index.html` (英文) 和 `index-cn.html` (中文) 的 "Publications" (发表论文) 部分添加一个新的列表项 (`<li>`)。
        ```html
        [<a href="javascript:void(0);" onclick="openBibModal('assets/bibtex/YOUR_FILE.bib')">bib</a>]
        ```

### 3. 更新奖项与新闻 (Awards & News)
直接在 `index.html` 和 `index-cn.html` 中编辑 "Award" (获奖情况) 或 "News" 部分。

## 本地开发

要在本地运行网站，请确保已安装 Jekyll，然后运行：

```bash
bundle exec jekyll serve
```

访问 `http://localhost:4000` 查看网站。
