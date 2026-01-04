# 添加新文章指南

本指南概述了向网站添加新博客文章（Blog Post）和学习文章（Learning Post）的步骤。

## 1. 添加博客文章 (Blog Post)

### 步骤 1：创建文章文件
1.  找到模板文件 `_templates/blog_post_template.html`。
2.  复制其中的内容。
3.  在 `blog/posts/` 目录下创建一个新的 HTML 文件（例如：`blog/posts/my-new-post.html`）。
    *   *中文版：* 请在 `cn/blog/posts/` 目录下创建。
4.  将模板内容粘贴到新文件中。

### 步骤 2：编辑内容
1.  **更新 Front Matter（文件头部信息）**：
    *   `title`: 文章标题。
    *   `date`: 日期，格式 YYYY-MM-DD。
    *   `author`: 作者姓名。
    *   `category`: 分类。
    *   `excerpt`: 博客列表摘要。
    *   `tags`: 标签列表。
    *   `reading_time`: 阅读时长（例如 `6 分钟`）。
    *   `cover_image`: 封面图路径（位于 `/assets/images/covers/`）。
    *   `lang`: 英文填 `en`，中文填 `zh-CN`。
    *   `translate_url`: 对应翻译版本的链接（例如：`/cn/blog/posts/my-new-post.html`）。
2.  **编写内容**：在第二个 `---` 下方编写 HTML 内容。

### 步骤 3：索引自动更新
博客索引页会根据 Front Matter 自动生成，无需手动编辑 `blog.html` 或 `cn/blog.html`。

---

## 2. 添加学习文章 (Learning Post)

### 步骤 1：创建文章文件
1.  找到模板文件 `_templates/learning_post_template.md`。
2.  复制其中的内容。
3.  确定该文章所属的主题文件夹，位于 `learning/` 下（例如：`learning/minimind/`）。
4.  在该文件夹中创建一个新的 Markdown (`.md`) 文件（例如：`learning/minimind/step3.md`）。
    *   *中文版：* 请在 `cn/learning/minimind/` 目录下创建。
5.  粘贴模板内容。

### 步骤 2：编辑内容
1.  **更新 Front Matter**：
    *   `title`: 文章标题。
    *   `date`: 日期 YYYY-MM-DD。
    *   `lang`: `en` 或 `zh-CN`。
    *   `topic`: 主题 key（如 `minimind`）。
    *   `order`: 显示顺序（数字）。
    *   `topic_url`: 主题索引页的链接（例如：`/learning/minimind.html`）。
    *   `translate_url`: 对应翻译版本的链接。
    *   `mathjax`: 如果使用数学公式，请设为 `true`。
2.  **编写内容**：使用 Markdown 语法编写。

### 步骤 3：主题页自动更新
主题页会根据 `topic` 和 `order` 自动生成文章列表，无需手动添加链接。

## 3. 本地预览
在本地运行网站以验证更改：
```bash
bundle exec jekyll serve
```
在浏览器中打开 [http://localhost:4000](http://localhost:4000)。
