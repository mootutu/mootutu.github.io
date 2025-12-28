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
    *   `lang`: 英文填 `en`，中文填 `zh-CN`。
    *   `translate_url`: 对应翻译版本的链接（例如：`/cn/blog/posts/my-new-post.html`）。
2.  **编写内容**：在第二个 `---` 下方编写 HTML 内容。

### 步骤 3：更新索引页 (`blog.html`)
由于博客索引页是手动维护的，你必须将新文章添加到 `blog.html`（中文版对应 `cn/blog.html`）中。

1.  打开 `blog.html`。
2.  找到 `.blog-posts` div。
3.  在列表顶部添加一个新的 `<article>` 块：
    ```html
    <article class="blog-post">
      <div class="blog-post-content">
        <h2 class="blog-post-title"><a href="/blog/posts/your-filename.html">文章标题</a></h2>
        <div class="blog-post-meta">
          <span>发布于: 年 月 日</span>
          <span>阅读时间: X 分钟</span>
        </div>
        <p class="blog-post-excerpt">
          文章内容的简短摘要...
        </p>
        <div class="blog-post-tags">
          <a href="#" class="blog-tag">标签1</a>
          <a href="#" class="blog-tag">标签2</a>
        </div>
        <a href="/blog/posts/your-filename.html" class="read-more">阅读更多</a>
      </div>
      <div class="blog-post-image">
        <!-- 请确保已将封面图片添加到 /assets/images/covers/ 目录 -->
        <img src="/assets/images/covers/your-cover-image.png" alt="文章标题">
      </div>
    </article>
    ```

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
    *   `topic_url`: 主题索引页的链接（例如：`/learning/minimind.html`）。
    *   `translate_url`: 对应翻译版本的链接。
    *   `mathjax`: 如果使用数学公式，请设为 `true`。
2.  **编写内容**：使用 Markdown 语法编写。

### 步骤 3：更新主题索引页
你必须在相应的主题 HTML 文件中添加新文章的链接。

1.  打开主题文件（例如：`learning/minimind.html` 或 `cn/learning/minimind.html`）。
2.  找到包含 `class="learning-entries"` 的容器。
3.  添加一个新链接：
    ```html
    <a href="/learning/minimind/your-filename.html" class="entry-link">文章标题</a>
    ```

## 3. 本地预览
在本地运行网站以验证更改：
```bash
bundle exec jekyll serve
```
在浏览器中打开 [http://localhost:4000](http://localhost:4000)。
