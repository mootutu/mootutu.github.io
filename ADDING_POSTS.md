# Adding New Posts Guide

This guide outlines the steps to add new Blog Posts and Learning Posts to the website.

## 1. Adding a Blog Post

### Step 1: Create the Post File
1.  Navigate to `_templates/blog_post_template.html`.
2.  Copy the content.
3.  Create a new HTML file in `blog/posts/` (e.g., `blog/posts/my-new-post.html`).
    *   *For Chinese version:* Create it in `cn/blog/posts/`.
4.  Paste the template content into your new file.

### Step 2: Edit Content
1.  **Update Front Matter**:
    *   `title`: The title of your post.
    *   `date`: YYYY-MM-DD.
    *   `author`: Author name.
    *   `category`: Category label.
    *   `excerpt`: Short summary for the blog index.
    *   `tags`: List of tags.
    *   `reading_time`: Reading time label (e.g., `6 minutes`).
    *   `cover_image`: Path under `/assets/images/covers/`.
    *   `lang`: `en` or `zh-CN`.
    *   `translate_url`: Link to the translated version (e.g., `/cn/blog/posts/my-new-post.html`).
2.  **Write Content**: Write your HTML content below the second `---`.

### Step 3: Blog Index Auto-Update
The blog index is generated automatically from front matter. No manual edits are needed for `blog.html` or `cn/blog.html`.

---

## 2. Adding a Learning Post

### Step 1: Create the Post File
1.  Navigate to `_templates/learning_post_template.md`.
2.  Copy the content.
3.  Identify the topic folder under `learning/` (e.g., `learning/minimind/`).
4.  Create a new Markdown (`.md`) file in that folder (e.g., `learning/minimind/step3.md`).
    *   *For Chinese version:* Create it in `cn/learning/minimind/`.
5.  Paste the template content.

### Step 2: Edit Content
1.  **Update Front Matter**:
    *   `title`: Post title.
    *   `date`: YYYY-MM-DD.
    *   `lang`: `en` or `zh-CN`.
    *   `topic`: Topic key (e.g., `minimind`).
    *   `order`: Numeric order for display.
    *   `topic_url`: Link to the topic index (e.g., `/learning/minimind.html`).
    *   `translate_url`: Link to translated version.
    *   `mathjax`: Set to `true` if you use math.
2.  **Write Content**: use Markdown.

### Step 3: Topic Index Auto-Update
Topic pages render their entry lists automatically based on `topic` and `order`.

## 3. Preview locally
Run the site locally to verify your changes:
```bash
bundle exec jekyll serve
```
Open [http://localhost:4000](http://localhost:4000) in your browser.
