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
    *   `lang`: `en` or `zh-CN`.
    *   `translate_url`: Link to the translated version (e.g., `/cn/blog/posts/my-new-post.html`).
2.  **Write Content**: Write your HTML content below the second `---`.

### Step 3: Update Index Page (`blog.html`)
Since the blog index is manually maintained, you must add your post to `blog.html` (and `cn/blog.html`).

1.  Open `blog.html`.
2.  Find the `.blog-posts` div.
3.  Add a new `<article>` block at the top of the list:
    ```html
    <article class="blog-post">
      <div class="blog-post-content">
        <h2 class="blog-post-title"><a href="/blog/posts/your-filename.html">Your Post Title</a></h2>
        <div class="blog-post-meta">
          <span>Published: Month Day, Year</span>
          <span>Reading Time: X minutes</span>
        </div>
        <p class="blog-post-excerpt">
          Brief summary of your post...
        </p>
        <div class="blog-post-tags">
          <a href="#" class="blog-tag">Tag1</a>
          <a href="#" class="blog-tag">Tag2</a>
        </div>
        <a href="/blog/posts/your-filename.html" class="read-more">Read More</a>
      </div>
      <div class="blog-post-image">
        <!-- Make sure to add a cover image to /assets/images/covers/ -->
        <img src="/assets/images/covers/your-cover-image.png" alt="Post Title">
      </div>
    </article>
    ```

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
    *   `topic_url`: Link to the topic index (e.g., `/learning/minimind.html`).
    *   `translate_url`: Link to translated version.
    *   `mathjax`: Set to `true` if you use math.
2.  **Write Content**: use Markdown.

### Step 3: Update Topic Index
You must add a link to the new post in the corresponding topic HTML file.

1.  Open the topic file (e.g., `learning/minimind.html` or `cn/learning/minimind.html`).
2.  Find the container with `class="learning-entries"`.
3.  Add a new link:
    ```html
    <a href="/learning/minimind/your-filename.html" class="entry-link">Your Post Title</a>
    ```

## 3. Preview locally
Run the site locally to verify your changes:
```bash
bundle exec jekyll serve
```
Open [http://localhost:4000](http://localhost:4000) in your browser.
