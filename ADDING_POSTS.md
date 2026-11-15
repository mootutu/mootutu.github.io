# Adding New Posts

Full instructions (in Chinese) are in [`docs/03-blog-maintenance.md`](docs/03-blog-maintenance.md) and [`docs/04-learning-maintenance.md`](docs/04-learning-maintenance.md). Quick version:

## Blog post

1. `cp _templates/blog_post_template.html blog/posts/my-new-post.html`
2. Fill in the front matter: `title`, `date`, `category`, `excerpt`, `tags`, `reading_time`, `cover_image`, `lang: en`.
3. Write the body in HTML below the second `---`. Use `<h2>`/`<h3>` for sections; the table of contents is generated automatically.
4. Add a 16:9 cover image at `assets/images/covers/my-new-post.png`.

The blog index updates automatically.

## Learning note

1. `cp _templates/learning_post_template.md learning/<topic-folder>/my-note.md`
2. Fill in `title`, `date`, `lang: en`, `topic`, `order`, `topic_url`, and `mathjax: true` if you use math.
3. Write the body in Markdown.

The topic page and the Learning index update automatically.

## Paper-reading note

1. `cp _templates/paper_reading_template.md learning/paper-reading/<arxiv-id>-short-title.md`
2. Fill in the paper fields, especially `status` (`unread`/`skimming`/`read`/`reproduced`/`integrated`), `relevance` (`high`/`medium`/`low`), `tags`, and `one_line_summary`.

## Preview

```bash
bundle exec jekyll serve --livereload
```

Open <http://localhost:4000>.
