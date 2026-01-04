# Repository Guidelines

## Project Structure & Module Organization
This is a Jekyll site for GitHub Pages. Core templates live in `_layouts/` and `_includes/`, with shared data in `_data/`. Static assets are under `assets/` (CSS, JS, images, fonts, and BibTeX files). Content is organized by section: `blog/` for English blog posts, `learning/` for English learning topics, and `cn/` for the Chinese mirror (same structure). Maintenance guides live in `docs/`, while templates for new content are in `_templates/`.

## Build, Test, and Development Commands
- `bundle exec jekyll serve`: Build and run the site locally at `http://localhost:4000`.
- `jekyll serve --drafts`: Include draft posts during preview.
- `jekyll serve --livereload`: Reload pages automatically on changes.

## Coding Style & Naming Conventions
- Indentation: 2 spaces for HTML/CSS/JS in templates and content files.
- Front Matter: use YAML keys like `title`, `date` (YYYY-MM-DD), `lang`, and `translate_url`.
- File naming: use lowercase with hyphens (e.g., `blog/posts/my-new-post.html`).
- Prefer shared layout classes and components via `_includes/` to avoid duplication across English/Chinese pages.

## Testing Guidelines
No automated test suite is configured. Validate changes by running `bundle exec jekyll serve` and checking key pages (home, blog, learning topics, CN mirror). For content updates, verify the index pages reflect new entries.

## Commit & Pull Request Guidelines
Use semantic commit messages as documented in `docs/11-deployment.md`, e.g.:
- `feat: add new blog post about X`
- `fix: correct typo in about page`
- `docs: update maintenance guide`
If you open a PR, include a short summary, affected pages, and screenshots for layout changes.

## Content Maintenance Notes
- Blog posts: create new files from `_templates/blog_post_template.html` and fill in `excerpt`, `tags`, `reading_time`, and `cover_image` (indexes auto-generate).
- Learning posts: create Markdown from `_templates/learning_post_template.md` and set `topic`/`order` (topic pages auto-generate).
- Publications: update `_data/publications.yml` and add BibTeX files under `assets/bibtex/` (homepages render via shared include).
