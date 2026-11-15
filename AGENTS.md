# Repository Guidelines

## Project Structure & Module Organization
This is an English-only Jekyll site for GitHub Pages. Every page renders through `_layouts/base.html`; list/intro pages use `_layouts/page.html` and articles use `_layouts/post.html`, while the older layout names (`blog-post-layout`, `learning-topic-layout`, etc.) are thin wrappers that set per-section defaults. Shared components live in `_includes/` (header, footer, list components, and SVG illustrations under `_includes/illo/`), with shared data in `_data/`. All styling is in `assets/css/site.css` (design tokens at the top) and all behavior in `assets/javascripts/site.js` plus `theme-toggle.js`. Content is organized by section: `blog/` for blog posts and `learning/` for learning topics and paper reading notes. Maintenance guides live in `docs/`, while templates for new content are in `_templates/`.

## Build, Test, and Development Commands
- `bundle exec jekyll serve`: Build and run the site locally at `http://localhost:4000`.
- `jekyll serve --drafts`: Include draft posts during preview.
- `jekyll serve --livereload`: Reload pages automatically on changes.

## Coding Style & Naming Conventions
- Indentation: 2 spaces for HTML/CSS/JS in templates and content files.
- Front Matter: use YAML keys like `title`, `date` (YYYY-MM-DD), and `lang: en`.
- File naming: use lowercase with hyphens (e.g., `blog/posts/my-new-post.html`).
- Prefer shared layout classes and components via `_includes/` to avoid duplication.
- Do not add page-level `<style>` blocks; extend `assets/css/site.css` and use its CSS variables so dark mode keeps working.

## Testing Guidelines
No automated test suite is configured. Validate changes by running `bundle exec jekyll serve` and checking key pages (home, blog, learning topics, and Paper Reading). For content updates, verify the index pages reflect new entries.

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
