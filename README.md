# mootutu.github.io

Source for my personal academic website, built with Jekyll and hosted on GitHub Pages: <https://mootutu.github.io>.

- **Homepage** (`index.html`) — introduction, research focus, publications, education, awards
- **Blog** (`blog/posts/`) and **Learning notes** (`learning/`), including a searchable paper-reading library
- **CV** — English (`cv/`) and Chinese (`cv_zh/`) LaTeX sources and PDFs

Publications are listed in `_data/publications.yml`, with BibTeX files under `assets/bibtex/`.

## Local preview

```bash
bundle install
bundle exec jekyll serve --livereload
```

Then open <http://localhost:4000>. The `github-pages` gem needs Ruby 3.x; see `docs/02-quickstart.md` for a workaround on Ruby 4.

## Documentation

The maintenance guide (in Chinese) lives in [`docs/`](docs/README.md) and is also readable on the site at `/maintenance.html`.

## License

The original theme was based on [orderedlist/minimal](https://github.com/orderedlist/minimal), licensed under [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/).
