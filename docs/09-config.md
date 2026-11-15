# 配置文件

## 9.1 概览

| 文件 | 作用 |
|------|------|
| `_config.yml` | Jekyll 构建设置 |
| `_data/publications.yml` | 首页论文列表，格式见 [7.2](./07-personal-info.md#72-添加论文) |
| `_data/ui_text.yml` | 导航栏文字，见 [5.2](./05-bilingual.md#52-界面文字_dataui_textyml) |
| `Gemfile` / `Gemfile.lock` | Ruby 依赖（`github-pages`、`webrick`） |
| `.gitignore` | 忽略 `_site/`、`.sass-cache/`、LaTeX 编译中间文件等 |

---

## 9.2 `_config.yml`

当前内容：

```yaml
# Build settings
markdown: kramdown
theme: minima
plugins:
  - jekyll-feed

# Exclude docs directory from Jekyll processing
exclude:
  - docs
  - README.md
  - MAINTENANCE.md
  - ADDING_POSTS.md
```

| 配置 | 说明 |
|------|------|
| `markdown: kramdown` | Markdown 解析器 |
| `theme: minima` | 历史遗留。站点所有页面都用自己的布局，实际上没有用到 minima 的模板和样式，但删掉需要同时确认 `github-pages` 构建不报错，暂时保留 |
| `plugins: jekyll-feed` | 生成 `/feed.xml`（只包含 `_posts`，本站没有使用 `_posts`，所以 feed 基本为空） |
| `exclude` | 不参与构建、不发布的文件 |

### 修改后的注意事项

- `jekyll serve` 不会重新读取 `_config.yml`，改完需要停掉服务（Ctrl+C）再重新启动
- 要在模板中使用自定义配置，加一行 `my_setting: value`，模板里用 `{{ site.my_setting }}`

---

## 9.3 `_data/` 目录

`_data/` 下的 YAML 文件会以 `site.data.文件名` 的形式在所有模板中可用：

```liquid
{% for pub in site.data.publications %}…{% endfor %}
{% assign ui = site.data.ui_text.en %}
```

新增数据文件（例如 `_data/news.yml`）后，就可以在 `index.html` 中用 `site.data.news` 循环生成内容。

---

## 9.4 Jekyll 的默认行为

- 以 `_` 开头的目录（`_layouts`、`_includes`、`_data`、`_templates`）不会被发布
- 有 Front Matter 的 `.html` / `.md` 文件会被渲染成页面；没有 Front Matter 的文件按原样复制（例如 PDF、图片、`learning/paper-reading/README.md`）
- Markdown 文件发布后的地址后缀是 `.html`

---

## 9.5 修改配置的流程

1. 修改前先确认本地能正常构建
2. 修改后重启 `jekyll serve`
3. 检查主要页面和终端里有没有 `Liquid Exception` 或 `Error`
4. 单独提交，提交信息写清楚改了什么（`chore: …`）

---

<div class="doc-nav">
  <a href="./08-layouts.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 布局与组件</span>
  </a>
  <a href="./10-features.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">特殊功能 →</span>
  </a>
</div>
