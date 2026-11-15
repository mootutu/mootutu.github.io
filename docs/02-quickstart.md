# 快速开始

## 2.1 本地预览

### 推荐做法（Ruby 3.x）

仓库的 `Gemfile` 使用 `github-pages` gem，保证本地和线上的 Jekyll 版本一致。它需要 **Ruby 3.x**。

```bash
git clone https://github.com/mootutu/mootutu.github.io.git
cd mootutu.github.io
bundle install
bundle exec jekyll serve --livereload
```

然后访问 `http://localhost:4000`。修改文件后页面会自动刷新；改了 `_config.yml` 需要重启服务。

### 如果本机是 Ruby 4

`github-pages` 依赖的 `posix-spawn` 在 Ruby 4 上编译失败，`bundle install` 会报错。可以在仓库**外面**单独建一个只装 Jekyll 3.10 的 Gemfile，用它来预览（不影响线上部署）：

```bash
mkdir -p ~/jekyll-preview && cd ~/jekyll-preview
cat > Gemfile <<'EOF'
source "https://rubygems.org"
gem "jekyll", "~> 3.10"
gem "jekyll-feed"
gem "minima", "~> 2.5"
gem "kramdown-parser-gfm"
gem "webrick"
gem "csv"
gem "base64"
gem "bigdecimal"
gem "logger"
EOF
BUNDLE_PATH=vendor bundle install
```

之后在仓库目录里运行：

```bash
BUNDLE_GEMFILE=~/jekyll-preview/Gemfile BUNDLE_PATH=~/jekyll-preview/vendor bundle exec jekyll serve --livereload
```

构建时会在仓库里生成 `.sass-cache/`，它已经写进 `.gitignore`。

### 常用命令

| 命令 | 说明 |
|------|------|
| `bundle exec jekyll serve` | 启动本地服务 |
| `bundle exec jekyll serve --livereload` | 修改后浏览器自动刷新 |
| `bundle exec jekyll serve --drafts` | 同时显示草稿 |
| `bundle exec jekyll build` | 构建到 `_site/` |
| `bundle exec jekyll clean` | 清理构建缓存 |

---

## 2.2 最常用的操作

### 新增博客文章

```bash
cp _templates/blog_post_template.html blog/posts/my-new-post.html
# 1. 填写 Front Matter：title、date、category、excerpt、tags、reading_time、cover_image
# 2. 在第二个 --- 下面用 HTML 写正文
# 3. 封面图放到 assets/images/covers/my-new-post.png
```

博客列表会自动出现这篇文章。详见 [03-博客文章维护](./03-blog-maintenance.md)。

### 新增学习笔记

```bash
cp _templates/learning_post_template.md learning/minimind/part5.md
# 填写 Front Matter：title、date、topic、order、topic_url，用到公式时加 mathjax: true
```

详见 [04-学习内容维护](./04-learning-maintenance.md)。

### 新增论文阅读笔记

```bash
cp _templates/paper_reading_template.md learning/paper-reading/2603.12345-short-title.md
```

### 添加论文

1. BibTeX 放到 `assets/bibtex/作者年份关键词.bib`
2. 在 `_data/publications.yml` 加一条记录

详见 [07-个人信息维护](./07-personal-info.md#72-添加论文)。

### 修改首页介绍、教育、奖项

直接编辑 `index.html`。详见 [07-个人信息维护](./07-personal-info.md#71-首页文字)。

---

## 2.3 提交与发布

```bash
git status
git add <改动的文件>
git commit -m "feat: add blog post about X"
git push origin main
```

推送到 `main` 后 GitHub Pages 会自动构建，通常 1–3 分钟后生效。提交信息规范见 [11-部署与发布](./11-deployment.md#112-提交信息规范)。

---

## 2.4 不会自动补全的信息

| 操作 | 需要手动填写 |
|------|-------------|
| 新增博客文章 | `excerpt`、`tags`、`category`、`reading_time`、`cover_image` |
| 新增学习主题 | 主题页的 `title`、`description`、`topic`、`order` |
| 新增学习笔记 | `topic`、`order`、`topic_url` |
| 新增论文笔记 | `status`、`relevance`、`tags`、`one_line_summary` 等 |
| 新增论文 | `_data/publications.yml` 条目和 BibTeX 文件 |

---

<div class="doc-nav">
  <a href="./01-overview.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 项目概述</span>
  </a>
  <a href="./03-blog-maintenance.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">博客文章维护 →</span>
  </a>
</div>
