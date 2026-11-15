# 常见问题 FAQ

## 内容

### Q：新增的文章没有出现在列表里？

依次检查：

1. `layout` 是否正确：博客是 `blog-post-layout`，学习笔记是 `learning-post-layout`
2. 学习笔记的 `topic` 是否和主题页完全一致（大小写、连字符）
3. `lang` 是否写了 `en`（主题页只列出同语言的笔记）
4. Front Matter 是否以 `---` 开头和结尾、YAML 缩进是否正确
5. 本地服务是否在终端里报错

### Q：为什么 Markdown 文件的链接要写成 `.html`？

Jekyll 把 `part2.md` 构建成 `part2.html`，线上只存在 `.html` 文件。

### Q：文章右侧没有目录？

- 目录至少需要 2 个 `h2` / `h3` 标题
- 屏幕宽度小于 1080px 时目录会隐藏
- 只有文章页（博客文章、学习笔记）有目录

### Q：标题下面的引言从哪里来？

文章页依次使用 `one_line_summary`、`excerpt`；列表页和 About 使用 `description`。都没有就不显示。

### Q：阅读时长和卡片上写的不一样？

文章页按字数自动计算，卡片显示手写的 `reading_time`。以哪个为准都可以，想一致就按文章页的数字改 `reading_time`。

---

## 样式

### Q：深色模式下某个元素颜色不对？

说明它用了写死的颜色。把颜色换成 `site.css` 里的变量（`var(--text)`、`var(--surface)` 等），或者补一条 `html.dark-mode .xxx { … }`。

### Q：代码没有高亮？

- 代码要写在 `<pre><code class="language-xxx">` 里，或 Markdown 代码块写了语言名
- 该语言是否已加载（见 [10.5](./10-features.md#105-代码高亮)）
- 只有文章页加载 Prism；在列表页或 About 里写代码块不会高亮

### Q：手机上页面能左右滑动？

通常是某个元素太宽：长公式、长链接、宽表格、固定宽度的图片。表格和公式已经做了横向滚动；其他情况可在浏览器开发者工具里找出超宽元素，给它加 `max-width: 100%` 或 `overflow-x: auto`。

### Q：改了 CSS，浏览器里没变化？

本地预览时刷新一次即可（`site.css` 带版本参数）。如果还不行，确认 `jekyll serve` 已经重新构建（终端会打印 `Regenerating`）。

### Q：想调整动画或关掉某个动画？

动画相关的规则在 `site.css` 的 “Motion” 和 “Illustrations” 区块。删掉元素上的 `data-reveal` 就不会渐入；插画元素去掉 `draw`、`pop` 等类就会静止显示。

---

## 图片

### Q：图片不显示？

- 路径以 `/assets/` 开头
- 文件名大小写与引用一致（线上区分大小写）
- 扩展名一致（`.png` / `.jpg`）

### Q：封面图被裁掉了一部分？

卡片按 16:9 裁切。请把封面做成 16:9，主体放在中间。

---

## 语言

### Q：怎么写中文版？

站点目前只有英文版，不再维护中文镜像，见 [05-站点语言与界面文字](./05-bilingual.md)。

---

## 部署

### Q：推送后网站没有更新？

到仓库的 Actions 页面看构建是否完成；完成后强制刷新浏览器。详见 [11.4](./11-deployment.md#114-部署验证)。

### Q：本地 `bundle install` 失败？

如果是 Ruby 4，`github-pages` 依赖装不上，参考 [2.1 如果本机是 Ruby 4](./02-quickstart.md#如果本机是-ruby-4) 的替代方法。

### Q：如何回滚？

用 `git revert <commit>` 生成反向提交再推送，见 [11.6](./11-deployment.md#116-回滚)。

---

## 配置

### Q：改了 `_config.yml` 没有生效？

`jekyll serve` 不会重新读取配置文件，停掉服务后重新启动。

### Q：怎么改导航栏文字？

改 `_data/ui_text.yml`。要增减导航项，修改 `_includes/site-header.html`。

---

## 搜索

### Q：搜不到某篇学习笔记？

搜索只收录博客文章和论文阅读笔记，见 [10.1](./10-features.md#101-搜索)。

### Q：怎么让某页不被搜到？

博客文章和论文笔记都会被收录。要排除某一篇，可以在它的 Front Matter 里加 `search: false`，并在 `search.html` 的筛选中加上 `where_exp: "doc", "doc.search != false"`。

---

## 其他

### Q：怎么查看构建后的静态文件？

```bash
bundle exec jekyll build
open _site/index.html
```

### Q：维护文档页（`/maintenance.html`）显示的还是旧内容？

这个页面从 GitHub 的 `main` 分支读取 `docs/`，文档要推送到 `main` 后才会更新。

---

<div class="doc-nav">
  <a href="./13-checklist.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 维护检查清单</span>
  </a>
  <a href="./appendix/templates.md" class="doc-nav-card next">
    <span class="doc-nav-label">附录</span>
    <span class="doc-nav-title">模板文件 →</span>
  </a>
</div>
