# 维护检查清单

## 13.1 新增博客文章

- [ ] 从 `_templates/blog_post_template.html` 复制，文件名小写 + 连字符
- [ ] Front Matter：`title`、`date`、`category`、`excerpt`、`tags`、`reading_time`、`cover_image`、`lang: en` 都已填写
- [ ] 正文用 `h2` / `h3` 分节，没有自定义 `<style>`
- [ ] 代码块写了 `language-xxx`
- [ ] 封面图放在 `assets/images/covers/同名.png`，16:9，主体居中
- [ ] 文章配图放在 `assets/images/posts/同名/`，路径以 `/` 开头
- [ ] `/blog.html` 中出现新卡片，封面、日期、摘要正确
- [ ] 文章页的目录、代码高亮正常

## 13.2 新增学习笔记

- [ ] 从 `_templates/learning_post_template.md` 复制，放在对应主题目录
- [ ] `topic` 与主题页一致，`order` 不与已有笔记重复，`topic_url` 正确
- [ ] 用到公式时写了 `mathjax: true`，公式渲染正常、手机上不撑宽页面
- [ ] 主题页中出现新笔记，顺序正确；学习区首页的笔记数量 +1
- [ ] 笔记页的返回链接回到正确的主题页

## 13.3 新增论文阅读笔记

- [ ] 从 `_templates/paper_reading_template.md` 复制，文件名 `arXiv编号-简短标题.md`
- [ ] `status`、`relevance` 使用规定的小写值
- [ ] 填写了 `one_line_summary`、`paper_url`、`tags`
- [ ] 论文阅读库中出现新卡片，搜索和两个筛选都能找到它

## 13.4 新增学习主题

- [ ] 新建 `learning/主题.html`，`layout: learning-topic-layout`，填写 `title`、`description`、`topic`、`order`、`lang`
- [ ] 正文只有 `{% include learning_entries.html %}`
- [ ] 标题里没有 emoji
- [ ] `/learning.html` 出现新卡片

## 13.5 提交前检查清单

### 内容

- [ ] 拼写、日期正确
- [ ] 所有链接可以打开
- [ ] 图片都能显示

### 外观

- [ ] 浅色、深色模式各检查一次
- [ ] 375px 宽（手机）下没有横向滚动条
- [ ] 改了样式时，至少检查：首页、博客列表、一篇博客、学习区首页、一个主题页、一篇带公式的笔记、论文阅读库、About

### 构建

- [ ] `jekyll serve` 终端没有 `Liquid Exception` / `Error`
- [ ] 只 `git add` 了相关文件；没有提交 `_site/`、`.sass-cache/`、LaTeX 中间文件

## 13.6 发布后检查清单

- [ ] GitHub Actions 中 “pages build and deployment” 成功
- [ ] 线上页面强制刷新后显示新内容
- [ ] 手机上打开一次
- [ ] 改了文档时，`/maintenance.html` 显示的是新内容

## 13.7 个人信息更新

- [ ] 首页对应区块已修改（[7.1](./07-personal-info.md#71-首页文字)）
- [ ] 新论文：`publications.yml` 条目 + BibTeX 文件，会议标签、年份、加粗正确
- [ ] 简历 `.tex` 已修改并重新编译，PDF 已提交
- [ ] `_includes/contact-buttons.html` 中的链接都能打开

## 13.8 定期维护

### 每月

- [ ] 首页信息是否过时
- [ ] 外部链接（论文、代码仓库）是否还能打开

### 每学期

- [ ] 简历和首页经历同步
- [ ] 清理不再使用的图片（见 [6.1](./06-assets.md#61-目录结构)）
- [ ] 检查 Prism、MathJax 的 CDN 版本是否需要更新

## 13.9 问题排查

| 现象 | 先检查 |
|------|-------|
| 页面空白或报错 | 终端里的 Liquid 报错；Front Matter 缩进和引号 |
| 新内容不在列表里 | `layout`、`topic`、`lang` 是否写对 |
| 样式错乱 | 是否在页面里写了 `<style>`；浏览器是否缓存了旧 CSS |
| 链接失效 | 路径是否以 `/` 开头；`.md` 文件链接要写 `.html` |
| 图片不显示 | 文件名大小写、扩展名、路径 |

---

<div class="doc-nav">
  <a href="./12-naming.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 命名规范汇总</span>
  </a>
  <a href="./14-faq.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">常见问题 FAQ →</span>
  </a>
</div>
