# 站点语言与界面文字

> 本篇原来是“双语内容维护”。2026-07-16 起站点改为只有英文，`/cn/` 中文镜像、`translate_url` 字段和语言切换按钮都已删除。文件名保留为 `05-bilingual.md`，是为了不让维护文档页和旧链接失效。

## 5.1 语言约定

- 所有页面内容一律用英文写，Front Matter 中写 `lang: en`
- 不要再新建 `cn/` 目录或填写 `translate_url`
- 例外：首页中文名（王伟钦）、“中文简历”按钮、`cv_zh/` 中文简历
- 维护文档 `docs/` 用中文写，它不参与站点构建

`lang` 字段仍然有用：

- `<html lang>` 属性取自它
- 学习主题列表和主题页只显示 `lang` 与当前页面相同的内容，漏写会导致新笔记不出现

---

## 5.2 界面文字：`_data/ui_text.yml`

导航栏上的文字集中放在这个文件里：

```yaml
en:
  home: Home
  publications: Publications
  experience: Experience
  awards: Awards
  service: Service
```

在模板里这样读取（`_includes/site-header.html`）：

```liquid
{% assign ui = site.data.ui_text.en %}
<a class="site-nav__link" href="/#publications">{{ ui.publications }}</a>
```

改导航上的文字只需要改这个文件。其他页面文字（按钮、小标题、提示语）直接写在对应的模板或页面里，例如：

| 文字 | 位置 |
|------|------|
| 页头小字（Writing、Learning…） | 各包装布局的 `eyebrow`，或页面 Front Matter 的 `eyebrow` |
| 返回链接（All posts、All topics…） | 包装布局的 `back_label` |
| 文章页 “On this page” | `_layouts/post.html` |
| 搜索框提示 | `_includes/site-header.html` |
| 页脚 | `_includes/site-footer.html` |

---

## 5.3 如果以后要恢复中文版

当前的布局没有为多语言做准备。需要时建议：

1. 在 `ui_text.yml` 增加 `zh` 一组文字，并让 `site-header.html` 按 `page.lang` 选择
2. 中文页面放在 `/zh/` 下，`lang: zh-CN`
3. 中文字体已在 `--font-serif` / `--font-sans` 的后备字体中包含（Songti SC、PingFang SC），无需额外引入

---

<div class="doc-nav">
  <a href="./04-learning-maintenance.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 学习内容维护</span>
  </a>
  <a href="./06-assets.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">静态资源管理 →</span>
  </a>
</div>
