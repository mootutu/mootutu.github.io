# 部署与发布

## 11.1 部署方式

仓库名是 `mootutu.github.io`，GitHub Pages 会自动从 `main` 分支构建并发布到 `https://mootutu.github.io`。

```
git push origin main  →  GitHub Pages 用 Jekyll 3.10 构建  →  1–3 分钟后网站更新
```

- 仓库里没有自定义 GitHub Actions 工作流，用的是 GitHub Pages 内置构建
- 线上构建使用 `github-pages` gem 锁定的版本，只能使用它白名单里的插件（本站只用了 `jekyll-feed`）
- 构建产物 `_site/` 不提交

---

## 11.2 提交信息规范

使用语义化提交信息，格式为 `类型: 简短说明`（英文、小写开头）：

| 类型 | 用于 | 示例 |
|------|------|------|
| `feat` | 新内容、新功能 | `feat: add blog post about git worktrees` |
| `fix` | 修正错误 | `fix: correct typo in about page` |
| `docs` | 维护文档、README | `docs: update maintenance guide` |
| `style` | 只改样式 | `style: tighten blog card spacing` |
| `refactor` | 重构，不改变外观 | `refactor: merge post layouts` |
| `chore` | 杂项、删除文件、依赖 | `chore: remove unused images` |

一次提交只做一件事；内容更新和样式调整尽量分开提交。

---

## 11.3 发布流程

```bash
# 1. 本地预览，确认无误
bundle exec jekyll serve

# 2. 查看改动
git status
git diff

# 3. 只添加相关文件
git add blog/posts/new-post.html assets/images/covers/new-post.png

# 4. 提交并推送
git commit -m "feat: add blog post about X"
git push origin main
```

推送前请过一遍 [13.5 提交前检查清单](./13-checklist.md#135-提交前检查清单)。

---

## 11.4 部署验证

1. 打开仓库的 **Actions** 页面，确认 “pages build and deployment” 成功（绿色对勾）
2. 打开线上页面，强制刷新（⌘⇧R / Ctrl+F5）
3. 检查改动的页面，并在手机上打开一次

`site.css`、`site.js` 带有构建时间作为版本参数，一般不会遇到样式缓存问题；如果仍是旧样式，等几分钟让 CDN 缓存过期。

---

## 11.5 问题排查

| 现象 | 排查方法 |
|------|---------|
| 推送后网站没变化 | 看 Actions 是否还在运行或失败；确认推送的是 `main` |
| 构建失败 | 在 Actions 日志中找 `Liquid Exception`、`YAML Exception`，通常是 Front Matter 缩进、引号或 include 文件名写错 |
| 页面 404 | 检查文件路径和链接大小写是否一致；Markdown 页面的地址后缀是 `.html` |
| 本地正常线上异常 | 本地 Jekyll 版本可能与线上不同，尽量用 `github-pages` gem 预览；检查是否用了线上不支持的插件 |

---

## 11.6 回滚

```bash
# 查看历史
git log --oneline

# 撤销某次提交（生成一个新的反向提交，推荐）
git revert <commit>
git push origin main
```

不要对 `main` 使用 `git push --force`，除非确定没有其他人或设备基于旧历史工作。

撤销本地未提交的修改：

```bash
git restore <file>            # 撤销工作区修改
git restore --staged <file>   # 取消暂存
```

---

## 11.7 分支

日常小改动可以直接提交到 `main`。较大的改版（例如重做设计）建议：

```bash
git switch -c redesign
# …修改、提交…
git push -u origin redesign
# 在 GitHub 上开 Pull Request，确认无误后合并到 main
```

---

## 11.8 自定义域名（可选）

1. 在仓库根目录新建 `CNAME` 文件，内容是域名（如 `www.example.com`）
2. 在域名服务商处添加指向 `mootutu.github.io` 的 CNAME 记录
3. 在仓库 Settings → Pages 中确认域名并开启 HTTPS

---

<div class="doc-nav">
  <a href="./10-features.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 特殊功能</span>
  </a>
  <a href="./12-naming.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">命名规范汇总 →</span>
  </a>
</div>
