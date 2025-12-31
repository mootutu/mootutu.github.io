# 部署与发布

## 11.1 GitHub Pages 部署原理

### 工作流程

```
本地修改 → git push → GitHub Pages 自动构建 → 网站更新
```

### 自动构建

GitHub Pages 会在以下情况自动触发构建：
- 推送到 `main` 分支
- 推送到配置的部署分支

### 构建过程

1. GitHub 检测到推送
2. 运行 Jekyll 构建
3. 生成静态文件到 `_site/` 目录
4. 部署到 GitHub Pages 服务器
5. 网站更新完成

### 部署地址

```
https://mootutu.github.io
```

---

## 11.2 发布流程

### 标准发布流程

```bash
# 1. 确认更改
git status
git diff

# 2. 添加文件
git add .

# 3. 提交
git commit -m "feat: 添加新功能描述"

# 4. 推送
git push origin main
```

### 提交信息规范

使用语义化提交信息：

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: Add new blog post about Git` |
| `fix` | 修复 | `fix: Correct typo in about page` |
| `docs` | 文档 | `docs: Update README` |
| `style` | 样式 | `style: Adjust blog card spacing` |
| `refactor` | 重构 | `refactor: Simplify navigation logic` |
| `chore` | 杂项 | `chore: Update dependencies` |

### 发布检查清单

在推送之前：

- [ ] 本地预览无误
- [ ] 所有链接可正常访问
- [ ] 图片正常显示
- [ ] 英文和中文版本同步更新
- [ ] 索引页已更新（如新增内容）
- [ ] 代码无语法错误

---

## 11.3 本地预览

### 启动服务器

```bash
# 标准启动
jekyll serve

# 或使用 bundle
bundle exec jekyll serve
```

### 访问地址

```
http://localhost:4000
```

### 常用选项

```bash
# 显示草稿
jekyll serve --drafts

# 实时重载
jekyll serve --livereload

# 指定端口
jekyll serve --port 3000

# 局域网访问
jekyll serve --host 0.0.0.0
```

### 停止服务器

按 `Ctrl + C`

---

## 11.4 部署验证

### 检查部署状态

1. 访问 GitHub 仓库页面
2. 点击 "Actions" 标签
3. 查看最新的构建状态

### 验证网站更新

1. 访问线上地址：`https://mootutu.github.io`
2. 强制刷新页面（Ctrl+F5 或 Cmd+Shift+R）
3. 检查更新内容是否生效

### 常见部署延迟

- 通常 1-3 分钟内完成
- 首次部署可能需要更长时间
- 缓存可能导致看到旧版本

---

## 11.5 部署问题排查

### 问题：推送后网站未更新

**可能原因**：
1. 构建失败
2. 浏览器缓存
3. CDN 缓存

**解决方法**：
1. 检查 GitHub Actions 构建日志
2. 强制刷新浏览器
3. 等待几分钟后重试

### 问题：构建失败

**查看错误**：
1. 进入 GitHub 仓库 → Actions
2. 点击失败的构建
3. 查看错误日志

**常见错误**：
- Front Matter 语法错误
- Liquid 模板语法错误
- 文件编码问题

### 问题：页面 404

**可能原因**：
1. 文件路径错误
2. 文件名大小写问题
3. 文件未提交

**解决方法**：
1. 检查文件是否存在于仓库
2. 确认 URL 路径正确
3. 确认文件已提交并推送

---

## 11.6 回滚更改

### 回滚到上一个提交

```bash
# 查看提交历史
git log --oneline

# 回滚到指定提交（保留更改）
git reset --soft HEAD~1

# 回滚到指定提交（丢弃更改）
git reset --hard HEAD~1

# 强制推送（谨慎使用）
git push --force origin main
```

### 撤销单个文件的更改

```bash
# 撤销工作区更改
git checkout -- path/to/file

# 撤销已暂存的更改
git reset HEAD path/to/file
```

---

## 11.7 分支管理

### 当前分支策略

- `main`：主分支，部署到线上

### 使用功能分支（可选）

```bash
# 创建新分支
git checkout -b feature/new-feature

# 开发完成后合并
git checkout main
git merge feature/new-feature

# 删除功能分支
git branch -d feature/new-feature
```

---

## 11.8 持续集成（可选）

### GitHub Actions

如需自定义构建流程，可创建 `.github/workflows/jekyll.yml`：

```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Jekyll
        uses: actions/jekyll-build-pages@v1
      - name: Deploy
        uses: actions/deploy-pages@v1
```

> **注意**：当前项目使用 GitHub Pages 默认构建，无需自定义 Actions。

---

## 11.9 域名配置（可选）

### 自定义域名

如需使用自定义域名：

1. 在仓库根目录创建 `CNAME` 文件
2. 写入域名：
   ```
   www.yourdomain.com
   ```
3. 在域名 DNS 中添加 CNAME 记录指向 `mootutu.github.io`

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
