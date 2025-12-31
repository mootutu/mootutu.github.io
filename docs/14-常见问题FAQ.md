# 常见问题 FAQ

## 内容相关问题

### Q: 为什么我新增的文章不显示在列表中？

**原因**：博客列表和学习列表不会自动更新，需要手动维护。

**解决方法**：
1. 博客文章：更新 `blog.html`（和 `cn/blog.html`）
2. 学习文章：更新对应的主题索引页

---

### Q: 为什么 Markdown 文件的链接要写成 .html？

**原因**：Jekyll 会将 `.md` 文件转换为 `.html` 文件。

**示例**：
```html
<!-- 源文件是 article.md，但链接写成 -->
<a href="/learning/topic/article.html">文章标题</a>
```

---

### Q: 为什么文章的目录（TOC）没有生成？

**可能原因**：
1. 屏幕宽度小于 1280px（TOC 只在大屏幕显示）
2. 标题没有 `id` 属性
3. 没有使用 H2/H3 标题

**解决方法**：
```html
<!-- 确保标题有 id -->
<h2 id="section-1">标题</h2>
```

---

### Q: 如何在学习文章中使用数学公式？

**方法**：
1. 在 Front Matter 中设置 `mathjax: true`
2. 使用 LaTeX 语法编写公式

```yaml
---
mathjax: true
---

行内公式：$E = mc^2$

独立公式：
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$
```

---

## 样式相关问题

### Q: 暗黑模式下某个元素显示异常怎么办？

**解决方法**：为该元素添加 `.dark-mode` 样式：

```css
/* 亮色模式 */
.my-element {
    background: #ffffff;
    color: #333333;
}

/* 暗黑模式 */
.dark-mode .my-element {
    background: #1a1a2e;
    color: #e2e8f0;
}
```

---

### Q: 代码高亮不工作怎么办？

**检查项**：
1. 是否使用了正确的语言标识符
2. HTML 格式是否正确

**正确格式**：
```html
<pre><code class="language-python">
def hello():
    print("Hello")
</code></pre>
```

**支持的语言**：`python`、`bash`、`javascript`、`json`、`yaml`、`html`、`css`

---

### Q: 移动端显示有问题怎么办？

**排查步骤**：
1. 使用浏览器开发者工具（F12）
2. 切换到移动设备模式
3. 检查是否有响应式样式

**添加响应式样式**：
```css
@media (max-width: 768px) {
    .my-element {
        /* 移动端样式 */
    }
}
```

---

## 图片相关问题

### Q: 图片不显示怎么办？

**检查项**：
1. 图片文件是否存在
2. 路径是否正确（使用绝对路径）
3. 文件名大小写是否匹配

**正确的图片引用**：
```html
<img src="/assets/images/covers/article.png" alt="描述">
```

---

### Q: 封面图尺寸应该是多少？

**推荐**：
- 比例：1:1（正方形）
- 尺寸：400x400 ~ 800x800 像素
- 格式：PNG 或 JPG

---

## 双语相关问题

### Q: 语言切换按钮不工作怎么办？

**检查项**：
1. `translate_url` 是否正确设置
2. 目标页面是否存在

**Front Matter 示例**：
```yaml
# 英文页面
translate_url: /cn/blog/posts/article.html

# 中文页面
translate_url: /blog/posts/article.html
```

---

### Q: 如何只创建单语言版本？

**方法**：将 `translate_url` 设置为默认首页：

```yaml
# 只有英文版本
translate_url: /cn/index.html
```

后续创建中文版本时再更新。

---

## 部署相关问题

### Q: 推送后网站没有更新怎么办？

**解决步骤**：
1. 检查 GitHub Actions 是否构建成功
2. 强制刷新浏览器（Ctrl+F5）
3. 等待 1-3 分钟后重试
4. 清除浏览器缓存

---

### Q: GitHub Actions 构建失败怎么办？

**排查步骤**：
1. 进入 GitHub 仓库 → Actions
2. 点击失败的构建
3. 查看错误日志

**常见错误**：
- Front Matter 语法错误（如缺少 `---`）
- Liquid 模板语法错误
- 文件编码问题（使用 UTF-8）

---

### Q: 如何回滚到之前的版本？

**方法**：
```bash
# 查看提交历史
git log --oneline

# 回滚到指定提交
git reset --hard <commit-hash>

# 强制推送
git push --force origin main
```

---

## 配置相关问题

### Q: 修改了 _config.yml 没有生效？

**原因**：修改配置文件后需要重启 Jekyll 服务器。

**解决方法**：
```bash
# 停止服务器 (Ctrl+C)
# 重新启动
jekyll serve
```

---

### Q: 如何添加新的 UI 文本？

**步骤**：
1. 编辑 `_data/ui_text.yml`
2. 同时添加 `en` 和 `zh-CN` 版本

```yaml
en:
  new_text: "New Text"

zh-CN:
  new_text: "新文本"
```

3. 在模板中使用：`{{ ui.new_text }}`

---

## 搜索相关问题

### Q: 搜索结果不正确或找不到内容？

**可能原因**：
1. 内容太新，索引未更新
2. 搜索关键词少于 4 个字符

**解决方法**：
1. 确保页面有正确的 Front Matter
2. 重新构建站点
3. 使用更长的搜索关键词

---

### Q: 如何让某个页面不被搜索到？

**目前不支持**：当前实现会索引所有带 Front Matter 的页面。

如需排除，需要修改 `search.html` 中的索引逻辑。

---

## 其他问题

### Q: 如何备份网站内容？

**方法**：
```bash
# 克隆完整仓库
git clone https://github.com/mootutu/mootutu.github.io.git backup

# 或下载 ZIP
# GitHub 仓库页面 → Code → Download ZIP
```

---

### Q: 如何在本地查看构建的静态文件？

**方法**：
```bash
# 构建站点
jekyll build

# 静态文件在 _site/ 目录
ls _site/
```

---

### Q: 遇到文档中没有的问题怎么办？

**建议**：
1. 查看 Jekyll 官方文档：https://jekyllrb.com/docs/
2. 搜索 GitHub Issues
3. 查看浏览器控制台错误信息
4. 检查 Jekyll 服务器输出日志

---

[← 上一篇：维护检查清单](./13-维护检查清单.md) | [返回目录](./README.md) | [附录 →](./appendix/模板文件.md)
