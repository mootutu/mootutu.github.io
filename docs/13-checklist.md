# 维护检查清单

## 13.1 新增博客文章检查清单

### 创建阶段

- [ ] 复制模板文件到正确位置
  - 英文：`blog/posts/article-name.html`
  - 中文：`cn/blog/posts/article-name.html`

- [ ] Front Matter 填写完整
  - [ ] `layout: blog-post-layout`
  - [ ] `title` 已填写
  - [ ] `date` 格式正确（YYYY-MM-DD）
  - [ ] `author` 已填写
  - [ ] `category` 已填写
  - [ ] `lang` 正确（en 或 zh-CN）
  - [ ] `translate_url` 指向正确

### 内容阶段

- [ ] 文章内容编写完成
- [ ] 标题有正确的 `id` 属性（用于 TOC）
- [ ] 代码块指定了语言（如 `language-python`）
- [ ] 图片路径正确

### 资源阶段

- [ ] 封面图已添加
  - 位置：`assets/images/covers/article-name.png`
  - 命名与文章文件名一致

- [ ] 文章内图片已添加（如需要）
  - 位置：`assets/images/posts/article-name/`

### 索引阶段

- [ ] 英文博客列表已更新（`blog.html`）
- [ ] 中文博客列表已更新（`cn/blog.html`）
- [ ] 新文章卡片添加在列表最前面

### 验证阶段

- [ ] 本地预览正常
- [ ] 链接可点击
- [ ] 图片正常显示
- [ ] TOC 正常生成
- [ ] 暗黑模式显示正常
- [ ] 语言切换正常

---

## 13.2 新增学习文章检查清单

### 创建阶段

- [ ] 创建文章文件
  - 英文：`learning/topic/article.md`
  - 中文：`cn/learning/topic/article.md`

- [ ] Front Matter 填写完整
  - [ ] `layout: learning-post-layout`
  - [ ] `title` 已填写
  - [ ] `date` 格式正确
  - [ ] `lang` 正确
  - [ ] `topic_url` 指向主题页面
  - [ ] `translate_url` 指向正确
  - [ ] `mathjax` 设置正确（如需公式）

### 内容阶段

- [ ] 文章内容编写完成
- [ ] Markdown 语法正确
- [ ] 代码块指定了语言

### 索引阶段

- [ ] 主题索引页已更新
  - 英文：`learning/topic.html`
  - 中文：`cn/learning/topic.html`

- [ ] 链接使用 `.html` 扩展名（即使源文件是 `.md`）

### 验证阶段

- [ ] 本地预览正常
- [ ] 返回主题按钮工作正常
- [ ] 代码高亮正常
- [ ] MathJax 公式正常（如启用）

---

## 13.3 新增学习主题检查清单

### 创建阶段

- [ ] 创建主题索引页
  - 英文：`learning/topic.html`
  - 中文：`cn/learning/topic.html`

- [ ] 创建文章目录
  - 英文：`learning/topic/`
  - 中文：`cn/learning/topic/`

- [ ] Front Matter 填写完整
  - [ ] `layout: learning-topic-layout`
  - [ ] `lang` 正确
  - [ ] `translate_url` 指向正确

### 索引阶段

- [ ] 学习主页已更新
  - 英文：`learning.html`
  - 中文：`cn/learning.html`

- [ ] 主题卡片信息正确
  - [ ] 标题
  - [ ] 描述
  - [ ] 链接

---

## 13.4 更新个人信息检查清单

### 更新个人简介

- [ ] 英文简介已更新（`_includes/bio.html`）
- [ ] 中文简介已更新（`_includes/bio-cn.html`）
- [ ] 两个版本内容一致

### 更新头像

- [ ] 新头像已替换
- [ ] 文件名保持不变（或已更新引用）
- [ ] 图片尺寸合适

### 更新出版物

- [ ] BibTeX 文件已添加（`assets/bibtex/`）
- [ ] 英文首页已更新（`index.html`）
- [ ] 中文首页已更新（`cn/index.html`）
- [ ] BibTeX 模态框内容正确
- [ ] PDF 链接有效

---

## 13.5 提交前检查清单

### 内容检查

- [ ] 所有文本无拼写错误
- [ ] 所有链接可正常访问
- [ ] 所有图片正常显示
- [ ] Front Matter 格式正确

### 双语检查

- [ ] 英文版本完整
- [ ] 中文版本完整
- [ ] translate_url 互相指向正确

### 样式检查

- [ ] 亮色模式显示正常
- [ ] 暗黑模式显示正常
- [ ] 移动端显示正常

### 功能检查

- [ ] 导航链接正常
- [ ] 搜索功能正常
- [ ] TOC 生成正常
- [ ] 返回按钮正常

---

## 13.6 发布后检查清单

### 部署验证

- [ ] GitHub Actions 构建成功
- [ ] 网站可正常访问

### 内容验证

- [ ] 新内容已上线
- [ ] 链接可正常访问
- [ ] 图片正常显示

### 功能验证

- [ ] 搜索可找到新内容
- [ ] 语言切换正常
- [ ] 所有交互功能正常

---

## 13.7 定期维护检查清单

### 月度检查

- [ ] 检查所有外部链接是否有效
- [ ] 检查 CDN 资源是否可访问
- [ ] 清理未使用的图片资源

### 季度检查

- [ ] 更新依赖版本（如有）
- [ ] 检查 GitHub Pages 构建日志
- [ ] 备份重要内容

---

## 13.8 问题排查检查清单

### 页面不显示

- [ ] 文件是否存在于正确位置
- [ ] 文件名大小写是否正确
- [ ] Front Matter 是否有语法错误
- [ ] 是否已提交并推送

### 样式异常

- [ ] CSS 类名是否正确
- [ ] 是否遗漏了暗黑模式样式
- [ ] 是否有 CSS 语法错误

### 链接失效

- [ ] URL 路径是否正确
- [ ] 是否使用了绝对路径
- [ ] 目标文件是否存在

### 图片不显示

- [ ] 图片文件是否存在
- [ ] 路径是否正确（绝对路径）
- [ ] 文件名大小写是否匹配

---

[← 上一篇：命名规范汇总](./12-naming.md) | [返回目录](./README.md) | [下一篇：常见问题FAQ →](./14-faq.md)
