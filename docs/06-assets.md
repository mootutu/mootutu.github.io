# 静态资源管理

## 6.1 资源目录结构

```
assets/
├── css/                    # 样式表
│   ├── pygment_trac.css    # 代码高亮样式
│   └── styles.css          # 通用样式
│
├── javascripts/            # JavaScript 文件
│   └── scale.fix.js        # 响应式修复脚本
│
├── images/                 # 图片资源
│   ├── ggbond.ico          # 网站图标
│   ├── mootutu.png         # 头像 1
│   ├── weiqin.png          # 头像 2
│   ├── bocchi.gif          # 导航栏动画
│   ├── covers/             # 博客封面图
│   └── posts/              # 文章内图片
│
├── fontawesome/            # FontAwesome 图标库
│   ├── css/
│   ├── js/
│   └── webfonts/
│
└── bibtex/                 # 论文引用文件
    ├── wang2025beyond.bib
    └── wang2025ranked.bib
```

---

## 6.2 图片资源管理

### 6.2.1 图片分类

| 类型 | 存放位置 | 用途 |
|------|----------|------|
| 网站图标 | `assets/images/` | favicon、头像 |
| 博客封面图 | `assets/images/covers/` | 博客列表展示 |
| 文章内图片 | `assets/images/posts/{article}/` | 文章正文配图 |

### 6.2.2 封面图规范

**存放位置：**
```
assets/images/covers/
├── modernizing-python.png
├── git-basics.png
├── automating-tasks-python.png
└── markdown-guide.png
```

**命名规则：**
- 与文章文件名一致（不含扩展名）
- 使用小写字母和连字符

**尺寸建议：**
- 比例：1:1（正方形）
- 尺寸：400x400 ~ 800x800 像素
- 格式：PNG 或 JPG

**示例对应关系：**

| 文章文件 | 封面图文件 |
|----------|-----------|
| `blog/posts/git-basics.html` | `assets/images/covers/git-basics.png` |
| `blog/posts/my-new-post.html` | `assets/images/covers/my-new-post.png` |

### 6.2.3 文章内图片规范

**存放位置：**
```
assets/images/posts/
└── article-name/           # 以文章名命名的目录
    ├── image.png           # 第一张图
    ├── image-1.png         # 第二张图
    ├── image-2.png         # 第三张图
    └── screenshot.png      # 也可用描述性名称
```

**命名规则：**
- 目录名与文章文件名一致
- 图片可用序号命名：`image.png`、`image-1.png`、`image-2.png`
- 也可用描述性名称：`vscode-config.png`、`terminal-output.png`

**引用方式（HTML）：**
```html
<figure class="image">
    <img src="/assets/images/posts/article-name/image.png" alt="描述文字">
    <figcaption>图片说明</figcaption>
</figure>
```

**引用方式（Markdown）：**
```markdown
![描述文字](/assets/images/posts/article-name/image.png)
```

### 6.2.4 添加新图片流程

1. **确定图片类型**
   - 封面图 → `assets/images/covers/`
   - 文章配图 → `assets/images/posts/{article}/`

2. **创建目录（如需要）**
   ```bash
   mkdir -p assets/images/posts/new-article
   ```

3. **复制图片并命名**
   ```bash
   cp ~/Downloads/screenshot.png assets/images/posts/new-article/image.png
   ```

4. **在文章中引用**

5. **本地预览确认**

---

## 6.3 CSS 管理

### 现有 CSS 文件

| 文件 | 用途 |
|------|------|
| `assets/css/pygment_trac.css` | 代码高亮基础样式 |
| `assets/css/styles.css` | 通用样式（较少使用） |

### 样式位置说明

本项目大部分样式直接写在布局文件和页面中（内联 `<style>` 标签），而非独立 CSS 文件：

- 博客列表样式 → `_includes/blog_page_styles.html`
- 博客文章样式 → `_layouts/blog-post-layout.html`
- 学习页面样式 → `_layouts/learning-*.html`
- 导航栏样式 → `_includes/navigation.html`

### 修改样式的注意事项

1. 确定要修改的样式所在文件
2. 注意暗黑模式兼容（检查 `.dark-mode` 相关样式）
3. 测试响应式效果（不同屏幕宽度）

---

## 6.4 JavaScript 管理

### 内置脚本

| 文件 | 用途 |
|------|------|
| `assets/javascripts/scale.fix.js` | 响应式视口修复 |

### 外部库（CDN）

| 库 | 版本 | 用途 | 引用位置 |
|----|------|------|----------|
| Prism.js | v1.29.0 | 代码语法高亮 | 博客/学习文章布局 |
| MathJax | v3 | 数学公式渲染 | 学习文章布局 |

### 内联脚本位置

主要功能脚本直接写在布局文件中：

- 暗黑模式切换 → `_layouts/default.html`
- TOC 生成 → `_layouts/blog-post-layout.html`
- 搜索功能 → `_includes/navigation.html`
- 返回顶部 → 各布局文件

---

## 6.5 字体资源

### 使用的字体

| 字体 | 用途 | 来源 |
|------|------|------|
| Fira Code | 代码显示 | Google Fonts CDN |
| System Fonts | 正文 | 系统默认 |

### 字体引用

```html
<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
```

---

## 6.6 图标资源

### FontAwesome

完整的 FontAwesome 库位于 `assets/fontawesome/`：

```
assets/fontawesome/
├── css/
│   ├── all.min.css
│   └── ...
├── js/
└── webfonts/
```

### 使用方式

```html
<i class="fas fa-home"></i>
<i class="fab fa-github"></i>
```

---

## 6.7 BibTeX 文件

### 存放位置

```
assets/bibtex/
├── wang2025beyond.bib
└── wang2025ranked.bib
```

### 命名规则

```
{作者姓}{年份}{关键词}.bib
```

示例：`wang2025beyond.bib`

### 添加新 BibTeX

1. 创建 `.bib` 文件
2. 放入 `assets/bibtex/` 目录
3. 在 `_data/publications.yml` 中添加引用

---

## 6.8 资源路径规范

### 绝对路径 vs 相对路径

**推荐使用绝对路径**（以 `/` 开头）：

```html
<!-- 推荐 -->
<img src="/assets/images/covers/article.png">

<!-- 不推荐 -->
<img src="../assets/images/covers/article.png">
```

### 路径示例

| 资源类型 | 路径格式 |
|----------|----------|
| 封面图 | `/assets/images/covers/{filename}.png` |
| 文章图片 | `/assets/images/posts/{article}/{image}.png` |
| CSS | `/assets/css/{filename}.css` |
| JS | `/assets/javascripts/{filename}.js` |
| BibTeX | `/assets/bibtex/{filename}.bib` |

---

<div class="doc-nav">
  <a href="./05-bilingual.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 双语内容维护</span>
  </a>
  <a href="./07-personal-info.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">个人信息维护 →</span>
  </a>
</div>
