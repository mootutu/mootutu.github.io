# 特殊功能

## 10.1 搜索功能

### 工作原理

1. **索引生成**：`search.html` 生成 `search.json`，包含所有可搜索内容
2. **前端搜索**：导航栏中的搜索框通过 JavaScript 实时搜索
3. **结果展示**：匹配结果以下拉列表形式显示

### 搜索索引（search.json）

**生成位置**：`/search.html` → `/search.json`

**索引内容**：
- 所有博客文章
- 所有学习文章
- 主题页面

**索引字段**：
```json
{
  "title": "文章标题",
  "category": "分类",
  "tags": "标签1, 标签2",
  "url": "/blog/posts/article.html",
  "date": "December 30, 2025",
  "lang": "en",
  "content": "文章内容前50词..."
}
```

### 搜索实现位置

搜索功能在 `_includes/navigation.html` 中实现：

```html
<input type="text" id="search-input" placeholder="{{ ui.search_placeholder }}">
<div id="search-results"></div>

<script>
// 搜索逻辑
</script>
```

### 搜索特性

| 特性 | 说明 |
|------|------|
| 实时搜索 | 输入即搜索，无需按回车 |
| 语言过滤 | 自动根据当前页面语言过滤结果 |
| 最大结果数 | 最多显示 5 条结果 |
| 最小搜索长度 | 4 个字符以上才触发搜索 |
| ESC 关闭 | 按 ESC 键关闭搜索结果 |

### 内容如何被索引

新增的文章会自动被索引，条件是：
- 文件有正确的 Front Matter
- 文件位于 `blog/posts/` 或 `learning/` 目录

---

## 10.2 暗黑模式

### 实现原理

1. **状态存储**：使用 `localStorage` 保存用户偏好
2. **即时应用**：页面加载时立即应用，避免闪烁
3. **切换按钮**：固定在右下角

### 触发方式

- 点击右下角的暗黑模式切换按钮
- 状态会持久化保存

### CSS 类

```css
/* 暗黑模式应用于 html 和 body */
.dark-mode {
    /* 暗黑模式样式 */
}
```

### 颜色方案

| 元素 | 亮色模式 | 暗黑模式 |
|------|----------|----------|
| 背景 | `#ffffff` | `#1a1a2e` → `#16213e` |
| 文字 | `#333333` | `#e2e8f0` |
| 次要文字 | `#666666` | `#cbd5e1` |
| 链接 | `#0066cc` | `#60a5fa` |
| 边框 | `#e9ecef` | `#334155` |
| 代码背景 | `#f8f9fa` | `#1e293b` |

### 添加暗黑模式支持

为新元素添加暗黑模式样式：

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

### 实现代码位置

暗黑模式逻辑在 `_layouts/default.html` 中：

```javascript
// 页面加载时检查偏好
if (localStorage.getItem('darkMode') === 'true') {
    document.documentElement.classList.add('dark-mode');
}

// 切换函数
function toggleDarkMode() {
    document.documentElement.classList.toggle('dark-mode');
    localStorage.setItem('darkMode',
        document.documentElement.classList.contains('dark-mode'));
}
```

---

## 10.3 目录自动生成（TOC）

### 工作原理

1. JavaScript 扫描页面中的 H2 和 H3 标题
2. 自动生成目录结构
3. 在侧边栏显示（大屏幕）

### 支持的布局

- `blog-post-layout.html`
- `learning-post-layout.html`

### 标题要求

```html
<!-- 标题必须有 id 属性 -->
<h2 id="section-1">第一部分</h2>
<h3 id="section-1-1">1.1 子标题</h3>
```

### 显示条件

- 屏幕宽度 ≥ 1280px
- 页面中存在 H2/H3 标题

### TOC 特性

| 特性 | 说明 |
|------|------|
| 粘性定位 | 滚动时固定在侧边 |
| 滚动跟随 | 当前阅读位置高亮 |
| 点击跳转 | 平滑滚动到对应章节 |

---

## 10.4 阅读时间计算

### 计算算法

```javascript
// 英文：200 词/分钟
// 中文：500 字符/分钟
// 最小值：1 分钟

function calculateReadingTime(content, lang) {
    if (lang === 'zh-CN') {
        return Math.max(1, Math.ceil(content.length / 500));
    } else {
        const words = content.split(/\s+/).length;
        return Math.max(1, Math.ceil(words / 200));
    }
}
```

### 显示位置

- 博客文章元信息区
- 学习文章元信息区
- 博客列表卡片

---

## 10.5 代码高亮（Prism.js）

### 使用的库

- **Prism.js** v1.29.0
- **主题**：Tomorrow Night

### 支持的语言

| 语言 | 标识符 |
|------|--------|
| Python | `python` |
| Bash/Shell | `bash` |
| JavaScript | `javascript` |
| JSON | `json` |
| TOML | `toml` |
| YAML | `yaml` |
| HTML | `html` |
| CSS | `css` |

### 使用方式

**HTML 文件中**：
```html
<pre><code class="language-python">def hello():
    print("Hello, World!")
</code></pre>
```

**Markdown 文件中**：
````markdown
```python
def hello():
    print("Hello, World!")
```
````

### 添加新语言支持

1. 找到 Prism.js CDN 链接（在布局文件中）
2. 添加对应语言组件：
   ```html
   <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-新语言.min.js"></script>
   ```

### 代码块样式

代码块使用 Fira Code 字体，具有：
- 圆角边框
- 行号显示
- 复制按钮（部分主题）

---

## 10.6 数学公式（MathJax）

### 启用方式

在 Front Matter 中设置：

```yaml
---
mathjax: true
---
```

### 仅在 Learning Post 中可用

MathJax 只在 `learning-post-layout.html` 中加载。

### 语法

**行内公式**：
```markdown
当 $a \ne 0$ 时，方程 $ax^2 + bx + c = 0$ 有解。
```

**独立公式**：
```markdown
$$
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$
```

### 配置位置

MathJax 配置在 `_layouts/learning-post-layout.html` 中：

```html
{% if page.mathjax %}
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
{% endif %}
```

---

## 10.7 返回顶部按钮

### 位置

固定在页面右下角（第一个按钮）

### 显示条件

滚动超过 300px 时显示

### 实现

```javascript
window.onscroll = function() {
    if (document.documentElement.scrollTop > 300) {
        backToTopBtn.style.display = "block";
    } else {
        backToTopBtn.style.display = "none";
    }
};
```

---

## 10.8 语言切换按钮

### 位置

固定在页面右下角（第二个按钮）

### 工作方式

1. 读取当前页面的 `translate_url` 字段
2. 如果存在，跳转到对应翻译页面
3. 如果不存在，跳转到默认语言首页

### 按钮显示

- 英文页面显示：`中`
- 中文页面显示：`EN`

---

## 10.9 头像切换

### 位置

首页个人简介区域

### 功能

点击头像可以在两个头像之间切换：
- `mootutu.png`
- `weiqin.png`

### 实现

```javascript
function toggleAvatar() {
    const avatar = document.getElementById('avatar');
    if (avatar.src.includes('mootutu')) {
        avatar.src = '/assets/images/weiqin.png';
    } else {
        avatar.src = '/assets/images/mootutu.png';
    }
}
```

---

## 10.10 BibTeX 模态框

### 功能

点击论文的 `[bib]` 链接，弹出模态框显示 BibTeX 引用内容。

### 实现位置

在 `index.html` 中：

```html
<!-- 触发按钮 -->
<a href="javascript:void(0)" onclick="showBibtex('citation-id')">[bib]</a>

<!-- 模态框内容 -->
<div id="citation-id" class="bibtex-content" style="display:none;">
    <pre>@article{...}</pre>
</div>

<!-- 模态框容器 -->
<div id="bibtex-modal" class="modal">
    <div class="modal-content">
        <span class="close">&times;</span>
        <pre id="bibtex-text"></pre>
    </div>
</div>
```

---

[← 上一篇：配置文件](./09-config.md) | [返回目录](./README.md) | [下一篇：部署与发布 →](./11-deployment.md)
