# 学习内容维护

## 4.1 学习内容概述

学习内容采用两层结构：

```
Learning Topic（学习主题）
    └── Learning Post（学习文章）
        └── Learning Post
        └── ...
```

### Learning Topic（学习主题）

- 主题索引页，用于组织一系列相关文章
- 例如：Python Guidelines、Minimind Learning

### Learning Post（学习文章）

- 具体的学习笔记/文章
- 属于某个主题
- 使用 Markdown 格式编写

### 使用场景

| 内容类型 | 适用场景 |
|----------|----------|
| Learning Topic | 一个系统化的学习领域（如 Python 规范、机器学习笔记） |
| Learning Post | 该领域下的具体知识点 |

---

## 4.2 目录结构

```
learning/
├── python.html                    # Python 主题索引页
├── python_guidelines/             # Python 文章目录
│   ├── python_style_conventions.md
│   └── python_language_conventions.md
├── minimind.html                  # Minimind 主题索引页
└── minimind/                      # Minimind 文章目录
    └── part2.md

cn/learning/
├── python.html
├── python_guidelines/
│   ├── python_style_conventions.html
│   └── python_language_conventions.html
├── minimind.html
└── minimind/
```

---

## 4.3 Learning Topic 维护

### 4.3.1 主题页面结构

主题索引页使用 `learning-topic-layout` 布局：

```html
---
layout: learning-topic-layout
lang: en
translate_url: /cn/learning/python.html
---

<div class="learning-header">
    <h1 class="learning-title">Python Guidelines 🐍</h1>
    <p class="learning-subtitle">
        主题描述文字...
    </p>
</div>

<div class="learning-entries">
    <a href="/learning/python_guidelines/article1.html" class="entry-link">文章标题 1</a>
    <a href="/learning/python_guidelines/article2.html" class="entry-link">文章标题 2</a>
</div>
```

### 4.3.2 新增学习主题

#### 步骤 1：创建主题索引页

创建 `learning/new-topic.html`：

```html
---
layout: learning-topic-layout
lang: en
translate_url: /cn/learning/new-topic.html
---

<div class="learning-header">
    <h1 class="learning-title">New Topic Name 🔥</h1>
    <p class="learning-subtitle">
        主题描述...
    </p>
</div>

<div class="learning-entries">
    <!-- 文章链接将在这里添加 -->
</div>
```

#### 步骤 2：创建文章目录

```bash
mkdir learning/new_topic
mkdir cn/learning/new_topic
```

#### 步骤 3：更新学习主页

编辑 `learning.html`，在 `<div class="learning-topics">` 中添加：

```html
<a href="/learning/new-topic.html" class="topic-card">
    <div class="topic-title">New Topic Name 🔥</div>
    <div class="topic-desc">主题简短描述</div>
</a>
```

#### 步骤 4：创建中文版本

- 创建 `cn/learning/new-topic.html`
- 更新 `cn/learning.html`

---

## 4.4 Learning Post 维护

### 4.4.1 Front Matter 字段

```yaml
---
layout: learning-post-layout
title: "文章标题"
date: 2025-12-31
lang: en
topic_url: /learning/python.html
translate_url: /cn/learning/python_guidelines/article.html
mathjax: false
---
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `layout` | 是 | 必须为 `learning-post-layout` |
| `title` | 是 | 文章标题 |
| `date` | 是 | 发布日期 |
| `lang` | 是 | `en` 或 `zh-CN` |
| `topic_url` | 是 | 所属主题页面的 URL |
| `translate_url` | 是 | 翻译版本 URL |
| `mathjax` | 否 | 是否启用数学公式，默认 `false` |

### 4.4.2 新增学习文章

#### 步骤 1：创建文章文件

```bash
# 复制模板
cp _templates/learning_post_template.md learning/topic_name/new-article.md
```

#### 步骤 2：编辑 Front Matter

```yaml
---
layout: learning-post-layout
title: "New Article Title"
date: 2025-12-31
lang: en
topic_url: /learning/topic-name.html
translate_url: /cn/learning/topic_name/new-article.html
mathjax: false
---
```

#### 步骤 3：编写 Markdown 内容

```markdown
## 第一部分

正文内容...

### 1.1 子标题

更多内容...

## 第二部分

```python
# 代码示例
def hello():
    print("Hello!")
```

```

**注意**：Markdown 文件会被 Jekyll 自动转换为 HTML。

#### 步骤 4：更新主题索引页

编辑主题页面（如 `learning/python.html`），添加文章链接：

```html
<div class="learning-entries">
    <a href="/learning/python_guidelines/new-article.html" class="entry-link">New Article Title</a>
    <!-- 注意：Markdown 文件的链接使用 .html 扩展名 -->
</div>
```

> ⚠️ **重要**：即使原文件是 `.md`，链接也要写成 `.html`！

#### 步骤 5：创建中文版本

- 创建 `cn/learning/topic_name/new-article.md`
- 更新 `cn/learning/topic-name.html`

#### 步骤 6：本地预览

```bash
jekyll serve
# 访问 http://localhost:4000/learning/topic_name/new-article.html
```

### 4.4.3 修改学习文章

1. 直接编辑 `.md` 文件
2. 同步修改中文版本
3. 本地预览
4. 提交推送

### 4.4.4 删除学习文章

1. 删除文章文件
   ```bash
   rm learning/topic_name/article.md
   rm cn/learning/topic_name/article.md
   ```

2. 从主题索引页删除链接

3. 提交更改

---

## 4.5 Markdown 编写规范

### 标题层级

```markdown
## H2 标题 - 主要章节
### H3 标题 - 子章节
#### H4 标题 - 更细分（较少使用）
```

- 文章内不要使用 H1（`#`），H1 由布局自动生成
- 标题会自动生成侧边目录

### 代码块

````markdown
```python
def example():
    return "Hello"
```

```bash
npm install
```
````

### 提示框（使用引用语法）

```markdown
> 💡 **Tip**: 这是一个提示信息

> ⚠️ **Warning**: 这是一个警告
```

### 表格

```markdown
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| A   | B   | C   |
```

### 图片

```markdown
![描述文字](/assets/images/posts/topic/image.png)
```

---

## 4.6 启用数学公式

如需在文章中使用 LaTeX 数学公式：

1. 在 Front Matter 中设置 `mathjax: true`
2. 使用标准 LaTeX 语法

行内公式：
```markdown
当 $a \ne 0$ 时，方程 $ax^2 + bx + c = 0$ 有两个解。
```

独立公式：
```markdown
$$
x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
$$
```

---

## 4.7 现有主题一览

| 主题 | 索引页 | 文章目录 | 说明 |
|------|--------|----------|------|
| Python Guidelines | `learning/python.html` | `learning/python_guidelines/` | Python 编码规范 |
| Minimind Learning | `learning/minimind.html` | `learning/minimind/` | 大模型学习笔记 |

---

## 4.8 学习主页维护（learning.html）

### 文件位置

- 英文：`/learning.html`
- 中文：`/cn/learning.html`

### 主题卡片模板

```html
<a href="/learning/topic-name.html" class="topic-card">
    <div class="topic-title">Topic Name 🔥</div>
    <div class="topic-desc">简短描述，说明这个主题包含什么内容</div>
</a>
```

---

[← 上一篇：博客文章维护](./03-blog-maintenance.md) | [返回目录](./README.md) | [下一篇：双语内容维护 →](./05-bilingual.md)
