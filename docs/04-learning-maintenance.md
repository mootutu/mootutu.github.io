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
│   ├── python_style_conventions.md
│   └── python_language_conventions.md
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
title: "Python Guidelines 🐍"
description: "主题描述文字..."
topic: python
order: 1
lang: en
translate_url: /cn/learning/python.html
---
```

### 4.3.2 新增学习主题

#### 步骤 1：创建主题索引页

创建 `learning/new-topic.html`：

```html
---
layout: learning-topic-layout
title: "New Topic Name 🔥"
description: "主题描述..."
topic: new-topic
order: 3
lang: en
translate_url: /cn/learning/new-topic.html
---
```

#### 步骤 2：创建文章目录

```bash
mkdir learning/new_topic
mkdir cn/learning/new_topic
```

#### 步骤 3：创建中文版本

- 创建 `cn/learning/new-topic.html`
- 填写中文 `title`/`description`/`topic`/`order`

---

## 4.4 Learning Post 维护

### 4.4.1 Front Matter 字段

```yaml
---
layout: learning-post-layout
title: "文章标题"
date: 2025-12-31
lang: en
topic: python
order: 1
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
| `topic` | 是 | 主题 key（如 `minimind`） |
| `order` | 是 | 文章排序（数字） |
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
topic: topic-name
order: 1
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

#### 步骤 4：创建中文版本

- 创建 `cn/learning/topic_name/new-article.md`
- 设置 `topic` 和 `order` 与英文一致

#### 步骤 5：本地预览

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

2. 提交更改

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

学习主页会根据主题页的 Front Matter 自动生成卡片，无需手动编辑 `learning.html` 或 `cn/learning.html`。

---

<div class="doc-nav">
  <a href="./03-blog-maintenance.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 博客文章维护</span>
  </a>
  <a href="./05-bilingual.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">双语内容维护 →</span>
  </a>
</div>
