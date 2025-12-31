# 布局与组件

## 8.1 布局系统概述

Jekyll 使用布局（Layout）系统来定义页面的 HTML 结构。内容文件通过 Front Matter 中的 `layout` 字段指定使用哪个布局。

### 布局继承

```
default.html (基础布局)
    ├── blog-layout.html
    ├── blog-post-layout.html
    ├── learning-layout.html
    ├── learning-topic-layout.html
    ├── learning-post-layout.html
    └── about-layout.html
```

---

## 8.2 布局文件一览

### 目录位置

```
_layouts/
├── default.html
├── blog-layout.html
├── blog-post-layout.html
├── learning-layout.html
├── learning-topic-layout.html
├── learning-post-layout.html
└── about-layout.html
```

### 布局用途说明

| 布局文件 | 用途 | 使用页面 |
|----------|------|----------|
| `default.html` | 基础布局，包含导航栏、暗黑模式 | 首页、关于页 |
| `blog-layout.html` | 博客列表页布局 | `blog.html` |
| `blog-post-layout.html` | 博客文章布局（含 TOC） | 博客文章 |
| `learning-layout.html` | 学习主题列表布局 | `learning.html` |
| `learning-topic-layout.html` | 单个学习主题布局 | 主题索引页 |
| `learning-post-layout.html` | 学习文章布局（含 TOC、MathJax） | 学习文章 |
| `about-layout.html` | 关于页面布局 | `about.html` |

---

## 8.3 各布局详解

### default.html

**用途**：基础布局，其他布局的父级

**包含功能**：
- HTML 头部（meta、CSS、字体）
- 导航栏组件
- 暗黑模式切换
- 语言切换按钮
- 返回顶部按钮
- 个人简介组件（首页）

**结构**：
```html
<!DOCTYPE html>
<html>
<head>
    <!-- Meta、CSS、字体 -->
</head>
<body>
    {% include navigation.html %}

    <div class="wrapper">
        {{ content }}
    </div>

    <!-- 浮动按钮：返回顶部、语言切换、暗黑模式 -->
    <!-- JavaScript -->
</body>
</html>
```

### blog-post-layout.html

**用途**：博客文章页面

**包含功能**：
- 文章标题、元信息（作者、日期、分类、阅读时间）
- 侧边栏目录（TOC）- 大屏幕显示
- Prism.js 代码高亮
- 返回博客列表按钮

**Front Matter 要求**：
```yaml
layout: blog-post-layout
title: "文章标题"
date: 2025-12-31
author: "作者名"
category: "分类"
lang: en
translate_url: /cn/blog/posts/xxx.html
```

### learning-post-layout.html

**用途**：学习文章页面

**包含功能**：
- 文章标题、元信息
- 侧边栏目录（TOC）
- MathJax 数学公式支持（可选）
- Prism.js 代码高亮
- 返回主题页按钮

**Front Matter 要求**：
```yaml
layout: learning-post-layout
title: "文章标题"
date: 2025-12-31
lang: en
topic_url: /learning/topic.html
translate_url: /cn/learning/topic/xxx.html
mathjax: false  # 是否启用数学公式
```

### learning-topic-layout.html

**用途**：学习主题索引页

**包含功能**：
- 主题标题和描述
- 文章链接列表
- 返回学习主页按钮

**Front Matter 要求**：
```yaml
layout: learning-topic-layout
lang: en
translate_url: /cn/learning/topic.html
```

---

## 8.4 可复用组件

### 组件目录

```
_includes/
├── navigation.html      # 导航栏
├── bio.html             # 英文个人简介
├── bio-cn.html          # 中文个人简介
├── floating-buttons.html     # 浮动按钮（未使用）
└── floating-buttons-css.html # 浮动按钮样式（未使用）
```

### navigation.html

**用途**：网站顶部导航栏

**包含功能**：
- 品牌名称（带闪烁光标动画）
- 导航链接（Home、Blog、Learning、About）
- 搜索功能（实时搜索）
- 活动页面高亮
- 毛玻璃（Glassmorphism）效果
- 响应式设计
- Bocchi GIF 装饰

**使用方式**：
```liquid
{% include navigation.html %}
```

### bio.html / bio-cn.html

**用途**：个人简介组件

**包含内容**：
- 头像（可点击切换）
- 姓名和职位
- 联系方式图标链接
- 简介文字

**使用方式**：
```liquid
{% include bio.html %}
{% include bio-cn.html %}
```

**动态选择**（根据语言）：
```liquid
{% assign ui = site.data.ui_text[page.lang] %}
{% include {{ ui.bio_include }} %}
```

---

## 8.5 修改布局的注意事项

### 修改前备份

```bash
cp _layouts/target-layout.html _layouts/target-layout.html.bak
```

### 测试所有相关页面

修改布局后，测试所有使用该布局的页面：

| 修改的布局 | 需要测试的页面 |
|------------|---------------|
| `default.html` | 首页、关于页 |
| `blog-post-layout.html` | 所有博客文章 |
| `learning-post-layout.html` | 所有学习文章 |

### 暗黑模式兼容

修改样式时，确保添加 `.dark-mode` 对应样式：

```css
/* 亮色模式 */
.element {
    background: #ffffff;
    color: #333333;
}

/* 暗黑模式 */
.dark-mode .element {
    background: #1a1a2e;
    color: #e2e8f0;
}
```

### 响应式设计

使用媒体查询确保不同屏幕尺寸的兼容：

```css
/* 桌面端 */
.sidebar {
    display: block;
}

/* 移动端 */
@media (max-width: 768px) {
    .sidebar {
        display: none;
    }
}
```

---

## 8.6 Liquid 模板语法

### 常用语法

**变量输出**：
```liquid
{{ page.title }}
{{ site.data.ui_text[page.lang].home }}
```

**条件判断**：
```liquid
{% if page.mathjax %}
    <!-- 加载 MathJax -->
{% endif %}
```

**循环**：
```liquid
{% for post in site.posts %}
    <a href="{{ post.url }}">{{ post.title }}</a>
{% endfor %}
```

**包含组件**：
```liquid
{% include navigation.html %}
{% include {{ ui.bio_include }} %}
```

### 常用变量

| 变量 | 说明 |
|------|------|
| `page.title` | 当前页面标题 |
| `page.date` | 当前页面日期 |
| `page.lang` | 当前页面语言 |
| `page.url` | 当前页面 URL |
| `page.content` | 当前页面内容 |
| `site.data.xxx` | `_data/xxx.yml` 中的数据 |
| `content` | 子布局/页面的内容 |

---

## 8.7 添加新布局

### 步骤

1. **创建布局文件**
   ```bash
   touch _layouts/new-layout.html
   ```

2. **定义布局结构**
   ```html
   ---
   layout: default
   ---

   <div class="new-layout-container">
       {{ content }}
   </div>

   <style>
       /* 布局样式 */
   </style>
   ```

3. **在页面中使用**
   ```yaml
   ---
   layout: new-layout
   title: "页面标题"
   ---
   ```

---

[← 上一篇：个人信息维护](./07-personal-info.md) | [返回目录](./README.md) | [下一篇：配置文件 →](./09-config.md)
