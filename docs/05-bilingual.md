# 双语内容维护

## 5.1 双语架构原理

本站采用**目录镜像**方式实现双语支持：

```
/                           # 英文版本（默认）
├── index.html
├── blog.html
├── blog/posts/
├── learning.html
└── learning/

/cn/                        # 中文版本（镜像结构）
├── index.html
├── blog.html
├── blog/posts/
├── learning.html
└── learning/
```

### 核心特点

1. **目录结构完全镜像**：中文版本在 `/cn/` 目录下，保持与英文版本相同的结构
2. **通过 translate_url 链接**：每个页面通过 Front Matter 中的 `translate_url` 字段链接到对应翻译
3. **语言切换按钮**：页面右下角的语言切换按钮读取 `translate_url` 进行跳转

---

## 5.2 英文与中文目录对照

| 英文路径 | 中文路径 |
|----------|----------|
| `/index.html` | `/cn/index.html` |
| `/blog.html` | `/cn/blog.html` |
| `/blog/posts/article.html` | `/cn/blog/posts/article.html` |
| `/learning.html` | `/cn/learning.html` |
| `/learning/topic.html` | `/cn/learning/topic.html` |
| `/learning/topic/post.md` | `/cn/learning/topic/post.md` |
| `/about.html` | `/cn/about.html` |

---

## 5.3 翻译链接配置（translate_url）

### 配置方式

在每个页面的 Front Matter 中添加 `translate_url` 字段：

**英文页面示例：**
```yaml
---
layout: blog-post-layout
title: "Git Basics"
lang: en
translate_url: /cn/blog/posts/git-basics.html
---
```

**中文页面示例：**
```yaml
---
layout: blog-post-layout
title: "Git 基础教程"
lang: zh-CN
translate_url: /blog/posts/git-basics.html
---
```

### 注意事项

1. `translate_url` 必须是**绝对路径**（以 `/` 开头）
2. 英文页面指向中文版本，中文页面指向英文版本
3. 如果对应翻译不存在，可以暂时指向主页或留空

---

## 5.4 新增双语内容的工作流程

### 完整流程（以博客文章为例）

```
1. 创建英文文章
   blog/posts/new-article.html

2. 创建中文文章
   cn/blog/posts/new-article.html

3. 配置互相链接
   - 英文文章: translate_url: /cn/blog/posts/new-article.html
   - 中文文章: translate_url: /blog/posts/new-article.html

4. 更新英文索引
   blog.html

5. 更新中文索引
   cn/blog.html

6. 添加封面图（共用）
   assets/images/covers/new-article.png
```

### 检查清单

- [ ] 英文版本文件已创建
- [ ] 中文版本文件已创建
- [ ] 英文版本的 `translate_url` 指向中文版本
- [ ] 中文版本的 `translate_url` 指向英文版本
- [ ] 英文版本的 `lang` 设为 `en`
- [ ] 中文版本的 `lang` 设为 `zh-CN`
- [ ] 英文索引页已更新
- [ ] 中文索引页已更新

---

## 5.5 UI 文本翻译维护

### 配置文件位置

```
_data/ui_text.yml
```

### 文件结构

```yaml
en:
  home: Home
  blog: Blog
  learning: Learning
  about: About
  language_name: 中文
  language_url: /cn/index.html
  bio_include: bio.html
  back_to_blog: "← Back to Blog List"
  author: "Author"
  published: "Published"
  reading_time: "Reading Time"
  minutes: "min"
  category: "Category"
  search_placeholder: "Search..."
  no_results: "No results found"

zh-CN:
  home: 首页
  blog: 博客
  learning: 学习记录
  about: 关于
  language_name: English
  language_url: /index.html
  bio_include: bio-cn.html
  back_to_blog: "← 返回博客列表"
  author: "作者"
  published: "发布日期"
  reading_time: "阅读时长"
  minutes: "分钟"
  category: "分类"
  search_placeholder: "搜索..."
  no_results: "未找到相关结果"
```

### 字段说明

| 字段 | 说明 | 使用位置 |
|------|------|----------|
| `home` | 首页导航文字 | 导航栏 |
| `blog` | 博客导航文字 | 导航栏 |
| `learning` | 学习导航文字 | 导航栏 |
| `about` | 关于导航文字 | 导航栏 |
| `language_name` | 语言切换按钮显示的目标语言名 | 语言切换按钮 |
| `language_url` | 语言切换的默认跳转地址 | 语言切换按钮 |
| `bio_include` | 使用的个人简介组件 | 首页 |
| `back_to_blog` | 返回博客列表按钮 | 博客文章页 |
| `author` | 作者标签 | 文章元信息 |
| `published` | 发布日期标签 | 文章元信息 |
| `reading_time` | 阅读时间标签 | 文章元信息 |
| `minutes` | 分钟单位 | 文章元信息 |
| `category` | 分类标签 | 文章元信息 |
| `search_placeholder` | 搜索框占位文字 | 导航栏搜索 |
| `no_results` | 无搜索结果提示 | 搜索结果 |

### 如何在模板中使用

```liquid
{% assign ui = site.data.ui_text[page.lang] %}

<a href="/">{{ ui.home }}</a>
<span>{{ ui.author }}: {{ page.author }}</span>
```

### 添加新的 UI 文本

1. 在 `_data/ui_text.yml` 中添加新字段
2. 同时添加 `en` 和 `zh-CN` 两个版本
3. 在模板中使用 `{{ ui.new_field }}` 引用

---

## 5.6 常见双语维护场景

### 场景 1：只创建英文版本，暂无中文

```yaml
# 英文文章
---
translate_url: /cn/index.html  # 暂时指向中文首页
---
```

后续创建中文版本时再更新 `translate_url`。

### 场景 2：修改了英文内容，需要同步中文

1. 完成英文版本修改
2. 打开对应的中文版本
3. 同步翻译修改的内容
4. 一起提交

### 场景 3：添加新的 UI 文本

示例：添加"标签"文本

```yaml
# _data/ui_text.yml
en:
  # ... 其他字段
  tags: "Tags"

zh-CN:
  # ... 其他字段
  tags: "标签"
```

---

## 5.7 语言代码规范

| 语言 | 代码 | 使用场景 |
|------|------|----------|
| 英文 | `en` | Front Matter 的 `lang` 字段 |
| 中文 | `zh-CN` | Front Matter 的 `lang` 字段 |

**注意**：必须使用 `zh-CN` 而不是 `zh` 或 `chinese`。

---

## 5.8 双语内容最佳实践

### 文件命名

中英文版本使用**相同的文件名**，便于对应和管理：

```
blog/posts/git-basics.html        # 英文
cn/blog/posts/git-basics.html     # 中文（同名）
```

### 图片资源

图片资源可以**共用**，不需要在 `/cn/` 目录下再复制一份：

```html
<!-- 英文和中文都使用相同的图片路径 -->
<img src="/assets/images/covers/git-basics.png">
```

### 保持同步

建议在每次更新内容时，同时更新两个语言版本，避免版本差异过大。

---

[← 上一篇：学习内容维护](./04-学习内容维护.md) | [返回目录](./README.md) | [下一篇：静态资源管理 →](./06-静态资源管理.md)
