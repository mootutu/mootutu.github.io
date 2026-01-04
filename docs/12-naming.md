# 命名规范汇总

## 12.1 文件命名规范

### 博客文章

| 规则 | 说明 | 示例 |
|------|------|------|
| 小写字母 | 所有字母小写 | `git-basics.html` |
| 连字符分隔 | 单词间用 `-` 分隔 | `my-new-post.html` |
| 扩展名 | 使用 `.html` | `article.html` |
| 描述性 | 名称应反映内容 | `modernizing-python.html` |

**正确示例**：
```
git-basics.html
modernizing-python.html
automating-tasks-python.html
```

**错误示例**：
```
Git_Basics.html      # 不要用下划线和大写
MyNewPost.HTML       # 不要用驼峰命名
my new post.html     # 不要有空格
```

### 学习文章

| 规则 | 说明 | 示例 |
|------|------|------|
| 小写字母 | 所有字母小写 | `python_style.md` |
| 下划线分隔 | 单词间用 `_` 分隔 | `python_language_conventions.md` |
| 扩展名 | 使用 `.md`（推荐）或 `.html` | `topic.md` |

**正确示例**：
```
python_style_conventions.md
python_language_conventions.md
part2.md
```

### 主题索引页

| 规则 | 说明 | 示例 |
|------|------|------|
| 小写字母 | 所有字母小写 | `python.html` |
| 简短 | 使用简洁的主题名 | `minimind.html` |
| 扩展名 | 使用 `.html` | `topic.html` |

---

## 12.2 目录命名规范

### 学习文章目录

| 规则 | 说明 | 示例 |
|------|------|------|
| 小写字母 | 所有字母小写 | `python_guidelines/` |
| 下划线分隔 | 单词间用 `_` 分隔 | `my_topic/` |
| 与主题相关 | 名称反映主题内容 | `minimind/` |

**正确示例**：
```
learning/
├── python_guidelines/
├── minimind/
└── machine_learning/
```

### 图片目录

| 规则 | 说明 | 示例 |
|------|------|------|
| 与文章同名 | 使用文章文件名 | `posts/git-basics/` |
| 连字符分隔 | 遵循文章命名规则 | `posts/my-article/` |

**正确示例**：
```
assets/images/posts/
├── modernizing-python/
├── git-basics/
└── automating-tasks-python/
```

---

## 12.3 图片命名规范

### 封面图

| 规则 | 说明 |
|------|------|
| 与文章文件名一致 | `git-basics.png` 对应 `git-basics.html` |
| 使用 PNG 或 JPG | 推荐 PNG 格式 |

**示例对应**：

| 文章 | 封面图 |
|------|--------|
| `blog/posts/git-basics.html` | `assets/images/covers/git-basics.png` |
| `blog/posts/my-post.html` | `assets/images/covers/my-post.png` |

### 文章内图片

| 命名方式 | 示例 |
|----------|------|
| 序号命名 | `image.png`, `image-1.png`, `image-2.png` |
| 描述性命名 | `vscode-config.png`, `terminal-output.png` |

---

## 12.4 Front Matter 字段规范

### 博客文章必填字段

```yaml
---
layout: blog-post-layout        # 固定值
title: "文章标题"               # 字符串，用引号包裹
date: 2025-12-31               # 日期格式 YYYY-MM-DD
author: "作者名"                # 字符串
category: "分类名"              # 字符串
excerpt: "列表摘要"             # 字符串
tags: ["Tag1", "Tag2"]          # 标签数组
reading_time: "8 minutes"       # 显示用时长
cover_image: /assets/images/covers/xxx.png # 封面图路径
lang: en                        # en 或 zh-CN
translate_url: /cn/blog/posts/xxx.html  # 绝对路径
---
```

### 学习文章必填字段

```yaml
---
layout: learning-post-layout    # 固定值
title: "文章标题"               # 字符串
date: 2025-12-31               # 日期格式
lang: en                        # en 或 zh-CN
topic: topic-name               # 主题 key
order: 1                        # 排序数字
topic_url: /learning/topic.html # 所属主题页面
translate_url: /cn/learning/xxx.html  # 翻译版本
mathjax: false                  # 可选，是否启用数学公式
---
```

### 主题页面必填字段

```yaml
---
layout: learning-topic-layout   # 固定值
title: "Topic Title"            # 主题标题
description: "简短描述"         # 主题描述
topic: topic-name               # 主题 key
order: 1                        # 主题排序
lang: en                        # en 或 zh-CN
translate_url: /cn/learning/topic.html  # 翻译版本
---
```

### 字段值规范

| 字段 | 格式要求 |
|------|----------|
| `title` | 字符串，建议用引号包裹 |
| `date` | `YYYY-MM-DD` 格式 |
| `lang` | 只能是 `en` 或 `zh-CN` |
| `translate_url` | 绝对路径，以 `/` 开头 |
| `topic_url` | 绝对路径，以 `/` 开头 |
| `mathjax` | 布尔值 `true` 或 `false` |
| `topic` | 主题 key（小写、连字符） |
| `order` | 数字排序，越小越靠前 |

---

## 12.5 URL 路径规范

### 路径格式

| 类型 | 格式 |
|------|------|
| 博客文章 | `/blog/posts/{filename}.html` |
| 学习主题 | `/learning/{topic}.html` |
| 学习文章 | `/learning/{topic}/{filename}.html` |
| 中文版本 | `/cn/...`（镜像结构） |

### 图片引用路径

| 类型 | 格式 |
|------|------|
| 封面图 | `/assets/images/covers/{filename}.png` |
| 文章图片 | `/assets/images/posts/{article}/{image}.png` |
| 通用图片 | `/assets/images/{filename}.png` |

### 路径注意事项

1. **使用绝对路径**：以 `/` 开头
2. **保持一致性**：同类内容使用相同格式
3. **注意大小写**：Linux 服务器区分大小写

---

## 12.6 CSS 类名规范

### 命名风格

使用 BEM 风格或连字符分隔：

```css
/* 块 */
.blog-post { }
.learning-topic { }

/* 元素 */
.blog-post-title { }
.blog-post-content { }

/* 修饰符 */
.blog-post--featured { }
.button--primary { }
```

### 常用类名

| 类名 | 用途 |
|------|------|
| `.blog-post` | 博客文章卡片 |
| `.blog-post-title` | 博客标题 |
| `.blog-post-content` | 博客内容 |
| `.learning-topic` | 学习主题 |
| `.entry-link` | 条目链接 |
| `.dark-mode` | 暗黑模式 |

---

## 12.7 语言代码规范

| 语言 | 代码 | 使用场景 |
|------|------|----------|
| 英文 | `en` | Front Matter 的 `lang` 字段 |
| 中文 | `zh-CN` | Front Matter 的 `lang` 字段 |

**注意**：必须使用 `zh-CN`，不能使用 `zh`、`chinese` 或其他变体。

---

## 12.8 Git 提交信息规范

### 格式

```
<type>: <subject>
```

### 类型

| 类型 | 说明 |
|------|------|
| `feat` | 新增功能、新文章 |
| `fix` | 修复问题 |
| `docs` | 文档更新 |
| `style` | 样式调整 |
| `refactor` | 代码重构 |
| `chore` | 杂项维护 |

### 示例

```bash
git commit -m "feat: Add blog post about Docker"
git commit -m "fix: Correct broken link in about page"
git commit -m "docs: Update maintenance documentation"
git commit -m "style: Improve mobile navigation"
```

---

## 12.9 规范速查表

### 文件命名速查

| 内容类型 | 命名规则 | 示例 |
|----------|----------|------|
| 博客文章 | `kebab-case.html` | `git-basics.html` |
| 学习文章 | `snake_case.md` | `python_style.md` |
| 封面图 | 与文章同名 | `git-basics.png` |
| 文章图片 | 序号或描述 | `image-1.png` |

### 路径速查

| 内容 | 路径 |
|------|------|
| 博客文章 | `/blog/posts/xxx.html` |
| 中文博客 | `/cn/blog/posts/xxx.html` |
| 学习文章 | `/learning/topic/xxx.html` |
| 封面图 | `/assets/images/covers/xxx.png` |

---

<div class="doc-nav">
  <a href="./11-deployment.md" class="doc-nav-card prev">
    <span class="doc-nav-label">上一篇</span>
    <span class="doc-nav-title">← 部署与发布</span>
  </a>
  <a href="./13-checklist.md" class="doc-nav-card next">
    <span class="doc-nav-label">下一篇</span>
    <span class="doc-nav-title">维护检查清单 →</span>
  </a>
</div>
