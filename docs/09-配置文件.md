# 配置文件

## 9.1 配置文件概览

| 文件 | 用途 |
|------|------|
| `_config.yml` | Jekyll 主配置文件 |
| `_data/ui_text.yml` | 多语言 UI 文本配置 |

---

## 9.2 _config.yml 详解

### 文件位置

```
/_config.yml
```

### 当前配置内容

```yaml
# Site metadata
# keywords: Cryptography, Privacy, Information Security, Digital Signatures
# description: Academic Website of ABC
# title: Homepage of ABC

# author: ABC
# authorimg: assets/images/dd.png
# authoroccup: Engineering Manager

# inst: DFINITY
# insturl: https://www.dfinity.org
# uni: Graz University of Technology
# uniurl: http://www.tugraz.at

# vcard: 'mailto:david@dfinity.org'
# office: https://online.tugraz.at/tug_online/ris.einzelraum?raumkey=4944
# twitter: dderler
# keybase: dderler
# linkedin: david-derler-08630495
# dblp: hd/d/Derler:David
# scholar: YfMKI0wAAAAJ

# Build settings
markdown: kramdown
theme: minima
plugins:
  - jekyll-feed
collections:
  - proc
  - jour
```

### 配置项说明

#### 构建设置

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| `markdown` | `kramdown` | Markdown 解析器 |
| `theme` | `minima` | 基础主题 |
| `plugins` | `[jekyll-feed]` | 启用的插件列表 |
| `collections` | `[proc, jour]` | 自定义集合（会议/期刊论文） |

#### 站点元数据（已注释）

这些配置项当前被注释，可按需启用：

| 配置项 | 说明 |
|--------|------|
| `title` | 网站标题 |
| `description` | 网站描述 |
| `keywords` | SEO 关键词 |
| `author` | 作者名称 |
| `authorimg` | 作者头像路径 |

### 修改配置的影响

| 修改项 | 影响范围 | 需要重启 |
|--------|----------|----------|
| `markdown` | 所有 Markdown 文件的解析 | 是 |
| `theme` | 全站样式 | 是 |
| `plugins` | 插件功能 | 是 |
| 站点元数据 | SEO、页面标题 | 是 |

> **注意**：修改 `_config.yml` 后，需要重启 Jekyll 服务器才能生效。

### 添加新配置

```yaml
# 添加自定义配置
my_custom_setting: value

# 在模板中使用
# {{ site.my_custom_setting }}
```

---

## 9.3 _data/ui_text.yml 详解

### 文件位置

```
/_data/ui_text.yml
```

### 完整配置内容

```yaml
en:
  home: Home
  blog: Blog
  learning: Learning
  about: About
  tools: Tools
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
  tools: 工具记录
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

### 字段分类说明

#### 导航栏文本

| 字段 | 英文值 | 中文值 | 使用位置 |
|------|--------|--------|----------|
| `home` | Home | 首页 | 导航栏链接 |
| `blog` | Blog | 博客 | 导航栏链接 |
| `learning` | Learning | 学习记录 | 导航栏链接 |
| `about` | About | 关于 | 导航栏链接 |
| `tools` | Tools | 工具记录 | 导航栏链接 |

#### 语言切换

| 字段 | 英文值 | 中文值 | 说明 |
|------|--------|--------|------|
| `language_name` | 中文 | English | 切换按钮显示的目标语言 |
| `language_url` | /cn/index.html | /index.html | 默认切换目标 |

#### 个人简介

| 字段 | 英文值 | 中文值 | 说明 |
|------|--------|--------|------|
| `bio_include` | bio.html | bio-cn.html | 使用的简介组件 |

#### 文章元信息

| 字段 | 英文值 | 中文值 | 使用位置 |
|------|--------|--------|----------|
| `author` | Author | 作者 | 文章作者标签 |
| `published` | Published | 发布日期 | 发布日期标签 |
| `reading_time` | Reading Time | 阅读时长 | 阅读时间标签 |
| `minutes` | min | 分钟 | 时间单位 |
| `category` | Category | 分类 | 分类标签 |
| `back_to_blog` | ← Back to Blog List | ← 返回博客列表 | 返回按钮 |

#### 搜索功能

| 字段 | 英文值 | 中文值 | 使用位置 |
|------|--------|--------|----------|
| `search_placeholder` | Search... | 搜索... | 搜索框占位符 |
| `no_results` | No results found | 未找到相关结果 | 无结果提示 |

### 在模板中使用

```liquid
{% assign ui = site.data.ui_text[page.lang] %}

<!-- 导航栏 -->
<a href="/">{{ ui.home }}</a>
<a href="/blog.html">{{ ui.blog }}</a>

<!-- 文章元信息 -->
<span>{{ ui.author }}: {{ page.author }}</span>
<span>{{ ui.published }}: {{ page.date | date: "%B %d, %Y" }}</span>

<!-- 搜索 -->
<input placeholder="{{ ui.search_placeholder }}">
```

### 添加新的 UI 文本

1. **在 `ui_text.yml` 中添加**
   ```yaml
   en:
     # 现有字段...
     new_field: "New Text"

   zh-CN:
     # 现有字段...
     new_field: "新文本"
   ```

2. **在模板中使用**
   ```liquid
   {{ ui.new_field }}
   ```

---

## 9.4 其他配置

### Jekyll 默认配置

Jekyll 有一些默认配置，无需在 `_config.yml` 中声明：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `source` | `.` | 源文件目录 |
| `destination` | `./_site` | 构建输出目录 |
| `port` | `4000` | 本地服务器端口 |
| `host` | `127.0.0.1` | 本地服务器地址 |

### 覆盖默认配置

```yaml
# _config.yml
port: 3000  # 更改本地端口
host: 0.0.0.0  # 允许局域网访问
```

---

## 9.5 配置修改最佳实践

### 修改前

1. 备份当前配置
   ```bash
   cp _config.yml _config.yml.bak
   ```

2. 了解修改的影响范围

### 修改后

1. 重启 Jekyll 服务器
   ```bash
   # 停止当前服务器 (Ctrl+C)
   jekyll serve
   ```

2. 测试相关功能

3. 提交更改
   ```bash
   git add _config.yml
   git commit -m "config: Update xxx setting"
   ```

---

[← 上一篇：布局与组件](./08-布局与组件.md) | [返回目录](./README.md) | [下一篇：特殊功能 →](./10-特殊功能.md)
