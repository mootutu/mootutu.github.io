# 个人信息维护

## 7.1 个人简介更新

### 文件位置

| 语言 | 文件路径 |
|------|----------|
| 英文 | `_includes/bio.html` |
| 中文 | `_includes/bio-cn.html` |

### 简介结构

个人简介组件包含：
- 头像图片
- 姓名
- 职位/身份
- 联系方式链接
- 简介文字

### 修改步骤

1. **打开对应文件**
   ```bash
   # 英文简介
   code _includes/bio.html

   # 中文简介
   code _includes/bio-cn.html
   ```

2. **修改内容**
   - 姓名、职位等文字直接编辑
   - 联系方式链接更新 `href` 属性
   - 简介段落更新文字内容

3. **本地预览**
   ```bash
   jekyll serve
   # 访问 http://localhost:4000
   ```

4. **提交更改**
   ```bash
   git add _includes/bio.html _includes/bio-cn.html
   git commit -m "docs: Update personal bio"
   git push
   ```

### 注意事项

- 英文和中文简介需要分别维护
- 修改后检查两个版本的一致性

---

## 7.2 头像更新

### 头像文件位置

```
assets/images/
├── mootutu.png      # 主头像
└── weiqin.png       # 备用头像（点击切换）
```

### 更换头像

1. **准备新头像图片**
   - 推荐尺寸：200x200 ~ 400x400 像素
   - 格式：PNG（支持透明背景）或 JPG

2. **替换文件**
   ```bash
   # 替换主头像
   cp ~/Downloads/new-avatar.png assets/images/mootutu.png

   # 或替换备用头像
   cp ~/Downloads/new-avatar-alt.png assets/images/weiqin.png
   ```

3. **保持文件名不变**
   - 如果使用相同文件名，无需修改代码
   - 如果使用新文件名，需要更新 `_includes/bio.html` 中的引用

4. **清除缓存预览**
   - 浏览器可能缓存旧图片
   - 使用 Ctrl+F5（Windows）或 Cmd+Shift+R（Mac）强制刷新

---

## 7.3 联系方式更新

### 联系方式位置

联系方式链接在 `_includes/bio.html` 和 `_includes/bio-cn.html` 中定义。

### 常见联系方式

```html
<!-- 邮箱 -->
<a href="mailto:your-email@example.com">
    <i class="fas fa-envelope"></i>
</a>

<!-- GitHub -->
<a href="https://github.com/username">
    <i class="fab fa-github"></i>
</a>

<!-- LinkedIn -->
<a href="https://linkedin.com/in/username">
    <i class="fab fa-linkedin"></i>
</a>

<!-- Twitter/X -->
<a href="https://twitter.com/username">
    <i class="fab fa-twitter"></i>
</a>

<!-- Google Scholar -->
<a href="https://scholar.google.com/citations?user=xxx">
    <i class="fas fa-graduation-cap"></i>
</a>
```

### 添加新联系方式

1. 在 bio.html 中找到联系方式区域
2. 添加新的链接元素
3. 使用 FontAwesome 图标
4. 同步更新中文版本

---

## 7.4 添加出版物

### 出版物位置

出版物信息在首页文件中：
- 英文：`index.html`
- 中文：`cn/index.html`

### 出版物结构

```html
<div class="publication">
    <div class="pub-title">
        <a href="论文链接">论文标题</a>
    </div>
    <div class="pub-authors">
        作者列表（粗体标注自己）
    </div>
    <div class="pub-venue">
        发表会议/期刊, 年份
    </div>
    <div class="pub-links">
        <a href="论文PDF链接">[PDF]</a>
        <a href="javascript:void(0)" onclick="showBibtex('bibtex-id')">[bib]</a>
    </div>
</div>
```

### 添加新出版物步骤

#### 步骤 1：创建 BibTeX 文件

在 `assets/bibtex/` 目录创建 `.bib` 文件：

```bash
# 文件名格式：作者年份关键词.bib
touch assets/bibtex/wang2025newtopic.bib
```

BibTeX 内容示例：
```bibtex
@inproceedings{wang2025newtopic,
  title={Paper Title Here},
  author={Wang, Weiqin and Others, Some},
  booktitle={Conference Name},
  year={2025}
}
```

#### 步骤 2：更新首页

编辑 `index.html`，在 Publications 部分添加：

```html
<!-- 新论文 -->
<div class="publication">
    <div class="pub-title">
        <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
            Paper Title Here
        </a>
    </div>
    <div class="pub-authors">
        <strong>Weiqin Wang</strong>, Some Others
    </div>
    <div class="pub-venue">
        Conference Name (CONF), 2025
    </div>
    <div class="pub-links">
        <a href="论文PDF链接" target="_blank">[PDF]</a>
        <a href="javascript:void(0)" onclick="showBibtex('wang2025newtopic')">[bib]</a>
    </div>
</div>

<!-- BibTeX 模态框内容 -->
<div id="wang2025newtopic" class="bibtex-content" style="display:none;">
    <pre>@inproceedings{wang2025newtopic,
  title={Paper Title Here},
  author={Wang, Weiqin and Others, Some},
  booktitle={Conference Name},
  year={2025}
}</pre>
</div>
```

#### 步骤 3：同步中文版本

在 `cn/index.html` 中添加相同内容（可翻译标题和会议名）。

#### 步骤 4：验证

1. 本地预览首页
2. 点击 [bib] 链接检查弹窗
3. 测试 PDF 链接

---

## 7.5 更新简历

### 简历文件位置

```
resume.pdf
```

### 更新步骤

1. **准备新简历**
   - 导出为 PDF 格式
   - 命名为 `resume.pdf`

2. **替换文件**
   ```bash
   cp ~/Downloads/new-resume.pdf resume.pdf
   ```

3. **提交更改**
   ```bash
   git add resume.pdf
   git commit -m "docs: Update resume"
   git push
   ```

### 简历链接位置

简历链接通常在个人简介或关于页面中：
```html
<a href="/resume.pdf" target="_blank">Resume/CV</a>
```

---

## 7.6 更新网站图标

### 图标文件位置

```
assets/images/ggbond.ico
```

### 更新步骤

1. **准备新图标**
   - 格式：ICO 或 PNG
   - 推荐尺寸：32x32 或 64x64 像素

2. **替换文件**
   ```bash
   cp ~/Downloads/new-favicon.ico assets/images/ggbond.ico
   ```

3. **如使用不同文件名**

   需要更新布局文件中的引用：
   ```html
   <link rel="icon" type="image/x-icon" href="/assets/images/new-name.ico">
   ```

---

## 7.7 维护检查清单

更新个人信息后，检查以下项目：

- [ ] 英文版本已更新
- [ ] 中文版本已同步更新
- [ ] 本地预览效果正常
- [ ] 链接可正常访问
- [ ] 图片正常显示
- [ ] 暗黑模式下显示正常

---

[← 上一篇：静态资源管理](./06-静态资源管理.md) | [返回目录](./README.md) | [下一篇：布局与组件 →](./08-布局与组件.md)
