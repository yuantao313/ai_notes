---
name: nav-refresh
description: 扫描 docs/ 目录下所有 .md 文件，按目录结构自动更新 mkdocs.yml 的 nav 多级菜单。当用户提到"刷新目录"、"更新侧边栏"、"同步导航"、"nav refresh"时使用此 skill。
---

# MkDocs 导航刷新

直接运行本目录下的 Python 脚本即可完成导航刷新。

## 执行

```bash
python .agents/skills/nav-refresh/refresh_nav.py
```

无需任何参数，脚本自动：

1. 扫描 `docs/` 下所有 `.md` 文件（跳过 `{{...}}.md` 等垃圾文件）
2. 提取每个文件首个 `# ` 标题作为菜单显示名（无标题则用文件名）
3. 按子目录分组，根目录文件为一级菜单，子目录为折叠二级菜单
4. 每组内 `index.md` 排在首位，`00_`、`01_` 等数字前缀自动去除
5. 替换 `mkdocs.yml` 中 `nav:` 节点，保留其他配置不变
