---
name: ai-notes-agent
description: AI 知识库全栈笔记助手。遵循"设计原因 → 函数速查 → 编程实战 → 硬件实现"模板，协助整理从芯片到 Agent 的七层知识体系。
---

# AI 知识库笔记规范

## 知识分层

```
01_ai_accelerator/         芯片与电路
02_xpu_programming/        XPU 编程（CUDA / Ascend）
03_frameworks/             系统软件
04_models/                 AI 框架
05_inference/              模型架构
06_training/               训练与推理
07_agent_application/      Agent 应用
```

## 单篇文章模板

### 设计原因

为什么需要这个东西？解决什么问题？用通俗语言和简单例子说明动机，可配 PTX 汇编展示问题本质。

### 函数速查

API 速查表，含接口签名、支持的类型、伪代码描述。附 Tips 说明易错点。

### 编程实战

一个简单的完整 kernel 示例。只放单个示例，不做多写法对比，综合运用留到独立文章。

### 硬件实现（扩展）

可选章节（标记"扩展"），深入 PTX/SASS 层面解释硬件原理。日常开发无需掌握。

## 导航维护

文档目录变更后执行：

```bash
source .venv/bin/activate && python .agents/skills/nav-refresh/refresh_nav.py
```

脚本自动扫描 `docs/` 下所有 `.md` 文件，提取 `# ` 标题，按目录结构更新 `mkdocs.yml` 的 `nav` 配置。

## 构建

```bash
source .venv/bin/activate && mkdocs build    # 构建站点
source .venv/bin/activate && mkdocs serve    # 本地预览
```
