# Inno Agent 数据分析助手

> 基于 [Inno Agent](https://github.com/hhyqhh/inno-agent) 二次开发的智能数据分析工作区，面向结构化数据分析任务，提供从数据导入、探索性分析到统计建模、诊断与报告生成的可追溯工作流。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Node](https://img.shields.io/badge/Node-%3E%3D20.6.0-brightgreen.svg)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

## 项目简介

本项目将数据分析助手集成到 Inno Agent 的工作区预设体系中。用户可在网页端选择“数据分析助手”工作区，上传结构化数据并以自然语言描述研究问题；助手会在明确数据质量、变量角色和分析目标后，按规范化流程完成数据准备、统计分析、模型诊断与最终报告输出。

项目强调以下原则：

- 数据处理与建模方案在执行前清晰呈现；
- 区分事实、统计推断与不确定性；
- 对缺失数据、共线性、类别样本量、模型假设和诊断结果进行约束检查；
- 保留可复现的中间产物、分析脚本与报告清单；
- 不将 API Key、用户上传数据、会话记录或运行时缓存提交到仓库。

## 核心能力

- **数据理解与质量检查**：识别表格结构、变量角色、缺失情况、重复值和异常输入。
- **探索性分析**：生成面向研究问题的描述统计、分布、分组比较与关联分析建议。
- **规范化数据准备**：在确认方案后执行清洗、变量转换与数据派生，并保留过程记录。
- **统计建模与诊断**：支持回归、分类、计数与有序结果等常见分析路径，并对模型适用条件、非线性、过度离散、零膨胀、比例优势等风险进行检查。
- **可解释结果输出**：生成包含分析方法、关键发现、局限性与风险提示的最终 HTML 报告。
- **完整交互支持**：包括数据上传、用户确认、流式输出、工作区预设选择及中文/英文界面文案。

## 工作流程

```text
研究问题与数据上传
          ↓
数据结构识别与探索性分析
          ↓
数据准备方案确认 → 数据清洗与派生
          ↓
分析任务与模型方案确认
          ↓
统计建模、诊断与稳健性检查
          ↓
最终报告与结果清单输出
```

每次分析的产物按会话分开保存：原始输入保持只读，中间文件存放在 `work/`，最终报告和清单存放在 `outputs/`。

## 项目结构

```text
apps/inno-agent/
├─ presets/data-analysis-assistant/  # 数据分析助手预设、Skills 和报告模板
├─ scripts/                          # Python 环境安装与自动化冒烟测试
├─ src/agent/                        # 运行策略、环境管理、确认与输出逻辑
└─ web/src/                          # 上传、工作区选择和对话交互界面
```

其中，数据分析助手预设位于：

```text
apps/inno-agent/presets/data-analysis-assistant/
```

## 快速启动

### 1. 环境要求

- Node.js 20.6 或更高版本；
- npm；
- Python 3（首次使用数据分析能力或执行测试时需要）；
- 自行配置可用的大语言模型服务。项目不提供 API Key。

### 2. 安装并构建

在项目根目录执行：

```powershell
npm.cmd ci
npm.cmd run build
```

### 3. 准备本地运行目录

```powershell
New-Item -ItemType Directory -Force runtime\config, runtime\data, runtime\skills, workspace
Copy-Item config.example.json runtime\config\config.json
```

随后根据 `config.example.json` 的字段说明，在本地配置模型服务与 API Key。请勿提交 `runtime/config/config.json`、`.env` 或任何真实凭据。

### 4. 启动服务

```powershell
npm.cmd run server -- --home ./runtime --workspace ./workspace --port 3000
```

浏览器访问 [http://localhost:3000](http://localhost:3000)，在预设工作区中选择“数据分析助手”即可开始使用。

### 5. 首次准备 Python 分析环境（可选）

首次执行统计分析时，系统会按提示创建本地 Python 环境。也可以预先执行：

```powershell
$env:INNO_DATA_DIR = "$PWD\runtime\data"
node apps/inno-agent/scripts/setup-data-analysis-env.mjs
```

## 测试

执行完整构建与数据分析助手冒烟测试：

```powershell
npm.cmd run test:data-analysis
```

测试覆盖数据分析工作流、用户确认、变量支持、模型选择、模型诊断、报告输出和运行策略等关键路径。

## 安全与隐私

以下目录和文件仅用于本地运行，已通过 `.gitignore` 排除，不应提交到仓库：

```text
runtime/
workspace/
tmp/
work/
.env
node_modules/
```

使用公开数据进行演示时，也应确认数据可公开、已脱敏且符合其许可证要求。

## 致谢与许可证

本项目基于 [hhyqhh/inno-agent](https://github.com/hhyqhh/inno-agent) 进行二次开发，保留原项目的 [MIT License](./LICENSE)。

Inno Agent 提供 Agent 运行时、工作区与预设机制、前后端基础能力；本项目在此基础上实现并集成数据分析助手工作区及其分析流程、统计检查与测试支持。
