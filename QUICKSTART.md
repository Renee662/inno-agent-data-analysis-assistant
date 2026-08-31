# Quickstart

本项目基于 Inno Agent 二次开发，集成“数据分析助手”工作区，用于完成结构化数据的导入、探索性分析、数据准备、统计建模、模型诊断与报告生成。

## 1. 环境要求

开始前请确认本机已安装：

- Node.js 20.6 或更高版本；
- npm；
- Python 3；
- 可用的大语言模型服务 API Key。

项目不会提供 API Key。请使用自己的模型服务配置，并且不要将任何真实密钥上传至 GitHub。

## 2. 获取并安装项目

在 PowerShell 中执行：

```powershell
git clone https://github.com/Renee662/inno-agent-data-analysis-assistant.git
cd inno-agent-data-analysis-assistant

npm.cmd ci
npm.cmd run build
```

构建成功后，后端与前端文件均已准备完成。

## 3. 创建本地运行目录

```powershell
New-Item -ItemType Directory -Force runtime\config, runtime\data, runtime\skills, workspace
Copy-Item config.example.json runtime\config\config.json
```

随后在本地编辑 `runtime\config\config.json`，根据 `config.example.json` 的字段说明填写模型服务地址、模型名称和 API Key。

`runtime/`、`workspace/`、`.env` 均为本地运行目录或文件，请勿提交至仓库。

## 4. 启动服务

```powershell
npm.cmd run server -- --home ./runtime --workspace ./workspace --port 3000
```

启动完成后，浏览器访问：

```text
http://localhost:3000
```

在欢迎页或工作区预设列表中选择“数据分析助手”，即可开始使用。

## 5. 首次使用数据分析能力

首次执行统计分析时，系统会按提示创建本地 Python 环境并安装所需依赖。请保持网络可用，并按界面提示完成环境准备。

如需提前安装数据分析环境，可执行：

```powershell
$env:INNO_DATA_DIR = "$PWD\runtime\data"
node apps/inno-agent/scripts/setup-data-analysis-env.mjs
```

## 6. 使用流程

1. 进入“数据分析助手”工作区；
2. 上传可公开、已脱敏的 CSV 或 Excel 数据；
3. 描述研究问题、目标变量和期望分析方向；
4. 查看并确认数据准备方案；
5. 查看并确认分析任务与模型方案；
6. 获取探索性分析、模型诊断和最终 HTML 报告。

系统会将原始输入、中间分析文件和最终报告按会话分别保存；请勿将含隐私信息的数据上传到公开环境。

## 7. 运行测试

完成 Python 环境准备后，可执行：

```powershell
npm.cmd run test:data-analysis
```

该命令会重新构建项目，并运行数据分析助手的自动化冒烟测试。

## 8. 常见问题

**构建后找不到页面或后端未更新**

重新执行：

```powershell
npm.cmd run build
```

然后停止并重新启动服务。

**端口 3000 被占用**

改用其他端口，例如：

```powershell
npm.cmd run server -- --home ./runtime --workspace ./workspace --port 3001
```

随后访问 `http://localhost:3001`。

**Python 环境安装失败**

确认本机已安装 Python 3，并重新执行“首次使用数据分析能力”中的环境准备命令。
