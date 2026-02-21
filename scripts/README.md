# hibro 安装指南 / Installation Guide

hibro 智能记忆系统一键安装脚本 / One-click installation scripts for the hibro Intelligent Memory System.

---

## 快速开始 / Quick Start

### Windows

```batch
# 安装 / Install
scripts\install.bat

# 验证安装 / Verify installation
scripts\verify_install.bat

# 卸载 / Uninstall
scripts\uninstall.bat
```

### Linux / macOS

```bash
# 安装 / Install
chmod +x scripts/install.sh
./scripts/install.sh

# 卸载 / Uninstall
chmod +x scripts/uninstall.sh
./scripts/uninstall.sh
```

---

## 系统要求 / Requirements

- Python 3.10+
- pip

---

## 安装流程 / What the installer does

| 步骤 / Step | 说明 / Description |
|-------------|-------------------|
| 1 | 安装依赖包 / Install dependencies from requirements.txt |
| 2 | 安装 hibro 包 / Install hibro package (`pip install -e .`) |
| 3 | 安装 MCP SDK / Install MCP SDK for Claude Code |
| 4 | 创建数据目录 / Create data directories (`~/.hibro/`) |
| 5 | 创建配置文件 / Create configuration (`~/.hibro/config.yaml`) |
| 6 | 配置 Claude Code / Configure Claude Code MCP integration |

---

## 目录结构 / Directory Structure

安装后的文件结构 / After installation:

```
~/.hibro/                      # hibro 数据目录 / Data directory
├── config.yaml                # 配置文件 / Configuration
├── data/                      # 记忆数据库 / Memory database
├── backups/                   # 备份文件 / Backup files
├── logs/                      # 日志文件 / Log files
└── cache/                     # 缓存文件 / Cache files

~/.claude.json                 # Claude Code MCP 配置 / MCP server config
~/.claude/settings.json        # Claude Code 设置 / Permissions & hooks
```

---

## 配置文件说明 / Configuration Files

| 文件 / File | 用途 / Purpose |
|-------------|----------------|
| `~/.hibro/config.yaml` | hibro 系统配置 / System configuration |
| `~/.claude.json` | Claude Code MCP 服务器配置 / MCP server configuration |
| `~/.claude/settings.json` | Claude Code 权限和钩子 / Permissions and hooks |

---

## MCP 工具列表 / Available MCP Tools

安装后可在 Claude Code 中使用以下 hibro 工具 / After installation, these tools are available:

### 核心工具 / Core Tools

| 工具 / Tool | 说明 / Description |
|-------------|-------------------|
| `mcp__hibro__get_quick_context` | 加载用户上下文（优先调用）/ Load user context (call first!) |
| `mcp__hibro__get_preferences` | 获取用户偏好 / Get user preferences |
| `mcp__hibro__search_memories` | 关键词搜索 / Search by keywords |
| `mcp__hibro__remember` | 存储记忆 / Store new memory |
| `mcp__hibro__forget` | 删除记忆 / Delete memory |
| `mcp__hibro__get_status` | 系统状态 / System status |

### 智能分析 / Intelligent Analysis

| 工具 / Tool | 说明 / Description |
|-------------|-------------------|
| `mcp__hibro__analyze_conversation` | 从对话提取记忆 / Extract memories from conversation |
| `mcp__hibro__search_semantic` | 语义搜索 / Semantic search |
| `mcp__hibro__analyze_project_deeply` | 深度项目分析 / Deep project analysis |
| `mcp__hibro__predict_next_needs` | 预测用户需求 / Predict user needs |
| `mcp__hibro__build_knowledge_graph` | 构建知识图谱 / Build knowledge graph |

### 项目上下文 / Project Context

| 工具 / Tool | 说明 / Description |
|-------------|-------------------|
| `mcp__hibro__get_project_context` | 获取项目记忆 / Get project-specific memories |
| `mcp__hibro__set_project_context` | 设置项目上下文 / Set project context |
| `mcp__hibro__set_active_task` | 设置当前任务 / Set active task |
| `mcp__hibro__scan_project` | 扫描项目结构 / Scan project structure |

以及 40+ 更多工具... / And 40+ more tools...

---

## 故障排除 / Troubleshooting

### Python 未找到 / Python not found

确保已安装 Python 3.10+ 并添加到 PATH / Make sure Python 3.10+ is installed and in PATH:

```bash
python --version   # Windows
python3 --version  # Linux/macOS
```

### 安装失败 / Installation fails

1. 以管理员身份运行 (Windows) 或使用 `sudo` (Linux/macOS) / Run as administrator or with sudo
2. 更新 pip: `python -m pip install --upgrade pip`
3. 使用国内镜像 / Use domestic mirror (China): 安装脚本会自动尝试清华源 / Installer auto-retries with Tsinghua mirror

### Claude Code 未识别 hibro / Claude Code not detecting hibro

1. 安装后重启 Claude Code / Restart Claude Code after installation
2. 检查 `~/.claude.json` 是否包含 `hibro` / Check if hibro exists in mcpServers
3. 运行验证脚本 / Run verification: `scripts\verify_install.bat`

### 卸载时文件被锁定 / Files locked during uninstall (Windows)

部分文件可能被进程锁定，重启后手动删除 / Some files may be locked. Restart and manually delete:

```
%USERPROFILE%\.hibro
```

---

## 手动安装 / Manual Installation

如果脚本无法运行，可手动安装 / If scripts don't work, install manually:

```bash
# 1. 进入项目目录 / Navigate to project root
cd /path/to/hibro-mcp

# 2. 安装包 / Install package
pip install -e .

# 3. 创建目录 / Create directories
mkdir -p ~/.hibro/data ~/.hibro/backups ~/.hibro/logs ~/.hibro/cache

# 4. 配置 Claude Code / Configure Claude Code
python scripts/setup_claude_config.py
```

## 手动卸载 / Manual Uninstallation

```bash
# 1. 卸载包 / Uninstall package
pip uninstall -y hibro

# 2. 删除数据（可选，先备份！）/ Remove data (optional - backup first!)
rm -rf ~/.hibro

# 3. 清理 Claude Code 配置 / Clean Claude Code config
python scripts/cleanup_claude_config.py
```

---

## 脚本文件说明 / Script Files

| 文件 / File | 说明 / Description |
|-------------|-------------------|
| `install.bat` | Windows 安装脚本 / Windows installer |
| `install.sh` | Linux/macOS 安装脚本 / Linux/macOS installer |
| `uninstall.bat` | Windows 卸载脚本 / Windows uninstaller |
| `uninstall.sh` | Linux/macOS 卸载脚本 / Linux/macOS uninstaller |
| `setup_claude_config.py` | 配置 Claude Code（安装时调用）/ Configure Claude Code |
| `cleanup_claude_config.py` | 清理 Claude Code 配置（卸载时调用）/ Cleanup Claude Code config |
| `verify_install.bat` | 验证安装状态 / Verify installation |
