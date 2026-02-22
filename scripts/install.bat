@echo off
chcp 65001 >nul
REM hibro One-Click Installation Script (Windows)

setlocal enabledelayedexpansion

REM Change to project root directory (parent of scripts folder)
cd /d "%~dp0.."

echo.
echo ========================================
echo   hibro Intelligent Memory System
echo   Installation Script
echo ========================================
echo.

REM Check Python
echo [1/7] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python not found
    echo   Please install Python 3.10+ first
    echo   Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo   [OK] Python version: %PYTHON_VERSION%

REM Install dependencies
echo.
echo [2/7] Installing Python dependencies...
python -m pip install --upgrade pip -q
if errorlevel 1 echo   [WARN] Pip upgrade failed, continuing...

python -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo   [WARN] Trying domestic mirror...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple -q
)

REM Install hibro package
echo.
echo [3/7] Installing hibro package...
python -m pip install -e . -q
if errorlevel 1 (
    echo   [ERROR] Package installation failed
    pause
    exit /b 1
)
echo   [OK] hibro package installed

REM Install MCP SDK
echo.
echo [4/7] Installing MCP SDK...
python -m pip install mcp -q
if errorlevel 1 (
    echo   [WARN] Trying domestic mirror...
    python -m pip install mcp -i https://pypi.tuna.tsinghua.edu.cn/simple -q
)

REM Create directories
echo.
echo [5/7] Creating data directories...
if not exist "%USERPROFILE%\.hibro" mkdir "%USERPROFILE%\.hibro"
if not exist "%USERPROFILE%\.hibro\data" mkdir "%USERPROFILE%\.hibro\data"
if not exist "%USERPROFILE%\.hibro\backups" mkdir "%USERPROFILE%\.hibro\backups"
if not exist "%USERPROFILE%\.hibro\logs" mkdir "%USERPROFILE%\.hibro\logs"
if not exist "%USERPROFILE%\.hibro\cache" mkdir "%USERPROFILE%\.hibro\cache"
echo   [OK] Data directories created

REM Create default configuration
echo.
echo [6/7] Creating configuration file...
if not exist "%USERPROFILE%\.hibro\config.yaml" (
    (
        echo # hibro Configuration File
        echo data_directory: %USERPROFILE%\.hibro\data
        echo.
        echo memory:
        echo   auto_learn: true
        echo   importance_threshold: 0.7
        echo   max_memories: 100000
        echo.
        echo forgetting:
        echo   time_decay_rate: 0.1
        echo   min_importance: 0.3
        echo   cleanup_interval_days: 7
        echo.
        echo ide:
        echo   type: auto
        echo   auto_inject: true
        echo   context_limit_kb: 200
        echo   monitor_conversations: true
        echo   injection_strategy: smart
        echo.
        echo ide_integration:
        echo   auto_inject: true
        echo   context_limit_kb: 200
        echo   monitor_conversations: true
        echo.
        echo security:
        echo   encryption_enabled: true
        echo   auto_cleanup_days: 365
        echo   sensitive_data_filter: true
        echo.
        echo storage:
        echo   database_path: %USERPROFILE%\.hibro\memories.db
        echo   max_size_gb: 10
        echo   backup_enabled: true
        echo   backup_interval_hours: 24
        echo.
        echo performance:
        echo   cache_size_mb: 100
        echo   enable_compression: true
        echo.
        echo logging:
        echo   level: INFO
        echo   file: %USERPROFILE%\.hibro\logs\hibro.log
    ) > "%USERPROFILE%\.hibro\config.yaml"
    echo   [OK] Configuration file created
) else (
    echo   [INFO] Configuration file already exists
)

REM Configure Claude Code MCP
echo.
echo [7/7] Configuring Claude Code MCP integration...

REM Run Python configuration script (uses sys.executable internally)
python "%~dp0setup_claude_config.py"
if errorlevel 1 (
    echo   [WARN] Claude Code configuration failed
) else (
    echo   [OK] Claude Code MCP integration configured
)

REM Verify installation
echo.
echo ========================================
echo   Verification
echo ========================================
python -c "import hibro; print('  [OK] hibro installation successful!')"

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Usage:
echo   hibro tools:          mcp__hibro__get_quick_context
echo   Store memory:         mcp__hibro__remember
echo   Search memories:      mcp__hibro__search_memories
echo.
echo Configuration files:
echo   ~/.hibro/config.yaml       hibro configuration
echo   ~/.claude.json             Claude Code MCP config
echo   ~/.claude/settings.json    Claude Code settings
echo.
echo Uninstall:
echo   scripts\uninstall.bat
echo.
pause
