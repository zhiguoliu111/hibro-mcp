@echo off
chcp 65001 >nul
REM hibro Installation Verification Script

echo.
echo ========================================
echo   hibro Installation Verification
echo ========================================
echo.

echo [1/5] Checking hibro package...
python -c "import hibro; print('  [OK] hibro package installed')" 2>nul
if errorlevel 1 (
    echo   [ERROR] hibro package not installed
)

echo.
echo [2/5] Checking configuration files...
if exist "%USERPROFILE%\.hibro\config.yaml" (
    echo   [OK] Configuration file exists
) else (
    echo   [WARN] Configuration file missing
)

echo.
echo [3/5] Checking data directories...
if exist "%USERPROFILE%\.hibro\data" (
    echo   [OK] Data directory exists
) else (
    echo   [WARN] Data directory missing
)

echo.
echo [4/5] Checking Claude Code MCP config...
python -c "import json; from pathlib import Path; c=json.load(open(Path.home()/'.claude.json')); print('  [OK] hibro MCP configured') if 'hibro' in c.get('mcpServers',{}) else print('  [ERROR] hibro MCP missing')" 2>nul
if errorlevel 1 (
    echo   [ERROR] Claude Code config not found
)

echo.
echo [5/5] Checking hibro tool permissions...
python -c "import json; from pathlib import Path; s=json.load(open(Path.home()/'.claude'/'settings.json')); perms=[p for p in s.get('permissions',{}).get('allow',[]) if 'hibro' in p]; print(f'  [OK] {len(perms)} hibro permissions configured')" 2>nul
if errorlevel 1 (
    echo   [WARN] No hibro permissions found
)

echo.
echo ========================================
echo   Verification Complete!
echo ========================================
echo.
pause
