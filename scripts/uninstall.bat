@echo off
chcp 65001 >nul
REM hibro Uninstallation Script (Windows)

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   hibro Uninstallation Script
echo ========================================
echo.

REM Confirm uninstallation
set /p confirm="Uninstall hibro? This will delete all data (y/n): "
if /i not "%confirm%"=="y" (
    echo Uninstallation cancelled
    pause
    exit /b 0
)

echo.
echo [1/5] Uninstalling hibro package...
pip uninstall -y hibro >nul 2>&1
if errorlevel 1 (
    echo   [INFO] hibro not installed via pip
) else (
    echo   [OK] hibro package uninstalled
)

echo.
echo [2/5] Removing configuration files...
if exist "%USERPROFILE%\.hibro\config.yaml" (
    del /f "%USERPROFILE%\.hibro\config.yaml" 2>nul
    echo   [OK] Configuration files removed
) else (
    echo   [INFO] No configuration files found
)

echo.
echo [3/5] Removing data directories...
if exist "%USERPROFILE%\.hibro\data" rmdir /s /q "%USERPROFILE%\.hibro\data" 2>nul
if exist "%USERPROFILE%\.hibro\backups" rmdir /s /q "%USERPROFILE%\.hibro\backups" 2>nul
if exist "%USERPROFILE%\.hibro\logs" rmdir /s /q "%USERPROFILE%\.hibro\logs" 2>nul
if exist "%USERPROFILE%\.hibro\cache" rmdir /s /q "%USERPROFILE%\.hibro\cache" 2>nul
echo   [OK] Data directories removed

echo.
echo [4/5] Removing Claude Code MCP configuration...
python "%~dp0cleanup_claude_config.py"

echo.
echo [5/5] Final cleanup...
rmdir /s /q "%USERPROFILE%\.hibro" 2>nul
if exist "%USERPROFILE%\.hibro" (
    echo   [WARN] Some files may be locked
    echo   Manual deletion required: %USERPROFILE%\.hibro
) else (
    echo   [OK] Cleanup completed
)

echo.
echo ========================================
echo   Uninstallation Complete!
echo ========================================
echo.
echo Note:
echo   - If database file is locked, restart and manually delete:
echo     %USERPROFILE%\.hibro
echo.
pause
