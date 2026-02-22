@echo off
chcp 65001 >nul
REM hibro Uninstallation Script (Windows)

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   hibro Uninstallation Script
echo ========================================
echo.

echo Select uninstallation mode:
echo   [1] Full uninstall (remove all data including memories)
echo   [2] Keep data (only remove program, preserve memories)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    set "mode=full"
) else if "%choice%"=="2" (
    set "mode=keepdata"
) else (
    echo Invalid choice. Uninstallation cancelled.
    pause
    exit /b 1
)

echo.
echo [1/4] Uninstalling hibro package...
pip uninstall -y hibro >nul 2>&1
if errorlevel 1 (
    echo   [INFO] hibro not installed via pip
) else (
    echo   [OK] hibro package uninstalled
)

echo.
echo [2/4] Removing configuration files...
if exist "%USERPROFILE%\.hibro\config.yaml" (
    del /f "%USERPROFILE%\.hibro\config.yaml" 2>nul
    echo   [OK] Configuration files removed
) else (
    echo   [INFO] No configuration files found
)

echo.
echo [3/4] Removing data directories...
REM Always remove logs and cache (temporary data)
if exist "%USERPROFILE%\.hibro\logs" rmdir /s /q "%USERPROFILE%\.hibro\logs" 2>nul
if exist "%USERPROFILE%\.hibro\cache" rmdir /s /q "%USERPROFILE%\.hibro\cache" 2>nul

if "%mode%"=="full" (
    if exist "%USERPROFILE%\.hibro\data" rmdir /s /q "%USERPROFILE%\.hibro\data" 2>nul
    if exist "%USERPROFILE%\.hibro\backups" rmdir /s /q "%USERPROFILE%\.hibro\backups" 2>nul
    if exist "%USERPROFILE%\.hibro\memories.db" del /f "%USERPROFILE%\.hibro\memories.db" 2>nul
    if exist "%USERPROFILE%\.hibro\memories.db-wal" del /f "%USERPROFILE%\.hibro\memories.db-wal" 2>nul
    if exist "%USERPROFILE%\.hibro\memories.db-shm" del /f "%USERPROFILE%\.hibro\memories.db-shm" 2>nul
    echo   [OK] All data removed
) else (
    echo   [OK] Memory data preserved
)

echo.
echo [4/4] Removing Claude Code MCP configuration...
python "%~dp0cleanup_claude_config.py"

echo.
if "%mode%"=="full" (
    rmdir /s /q "%USERPROFILE%\.hibro" 2>nul
    if exist "%USERPROFILE%\.hibro" (
        echo   [WARN] Some files may be locked
        echo   Manual deletion required: %USERPROFILE%\.hibro
    ) else (
        echo   [OK] Full cleanup completed
    )
) else (
    echo   [OK] Program removed, data preserved in: %USERPROFILE%\.hibro
)

echo.
echo ========================================
echo   Uninstallation Complete!
echo ========================================
echo.
if "%mode%"=="keepdata" (
    echo Note:
    echo   - Memory data has been preserved
    echo   - Reinstall hibro to continue using your memories
    echo   - To fully remove, manually delete: %USERPROFILE%\.hibro
) else (
    echo Note:
    echo   - If database file is locked, restart and manually delete:
    echo     %USERPROFILE%\.hibro
)
echo.
pause
