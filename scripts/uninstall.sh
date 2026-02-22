#!/usr/bin/env bash
# hibro Uninstallation Script (Linux/macOS)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  hibro Uninstallation Script"
echo "========================================"
echo ""

echo "Select uninstallation mode:"
echo "  [1] Full uninstall (remove all data including memories)"
echo "  [2] Keep data (only remove program, preserve memories)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    mode="full"
elif [ "$choice" = "2" ]; then
    mode="keepdata"
else
    echo "Invalid choice. Uninstallation cancelled."
    exit 1
fi

echo ""
echo "[1/4] Uninstalling hibro package..."
pip3 uninstall -y hibro > /dev/null 2>&1 || echo "  [INFO] hibro not installed via pip"
echo "  [OK] hibro package uninstalled"

echo ""
echo "[2/4] Removing configuration files..."
rm -f ~/.hibro/config.yaml 2>/dev/null
echo "  [OK] Configuration files removed"

echo ""
echo "[3/4] Removing data directories..."
# Always remove logs and cache (temporary data)
rm -rf ~/.hibro/logs ~/.hibro/cache 2>/dev/null

if [ "$mode" = "full" ]; then
    rm -rf ~/.hibro/data ~/.hibro/backups 2>/dev/null
    rm -f ~/.hibro/memories.db ~/.hibro/memories.db-wal ~/.hibro/memories.db-shm 2>/dev/null
    echo "  [OK] All data removed"
else
    echo "  [OK] Memory data preserved"
fi

echo ""
echo "[4/4] Removing Claude Code MCP configuration..."
python3 "$SCRIPT_DIR/cleanup_claude_config.py" 2>/dev/null || python "$SCRIPT_DIR/cleanup_claude_config.py" 2>/dev/null || echo "  [WARN] Could not run cleanup script"

echo ""
if [ "$mode" = "full" ]; then
    rm -rf ~/.hibro 2>/dev/null
    if [ -d ~/.hibro ]; then
        echo "  [WARN] Some files may be locked"
        echo "  Manual deletion required: ~/.hibro"
    else
        echo "  [OK] Full cleanup completed"
    fi
else
    echo "  [OK] Program removed, data preserved in: ~/.hibro"
fi

echo ""
echo "========================================"
echo "  Uninstallation Complete!"
echo "========================================"
echo ""
if [ "$mode" = "keepdata" ]; then
    echo "Note:"
    echo "  - Memory data has been preserved"
    echo "  - Reinstall hibro to continue using your memories"
    echo "  - To fully remove, manually delete: ~/.hibro"
else
    echo "Note:"
    echo "  - If database file is locked, restart and manually delete:"
    echo "    ~/.hibro"
fi
echo ""
