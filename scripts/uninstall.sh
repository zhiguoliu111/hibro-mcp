#!/usr/bin/env bash
# hibro Uninstallation Script (Linux/macOS)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  hibro Uninstallation Script"
echo "========================================"
echo ""

# Confirm uninstallation
read -p "Uninstall hibro? This will delete all data (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Uninstallation cancelled"
    exit 0
fi

echo ""
echo "[1/5] Uninstalling hibro package..."
pip3 uninstall -y hibro > /dev/null 2>&1 || echo "  [INFO] hibro not installed via pip"
echo "  [OK] hibro package uninstalled"

echo ""
echo "[2/5] Removing configuration files..."
rm -f ~/.hibro/config.yaml 2>/dev/null
echo "  [OK] Configuration files removed"

echo ""
echo "[3/5] Removing data directories..."
rm -rf ~/.hibro/data 2>/dev/null
rm -rf ~/.hibro/backups 2>/dev/null
rm -rf ~/.hibro/logs 2>/dev/null
rm -rf ~/.hibro/cache 2>/dev/null
echo "  [OK] Data directories removed"

echo ""
echo "[4/5] Removing Claude Code MCP configuration..."
python3 "$SCRIPT_DIR/cleanup_claude_config.py" 2>/dev/null || python "$SCRIPT_DIR/cleanup_claude_config.py" 2>/dev/null || echo "  [WARN] Could not run cleanup script"

echo ""
echo "[5/5] Final cleanup..."
rm -rf ~/.hibro 2>/dev/null
if [ -d ~/.hibro ]; then
    echo "  [WARN] Some files may be locked"
    echo "  Manual deletion required: ~/.hibro"
else
    echo "  [OK] Cleanup completed"
fi

echo ""
echo "========================================"
echo "  Uninstallation Complete!"
echo "========================================"
echo ""
echo "Note:"
echo "  - If database file is locked, restart and manually delete:"
echo "    ~/.hibro"
echo ""
