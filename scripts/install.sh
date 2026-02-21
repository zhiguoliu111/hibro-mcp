#!/usr/bin/env bash
# hibro Installation Script (Linux/macOS)

set -e

# Get script directory and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "  hibro Intelligent Memory System"
echo "  Installation Script"
echo "========================================"
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "  [ERROR] Python not found"
    echo "  Please install Python 3.10+ first"
    exit 1
fi

$PYTHON --version
echo "  [OK] Python found"

# Install dependencies
echo ""
echo "[2/7] Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip -q
$PYTHON -m pip install -r requirements.txt -q || {
    echo "  [WARN] Trying with pip..."
    $PYTHON -m pip install -r requirements.txt -q
}

# Install package
echo ""
echo "[3/7] Installing hibro package..."
$PYTHON -m pip install -e . -q
echo "  [OK] hibro package installed"

# Install MCP SDK
echo ""
echo "[4/7] Installing MCP SDK..."
$PYTHON -m pip install mcp -q
echo "  [OK] MCP SDK installed"

# Create directories
echo ""
echo "[5/7] Creating data directories..."
mkdir -p ~/.hibro/data
mkdir -p ~/.hibro/backups
mkdir -p ~/.hibro/logs
mkdir -p ~/.hibro/cache
echo "  [OK] Data directories created"

# Create default configuration
echo ""
echo "[6/7] Creating configuration file..."
if [ ! -f ~/.hibro/config.yaml ]; then
    cat > ~/.hibro/config.yaml << 'EOF'
# hibro Configuration File
data_directory: ~/.hibro/data

memory:
  auto_learn: true
  importance_threshold: 0.7
  max_memories: 100000

forgetting:
  time_decay_rate: 0.1
  min_importance: 0.3
  cleanup_interval_days: 7

ide:
  type: auto
  auto_inject: true
  context_limit_kb: 200
  monitor_conversations: true
  injection_strategy: smart

ide_integration:
  auto_inject: true
  context_limit_kb: 200
  monitor_conversations: true

security:
  encryption_enabled: true
  auto_cleanup_days: 365
  sensitive_data_filter: true

storage:
  database_path: ~/.hibro/memories.db
  max_size_gb: 10
  backup_enabled: true
  backup_interval_hours: 24

performance:
  cache_size_mb: 100
  enable_compression: true

logging:
  level: INFO
  file: ~/.hibro/logs/hibro.log
EOF
    echo "  [OK] Configuration file created"
else
    echo "  [INFO] Configuration file already exists"
fi

# Configure Claude Code MCP
echo ""
echo "[7/7] Configuring Claude Code MCP integration..."

# Get actual Python path
PYTHON_PATH=$(which $PYTHON)

# Configure ~/.claude.json
CLAUDE_CONFIG="$HOME/.claude.json"
if [ -f "$CLAUDE_CONFIG" ]; then
    # Use Python to update JSON
    $PYTHON -c "
import json
from pathlib import Path

config_path = Path.home() / '.claude.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['hibro'] = {
    'command': '$PYTHON_PATH',
    'args': ['-m', 'hibro.mcp.server'],
    'env': {}
}

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('  [OK] Claude Code MCP configured')
"
else
    # Create new config
    cat > "$CLAUDE_CONFIG" << EOF
{
  "mcpServers": {
    "hibro": {
      "command": "$PYTHON_PATH",
      "args": ["-m", "hibro.mcp.server"],
      "env": {}
    }
  }
}
EOF
    echo "  [OK] Claude Code MCP configured"
fi

# Run configuration script for permissions and hooks
$PYTHON "$SCRIPT_DIR/setup_claude_config.py"

# Verify installation
echo ""
echo "========================================"
echo "  Verification"
echo "========================================"
$PYTHON -c "import hibro; print('  [OK] hibro installation successful!')"

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Usage:"
echo "  hibro tools:          mcp__hibro__get_quick_context"
echo "  Store memory:         mcp__hibro__remember"
echo "  Search memories:      mcp__hibro__search_memories"
echo ""
echo "Configuration files:"
echo "  ~/.hibro/config.yaml       hibro configuration"
echo "  ~/.claude.json             Claude Code MCP config"
echo "  ~/.claude/settings.json    Claude Code settings"
echo ""
echo "Uninstall:"
echo "  scripts/uninstall.sh"
echo ""
