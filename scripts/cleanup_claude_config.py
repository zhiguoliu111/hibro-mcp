#!/usr/bin/env python3
"""hibro Claude Code MCP Uninstallation Script"""

import json
import sys
from pathlib import Path


def main():
    # Remove hibro from ~/.claude.json
    config_path = Path.home() / ".claude.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "mcpServers" in config and "hibro" in config["mcpServers"]:
                del config["mcpServers"]["hibro"]
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print("  [OK] hibro MCP configuration removed")
            else:
                print("  [INFO] No hibro MCP configuration found")
        except Exception as e:
            print(f"  [ERROR] Failed to remove MCP config: {e}")

    # Remove hibro permissions and hooks from ~/.claude/settings.json
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)

            # Remove hibro permissions
            if "permissions" in settings and "allow" in settings["permissions"]:
                original_count = len(settings["permissions"]["allow"])
                settings["permissions"]["allow"] = [
                    tool for tool in settings["permissions"]["allow"]
                    if "hibro" not in tool.lower()
                ]
                removed_perms = original_count - len(settings["permissions"]["allow"])
                if removed_perms > 0:
                    print(f"  [OK] Removed {removed_perms} hibro permissions")

            # Remove hibro hooks
            if "hooks" in settings and "SessionStart" in settings["hooks"]:
                original_count = len(settings["hooks"]["SessionStart"])
                settings["hooks"]["SessionStart"] = [
                    hook for hook in settings["hooks"]["SessionStart"]
                    if "hibro" not in str(hook).lower()
                ]
                removed_hooks = original_count - len(settings["hooks"]["SessionStart"])
                if removed_hooks > 0:
                    print(f"  [OK] Removed {removed_hooks} hibro hooks")
                if not settings["hooks"]["SessionStart"]:
                    del settings["hooks"]["SessionStart"]
                if not settings["hooks"]:
                    del settings["hooks"]

            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  [ERROR] Failed to clean settings: {e}")

    print("  [OK] Claude Code cleanup completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
