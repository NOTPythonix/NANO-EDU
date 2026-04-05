from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: launch_from_models.py <script> [args...]")
        return 2

    script_path = Path(sys.argv[1]).resolve()
    script_args = sys.argv[2:]
    repo_root = script_path.parent.parent
    models_dir = repo_root / "models"

    if not models_dir.exists():
        print(f"models directory not found: {models_dir}")
        return 2

    cmd = [sys.executable, str(script_path), *script_args]
    env = dict(os.environ)
    env["ROBOT_TUI_SERVER_LAUNCHED_FROM_MODELS"] = "1"

    proc = subprocess.run(cmd, cwd=str(models_dir), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
