#!/bin/bash
# Apple Silicon 向けに、ウィンドウで開く .app を作る。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d "./myenv" ]; then
  # shellcheck disable=SC1091
  source ./myenv/bin/activate
elif [ -d "./.venv" ]; then
  # shellcheck disable=SC1091
  source ./.venv/bin/activate
fi

python -c "import PyInstaller" 2>/dev/null || pip install "pyinstaller>=6.11"
exec python scripts/build_desktop.py "$@"
