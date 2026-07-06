#!/bin/bash
# GA Segmentation Pro — Flet 起動（MNV_analysis_pro/run_flet.sh 準拠・API なし）

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

: "${FLET_USE_WEB:=1}"
export FLET_USE_WEB
: "${FLET_PORT:=8551}"
export FLET_PORT
# 起動時に自動読込する入力フォルダ（任意）
: "${GA_INPUT_DIR:=}"
export GA_INPUT_DIR

cleanup_port() {
    local port=$1
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "Port $port in use (PID $pid) — cleaning up"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
    # 古い Flet プロセスが残るとブラウザがグルグルのままになる
    pkill -f "[f]let run.*${port}" 2>/dev/null || true
    pkill -f "[G]A-segmentation-pro/main_app.py" 2>/dev/null || true
    sleep 1
}
cleanup_port "$FLET_PORT"

if [ -d "./myenv" ]; then
  source ./myenv/bin/activate
elif [ -d "./.venv" ]; then
  source ./.venv/bin/activate
else
  echo "仮想環境 myenv または .venv がありません"
  exit 1
fi

pip install -q -r requirements-flet.txt 2>/dev/null || pip install -q -r requirements-flet.txt

if [ "$FLET_USE_WEB" = "0" ] || [ "$FLET_USE_WEB" = "false" ] || [ "$FLET_USE_WEB" = "no" ] || [ "$FLET_USE_WEB" = "native" ]; then
  echo "Flet: ネイティブウィンドウ（OSフォルダダイアログ可） http://127.0.0.1:${FLET_PORT}"
  flet run --port "$FLET_PORT" main_app.py
else
  echo "Flet: ブラウザ http://127.0.0.1:${FLET_PORT}"
  flet run --web --port "$FLET_PORT" main_app.py
fi
