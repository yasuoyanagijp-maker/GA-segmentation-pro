#!/bin/bash
# GA Segmentation Optos — 現行の main_app.py とは別ポートで起動する

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

: "${FLET_USE_WEB:=0}"
export FLET_USE_WEB
: "${FLET_PORT:=8552}"
export FLET_PORT

if [ -d "./myenv" ]; then
  source ./myenv/bin/activate
elif [ -d "./.venv" ]; then
  source ./.venv/bin/activate
else
  echo "仮想環境 myenv または .venv がありません"
  exit 1
fi

if [ "$FLET_USE_WEB" = "0" ] || [ "$FLET_USE_WEB" = "false" ] || [ "$FLET_USE_WEB" = "no" ] || [ "$FLET_USE_WEB" = "native" ]; then
  echo "GA Segmentation Optos: ネイティブ http://127.0.0.1:${FLET_PORT}"
  exec flet run --port "$FLET_PORT" optos_app.py
else
  echo "GA Segmentation Optos: ブラウザ http://127.0.0.1:${FLET_PORT}"
  exec flet run --web --port "$FLET_PORT" optos_app.py
fi
