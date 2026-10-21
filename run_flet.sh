#!/bin/bash
# GA Segmentation Pro — Flet 起動（MNV_analysis_pro/run_flet.sh 準拠・API なし）

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

: "${FLET_USE_WEB:=0}"
export FLET_USE_WEB
: "${FLET_PORT:=8551}"
export FLET_PORT
# 起動時に自動読込する入力フォルダ（任意）
: "${GA_INPUT_DIR:=}"
export GA_INPUT_DIR
: "${GA_FORCE_RESTART:=0}"
export GA_FORCE_RESTART

server_healthy() {
    curl -sf --max-time 2 "http://127.0.0.1:${FLET_PORT}/" >/dev/null 2>&1 || return 1
    # HTTP 200 alone is not enough: zombie/stuck Flet can still serve static HTML
    # while the websocket never delivers the first frame (browser spinner forever).
    # Count LISTEN only — plain lsof -ti also includes browser client sockets.
    local pids
    pids=$(lsof -tiTCP:"${FLET_PORT}" -sTCP:LISTEN 2>/dev/null | sort -u)
    [ -n "$pids" ] || return 1
    local count
    count=$(echo "$pids" | wc -l | tr -d ' ')
    [ "$count" -eq 1 ] || return 1
    ps -p "$(echo "$pids" | head -1)" -o command= 2>/dev/null | grep -q "main_app.py"
}

open_browser() {
    open "http://127.0.0.1:${FLET_PORT}/" 2>/dev/null || true
}

cleanup_port() {
    local port=$1
    local pid
    local i
    # 古い Flet プロセスが残るとブラウザがグルグルのままになる
    pkill -f "[f]let run.*${port}" 2>/dev/null || true
    pkill -f "[f]let.*${port}" 2>/dev/null || true
    pkill -f "[G]A-segmentation-pro/main_app.py" 2>/dev/null || true
    sleep 1
    for i in 1 2 3 4 5 6 8; do
        local left
        # LISTEN + any leftover holders on the port
        left=$(lsof -tiTCP:"$port" 2>/dev/null | sort -u || true)
        [ -z "$left" ] && break
        while read -r pid; do
            [ -n "$pid" ] || continue
            echo "Port $port in use (PID $pid) — cleaning up"
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$left"
        sleep 1
    done
}

# 既に正常起動中なら kill せずブラウザだけ開く（古いタブのグルグル化を防ぐ）
if [ "$GA_FORCE_RESTART" != "1" ] && server_healthy; then
    echo "Flet: 既に起動中 http://127.0.0.1:${FLET_PORT} — ブラウザを開きます"
    open_browser
    exit 0
fi

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
