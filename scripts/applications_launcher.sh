#!/bin/bash
# GA Segmentation Pro — /Applications 用ランチャー
#
# ダブルクリック起動のたびに:
#   1. プロジェクトを git pull（コード自動更新）
#   2. この run.sh 自身をリポジトリ版で同期（ランチャー自動更新）
#   3. run_flet.sh でサーバー起動
#
# インストール先: /Applications/GA Segmentation Pro/run.sh
# 上書きインストール: ./scripts/install_to_applications.sh

set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/tmp/ga_segmentation_pro.log"
PROJECT_DIR="${GA_PROJECT_DIR:-}"

if [ -z "$PROJECT_DIR" ] && [ -f "$INSTALL_DIR/project_dir" ]; then
    PROJECT_DIR="$(tr -d '[:space:]' < "$INSTALL_DIR/project_dir")"
fi
PROJECT_DIR="${PROJECT_DIR:-/Users/yy/GA-segmentation-pro}"

notify() {
    osascript -e "display notification \"$1\" with title \"GA Segmentation Pro\" sound name \"Glass\"" 2>/dev/null || true
}

if [ ! -d "$PROJECT_DIR" ]; then
    notify "プロジェクトが見つかりません: $PROJECT_DIR"
    exit 1
fi

# --- 1. プロジェクトコードを更新（対話プロンプトで止まらないよう GIT_TERMINAL_PROMPT=0）---
UPDATE_MSG=""
if [ -d "$PROJECT_DIR/.git" ]; then
    BEFORE="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "")"
    if GIT_TERMINAL_PROMPT=0 git -C "$PROJECT_DIR" pull --ff-only --quiet >>"$LOG" 2>&1; then
        AFTER="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "")"
        if [ -n "$BEFORE" ] && [ "$BEFORE" != "$AFTER" ]; then
            UPDATE_MSG="コードを更新しました"
        fi
    else
        echo "[$(date)] git pull skipped or failed (dirty tree / offline?)" >>"$LOG"
    fi
fi

# --- 2. ランチャー自身をリポジトリ版で同期 ---
REPO_LAUNCHER="$PROJECT_DIR/scripts/applications_launcher.sh"
if [ -f "$REPO_LAUNCHER" ]; then
    if ! cmp -s "$REPO_LAUNCHER" "$INSTALL_DIR/run.sh" 2>/dev/null; then
        cp "$REPO_LAUNCHER" "$INSTALL_DIR/run.sh"
        chmod +x "$INSTALL_DIR/run.sh"
        UPDATE_MSG="${UPDATE_MSG:+$UPDATE_MSG / }ランチャーを更新しました"
    fi
fi

# --- 3. サーバー起動 ---
{
    echo "=== $(date) ==="
    echo "project=$PROJECT_DIR"
    [ -n "$UPDATE_MSG" ] && echo "update: $UPDATE_MSG"
} >>"$LOG"

if [ -n "$UPDATE_MSG" ]; then
    notify "$UPDATE_MSG — サーバーを起動中…"
else
    notify "サーバーを起動中… http://127.0.0.1:8551"
fi

cd "$PROJECT_DIR"
exec ./run_flet.sh >>"$LOG" 2>&1
