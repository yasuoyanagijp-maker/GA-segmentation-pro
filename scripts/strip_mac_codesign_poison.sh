#!/usr/bin/env bash
# codesign --deep を壊す pip の *.dist-info / *.egg-info と AppleDouble を除く。
set -euo pipefail

APP_PATH="${1:?Usage: strip_mac_codesign_poison.sh App.app}"

if [[ ! -d "$APP_PATH" ]]; then
    echo "[strip] ERROR: not a directory: ${APP_PATH}" >&2
    exit 1
fi

removed=0
while IFS= read -r -d '' path; do
    rm -rf "$path"
    removed=$((removed + 1))
    echo "[strip] removed ${path#${APP_PATH}/}"
done < <(find "$APP_PATH" \( -name '*.dist-info' -o -name '*.egg-info' -o -name '._*' \) -print0 2>/dev/null)

if command -v dot_clean >/dev/null 2>&1; then
    dot_clean -m "$APP_PATH" 2>/dev/null || true
fi

left="$(find "$APP_PATH" \( -name '*.dist-info' -o -name '*.egg-info' \) 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$left" != "0" ]]; then
    echo "[strip] ERROR: ${left} metadata path(s) remain" >&2
    exit 1
fi

echo "[strip] OK (${removed} paths cleaned)"
