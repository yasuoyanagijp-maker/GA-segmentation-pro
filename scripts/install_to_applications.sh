#!/bin/bash
# /Applications に GA Segmentation Pro ランチャー（run.sh + .app）を配置・更新する。
#
# Usage:
#   ./scripts/install_to_applications.sh
#   GA_PROJECT_DIR=/path/to/repo ./scripts/install_to_applications.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="GA Segmentation Pro"
INSTALL_ROOT="/Applications/${APP_NAME}"
APP_BUNDLE="/Applications/${APP_NAME}.app"
PROJECT_DIR="${GA_PROJECT_DIR:-$REPO}"

mkdir -p "$INSTALL_ROOT"

# run.sh（自動更新ランチャー）
cp "$REPO/scripts/applications_launcher.sh" "$INSTALL_ROOT/run.sh"
chmod +x "$INSTALL_ROOT/run.sh"

# プロジェクトパス（環境変数 GA_PROJECT_DIR より優先度低）
printf '%s\n' "$PROJECT_DIR" > "$INSTALL_ROOT/project_dir"

# AppleScript .app — run.sh を完全デタッチで起動
cat > /tmp/ga_launcher.applescript <<EOF
-- ${APP_NAME} サーバー起動（/Applications/${APP_NAME}/run.sh）
set installDir to "/Applications/${APP_NAME}"
do shell script "cd " & quoted form of installDir & " && (./run.sh >/dev/null 2>&1 &) &>/dev/null"
display notification "アプリのウィンドウを開いています" with title "${APP_NAME}" sound name "Glass"
EOF

rm -rf "$APP_BUNDLE"
osacompile -o "$APP_BUNDLE" /tmp/ga_launcher.applescript

# アイコン（リポジトリにあれば適用）
ICON_SRC="$REPO/scripts/app_icon.icns"
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$APP_BUNDLE/Contents/Resources/applet.icns"
    /usr/bin/plutil -replace CFBundleIconFile -string "applet" "$APP_BUNDLE/Contents/Info.plist" 2>/dev/null || true
fi

/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_BUNDLE" 2>/dev/null || true

echo "Installed:"
echo "  $INSTALL_ROOT/run.sh"
echo "  $INSTALL_ROOT/project_dir → $PROJECT_DIR"
echo "  $APP_BUNDLE"
echo ""
echo "ダブルクリックで起動。起動のたびに git pull + ランチャー同期が走ります。"
