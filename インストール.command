#!/bin/bash
# 公証（Apple notary）は使いません。この Mac 上で:
#   1. xattr -cr … Gatekeeper の隔離属性を外す
#   2. codesign --sign - … アドホック再署名（Hardened Runtime なし）
# 右クリック →「開く」で実行してください。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo ""
echo "GA Segmentation — インストーラー"
echo ""
echo -e "${YELLOW}[重要]${NC} 初回は「インストール.command」を"
echo "       右クリック →「開く」で実行してください。"
echo ""

shopt -s nullglob
apps=("${SCRIPT_DIR}"/*.app)
shopt -u nullglob

if [[ ${#apps[@]} -eq 0 ]]; then
    echo -e "${RED}[エラー]${NC} 同じフォルダに .app がありません。"
    read -r -p "Enterキーで閉じる..."
    exit 1
fi

for APP_SRC in "${apps[@]}"; do
    APP_NAME="$(basename "$APP_SRC")"
    APP_DEST="/Applications/${APP_NAME}"
    echo "—— ${APP_NAME} ——"

    if [[ -d "$APP_DEST" ]]; then
        echo -e "${YELLOW}[確認]${NC} 既にインストールされています。上書きします。"
        rm -rf "$APP_DEST"
    fi

    echo -e "${GREEN}[1/4]${NC} セキュリティ属性をクリア中..."
    xattr -cr "$APP_SRC"

    echo -e "${GREEN}[2/4]${NC} Applications フォルダへインストール中..."
    cp -R "$APP_SRC" /Applications/

    if [[ ! -d "$APP_DEST" ]]; then
        echo -e "${RED}[エラー]${NC} インストールに失敗しました。"
        read -r -p "Enterキーで閉じる..."
        exit 1
    fi

    echo -e "${GREEN}[3/4]${NC} このMac向けに再署名中（公証なし）..."
    xattr -cr "$APP_DEST"
    if codesign --force --deep --sign - "$APP_DEST"; then
        echo "       再署名が完了しました。"
    else
        echo -e "${YELLOW}[警告]${NC} 再署名に失敗しました。ターミナルで次を実行してください:"
        echo "         xattr -cr \"${APP_DEST}\""
        echo "         codesign --force --deep --sign - \"${APP_DEST}\""
    fi

    echo -e "${GREEN}[4/4]${NC} アプリを起動中..."
    open "$APP_DEST"
    echo ""
done

echo "インストール完了"
echo "次回以降は Applications フォルダから起動できます。"
echo ""
read -r -p "Enterキーで閉じる..."
