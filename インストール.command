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

fail() {
    echo -e "${RED}[エラー]${NC} $1"
    read -r -p "Enterキーで閉じる..."
    exit 1
}

for APP_SRC in "${apps[@]}"; do
    APP_NAME="$(basename "$APP_SRC")"
    APP_DEST="/Applications/${APP_NAME}"
    STAGE_APP="${APP_DEST}.installing"
    BACKUP_APP="${APP_DEST}.previous"
    echo "—— ${APP_NAME} ——"

    if [[ -d "$APP_DEST" ]]; then
        echo -e "${YELLOW}[確認]${NC} 既にインストールされています。署名できたあとで置き換えます。"
    fi

    echo -e "${GREEN}[1/4]${NC} セキュリティ属性をクリア中..."
    xattr -cr "$APP_SRC"

    echo -e "${GREEN}[2/4]${NC} Applications フォルダへインストール中..."
    rm -rf "$STAGE_APP"
    if ! cp -R "$APP_SRC" "$STAGE_APP"; then
        rm -rf "$STAGE_APP"
        fail "コピーに失敗しました。今入っているアプリはそのままです。"
    fi

    echo -e "${GREEN}[3/4]${NC} このMac向けに再署名中（公証なし）..."
    xattr -cr "$STAGE_APP"
    # ZIP 展開で戻る AppleDouble と pip の metadata は codesign --deep を止める。
    while IFS= read -r -d '' path; do
        rm -rf "$path"
    done < <(find "$STAGE_APP" \( -name '*.dist-info' -o -name '*.egg-info' -o -name '._*' \) -print0 2>/dev/null)
    if command -v dot_clean >/dev/null 2>&1; then
        dot_clean -m "$STAGE_APP" 2>/dev/null || true
    fi
    if codesign --force --deep --sign - "$STAGE_APP"; then
        echo "       再署名が完了しました。"
    else
        rm -rf "$STAGE_APP"
        fail "再署名に失敗したので、今入っているアプリはそのままです。"
    fi

    if [[ -d "$APP_DEST" ]]; then
        rm -rf "$BACKUP_APP"
        if ! mv "$APP_DEST" "$BACKUP_APP" || [[ -e "$APP_DEST" ]]; then
            rm -rf "$STAGE_APP"
            fail "今のアプリを退避できなかったので、置き換えませんでした。"
        fi
    fi
    if [[ -e "$APP_DEST" ]]; then
        rm -rf "$STAGE_APP"
        fail "置き換え先が残っているので、中止しました。"
    fi
    if ! mv "$STAGE_APP" "$APP_DEST" || [[ ! -d "$APP_DEST" ]]; then
        rm -rf "$STAGE_APP"
        if [[ -d "$BACKUP_APP" && ! -e "$APP_DEST" ]]; then
            mv "$BACKUP_APP" "$APP_DEST"
        fi
        fail "置き換えに失敗しました。以前のアプリを戻しました。"
    fi
    rm -rf "$BACKUP_APP"

    echo -e "${GREEN}[4/4]${NC} アプリを起動中..."
    open "$APP_DEST"
    echo ""
done

echo "インストール完了"
echo "次回以降は Applications フォルダから起動できます。"
echo ""
read -r -p "Enterキーで閉じる..."
