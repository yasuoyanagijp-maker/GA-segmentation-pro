#!/bin/bash
# Apple Silicon 向けに、ウィンドウで開く .app を作る。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Xcode のライセンス未同意でも、Command Line Tools の lipo / codesign は使える。
if [[ -x /Library/Developer/CommandLineTools/usr/bin/lipo ]]; then
  export DEVELOPER_DIR=/Library/Developer/CommandLineTools
  export PATH="/Library/Developer/CommandLineTools/usr/bin:${PATH}"
fi

if [ -d "./myenv" ]; then
  # shellcheck disable=SC1091
  source ./myenv/bin/activate
elif [ -d "./.venv" ]; then
  # shellcheck disable=SC1091
  source ./.venv/bin/activate
fi

python -c "import PyInstaller" 2>/dev/null || pip install "pyinstaller>=6.11"
python scripts/build_desktop.py "$@"

# build_desktop.py の --only と同じく、指定したアプリだけ ZIP にする。
ONLY="all"
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--only" ]]; then
    ONLY="$arg"
  elif [[ "$arg" == --only=* ]]; then
    ONLY="${arg#--only=}"
  fi
  prev="$arg"
done
case "$ONLY" in
  all|optos|pro) ;;
  *)
    echo "unknown --only ${ONLY}" >&2
    exit 1
    ;;
esac

STAGE="${ROOT}/dist/mac-release"
rm -rf "$STAGE"
mkdir -p "$STAGE"
chmod +x "${ROOT}/インストール.command"

bundle_one() {
  local name="$1"
  local slug="$2"
  local app="${ROOT}/dist/${slug}/${name}.app"
  local folder="${STAGE}/${name}"
  if [[ ! -d "$app" ]]; then
    echo "missing ${app}" >&2
    exit 1
  fi
  mkdir -p "$folder"
  ditto --noqtn --norsrc "$app" "${folder}/${name}.app"
  cp "${ROOT}/インストール.command" "${ROOT}/Macインストール手順.txt" "$folder/"
  chmod +x "${folder}/インストール.command"
  xattr -cr "$folder"
  find "$folder" -name '._*' -delete 2>/dev/null || true
  if command -v dot_clean >/dev/null 2>&1; then
    dot_clean -m "$folder" 2>/dev/null || true
  fi
  local zip="${STAGE}/${slug}-macOS-arm64.zip"
  rm -f "$zip"
  # ditto の zip は AppleDouble (._*) を同梱し、日本語ファイル名の UTF-8 フラグも立たない。
  # MNV と同じく COPYFILE_DISABLE=1 の zip -y にする。
  (
    export COPYFILE_DISABLE=1
    cd "$STAGE"
    zip -r -y -X "$zip" "$(basename "$folder")"
  )
  echo "zip ${zip}"
}

if [[ "$ONLY" == "all" || "$ONLY" == "optos" ]]; then
  bundle_one "GA Segmentation Optos" "optos"
fi
if [[ "$ONLY" == "all" || "$ONLY" == "pro" ]]; then
  bundle_one "GA Segmentation Pro" "pro"
fi
