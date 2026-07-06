#!/bin/bash
# GA Segmentation Pro - 起動スクリプト (macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 仮想環境の確認・有効化
if [ ! -d "myenv" ]; then
    echo "⚠️  仮想環境が見つかりません。セットアップを実行します..."
    echo ""

    # pyenvのpython3.12を優先、なければ/opt/homebrewのpython3
    if [ -x "$HOME/.pyenv/versions/3.12.9/bin/python3" ]; then
        PYTHON="$HOME/.pyenv/versions/3.12.9/bin/python3"
    elif command -v python3.12 &>/dev/null; then
        PYTHON="$(command -v python3.12)"
    else
        echo "❌ Python 3.12 が見つかりません。README.md のセットアップ手順を確認してください。"
        exit 1
    fi

    echo "✅ 使用するPython: $PYTHON ($($PYTHON --version))"
    $PYTHON -m venv myenv
    source myenv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo ""
    echo "✅ セットアップ完了"
else
    source myenv/bin/activate
fi

echo ""
echo "🚀 GA Segmentation Pro を起動します..."
echo "   ブラウザが開かない場合は http://localhost:8501 を開いてください"
echo "   終了するには Ctrl+C を押してください"
echo ""

streamlit run ga_app.py
