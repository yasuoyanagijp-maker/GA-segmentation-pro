# GA Segmentation Pro

FAF画像から地図状萎縮（GA）を自動セグメンテーションし、中心窩距離・進行予測を計算するStreamlitアプリ。

---

## フォルダ構成

```
~/GA-segmentation-pro/
├── ga_app.py              ← Streamlit メインアプリ（UI）
├── engine/                ← UI非依存のコアロジック
│   ├── io_utils.py        ← 画像読み込み
│   ├── segmentation.py    ← GAセグメンテーション（CLAHE→Otsu→形態学演算）
│   ├── calibration.py     ← 乳頭径クリックからのスケール較正（um/px）
│   ├── stats.py           ← GA面積・中心窩距離の計測
│   └── overlay.py         ← オーバーレイ画像生成
├── tests/                 ← pytestテスト（engine配下のロジックを検証）
├── conftest.py            ← pytestがengineパッケージを解決するための設定
├── requirements.txt       ← 依存パッケージ（バージョン固定）
├── requirements-dev.txt   ← 開発用（pytest等）
├── README.md              ← このファイル
└── myenv/                 ← 仮想環境（Gitには含めない）
```

> `engine/`はUI（Streamlit）に依存しないため、将来的にGAWAIN計算・複数時期レジストレーション・
> stlite/デスクトップ化などのUI層を追加しても、ロジックを再利用できるようにしています。

---

## セットアップ（初回のみ）

### 前提：Python 3.14はパッケージ非対応のためpyenvで3.12を使う

```bash
# 1. pyenvインストール（未インストールの場合）
brew install pyenv

# 2. Python 3.12をインストール
pyenv install 3.12.9

# 3. このフォルダだけ3.12を使う設定
cd ~/GA-segmentation-pro
pyenv local 3.12.9

# 4. 仮想環境を作成・有効化
python3 -m venv myenv
source myenv/bin/activate

# 5. pipを更新してからパッケージをインストール
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 起動方法

```bash
cd ~/GA-segmentation-pro
source myenv/bin/activate
streamlit run ga_app.py
```

ブラウザが自動で開く（`http://localhost:8501`）。

---

## テスト

```bash
source myenv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

`engine/`配下のセグメンテーション・較正・統計ロジックをpytestで検証しています。

---

## 操作フロー

1. サイドバーで**入力フォルダ**・**出力フォルダ**を指定
2. **「画像リストを読込」**ボタン → 対応形式: `.tif` `.tiff` `.png` `.jpg`
3. **① 乳頭計測** → 視神経乳頭の端を2点クリック（= 1800 μm として較正）
4. **② 中心窩指定** → 中心窩を1点クリック（シアン十字表示）
5. **③ 解析実行** → GAセグメンテーション自動実行
6. 右パネルで結果を確認
7. **「💾 結果を保存」** → 出力フォルダにPNG・JSONを保存

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `{stem}_overlay.png` | GAマスク（赤）＋中心窩（シアン）のオーバーレイ |
| `{stem}_mask.png` | バイナリマスク（白=GA） |
| `{stem}_result.json` | 全計測値のJSON |

---

## 計算仕様

| 項目 | 算出方法 |
|---|---|
| スケール (μm/px) | 乳頭径 px ÷ 1800 μm |
| GA Area | マスクピクセル数 × (μm/px)² → mm² 換算 |
| Distance to fovea | 中心窩座標とGAマスク全ピクセルのユークリッド距離の最小値 |
| Cross Point | Distance to fovea ÷ Progression Rate (years) |

---

## セグメンテーションパイプライン

```
load_gray → CLAHE → Otsu閾値（反転）→ Morphological Open/Close/Dilate
         → remove_small_objects → 中心部病変フィルタ（distance_transform_edt）
```

主要パラメータはサイドバーから調整可能：

| パラメータ | デフォルト | 効果 |
|---|---|---|
| CLAHE clip limit | 2.0 | 高くすると暗いGAが強調される |
| 最小GA面積 (px) | 500 | 小さいノイズ除去の閾値 |

---

## 依存パッケージ（requirements.txt）

```
streamlit==1.32.2
streamlit-image-coordinates==0.2.2
opencv-python==4.9.0.80
numpy==1.26.4
Pillow==10.3.0
scikit-image==0.22.0
scipy==1.13.0
tifffile==2024.2.12
```

> **注意**: Python 3.14では動作しない。必ず pyenv 経由の Python 3.12 を使うこと。

---

## トラブルシューティング

### `streamlit: command not found`
仮想環境が有効化されていない。
```bash
source myenv/bin/activate
```

### `Pillow` のビルドエラー
Python 3.14 が使われている。`pyenv local 3.12.9` を確認してから venv を作り直す。
```bash
python3 --version  # → 3.12.x であること
```

### `streamlit-image-coordinates` が見つからない
pip が古い。
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
