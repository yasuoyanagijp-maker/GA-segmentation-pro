# GA Segmentation Pro

FAF画像から地図状萎縮（GA）を自動セグメンテーションし、中心窩距離・進行予測を計算する Flet アプリ。

---

## フォルダ構成

```
~/GA-segmentation-pro/
├── main_app.py            ← Flet エントリポイント
├── flet_ui/               ← Flet UI（main_view, state, theme）
├── run_flet.sh            ← Flet 起動スクリプト
├── scripts/
│   ├── applications_launcher.sh  ← /Applications 用ランチャー（git pull + run_flet.sh）
│   └── install_to_applications.sh
├── engine/                ← UI非依存のコアロジック
│   ├── io_utils.py        ← 画像読み込み
│   ├── segmentation.py    ← GAセグメンテーション（CLAHE→Otsu→形態学演算）
│   ├── calibration.py     ← 乳頭径クリックからのスケール較正（um/px）
│   ├── crop.py            ← 中心部矩形クロップ
│   ├── foveal_correction.py ← 中心窩周辺の輝度補正
│   ├── region_growing.py  ← リージョングローイング補助
│   ├── stats.py           ← GA面積・病変別統計・中心窩距離・GAWAIN
│   ├── gawain.py          ← GAWAIN指数計算
│   ├── gawain_figure.py   ← GAWAIN図の生成
│   ├── life_table.py      ← 令和5年簡易生命表による平均余命ルックアップ
│   ├── series_registration.py ← 複数時期の位置合わせ
│   ├── timeseries.py      ← 複数時点の解析・GIF/グラフ用データ生成
│   ├── overlay.py         ← オーバーレイ画像生成
│   └── export.py          ← 結果のエクスポート
├── data/
│   └── gawain_weights.csv ← GAWAIN 60環の面積係数・重み（Leonlevde/GAWAIN由来）
│   └── life_table_jp_r5.csv ← 令和5年簡易生命表（死亡率.pdf由来）
├── tests/                 ← pytestテスト（engine配下のロジックを検証）
├── conftest.py            ← pytestがengineパッケージを解決するための設定
├── requirements.txt       ← 実行時依存（Flet UI + engine）
├── requirements-flet.txt  ← requirements.txt へのエイリアス
├── requirements-dev.txt   ← 開発用（pytest 等）
├── README.md              ← このファイル
└── myenv/                 ← 仮想環境（Gitには含めない）
```

> `engine/` は UI に依存しないため、Flet UI からロジックをそのまま再利用できます。

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

## ダウンロード

ビルド済みのウィンドウアプリは [Releases](https://github.com/yasuoyanagijp-maker/GA-segmentation-pro/releases) にある。Mac は Apple Silicon、Windows は 64 bit。ブラウザではなくアプリのウィンドウが開く。

## 起動方法

リポジトリ直下の起動スクリプトは **`run_flet.sh`** です。`/Applications/GA Segmentation Pro/run.sh` は git pull とランチャー同期のあと `run_flet.sh` を呼ぶ **ラッパー** であり、旧 Streamlit 用の `run.sh` とは別物です。

### コマンドライン

```bash
cd ~/GA-segmentation-pro
source myenv/bin/activate
./run_flet.sh
```

ウィンドウが開く。ブラウザで開く場合は `FLET_USE_WEB=1 ./run_flet.sh`（`http://127.0.0.1:8551`）。

### /Applications ランチャー

```bash
./scripts/install_to_applications.sh
```

インストール後、`/Applications/GA Segmentation Pro.app` をダブルクリック。起動のたびに `git pull` とランチャー同期のあと `run_flet.sh` が実行されます。

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
2. **「画像リストを読込」**ボタン → 対応形式: `.tif` `.tiff` `.png` `.jpg` `.jpeg` `.gif`
3. **⓪ 中心クロップ**（任意）→ 黄斑中心付近を対角2点クリックで矩形切り出し
4. **① 乳頭計測** → 視神経乳頭の端を2点クリック（サイドバーの乳頭径 μm で較正、デフォルト1800）
5. **② 中心窩指定** → 中心窩を1点クリック（シアン十字表示）
6. **③ 解析実行** → GAセグメンテーション自動実行
7. 右パネルで合計面積・**病変ごとの面積テーブル**を確認
8. **複数画像がある場合**: サイドバーで撮影日を設定 → **④ 時系列を位置合わせ＋解析**
9. 左パネルで **単一時点 / GIF / グラフ** を切り替え（撮影日プルダウンで時点選択）
10. **「💾 結果を保存」** → 出力フォルダにPNG・JSONを保存

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
| GAWAIN | 中心窩中心・半径3mmの60環（各0.05mm）でGA占有率に重み付け（0–100） |
| Distance to fovea | 中心窩座標とGAマスク全ピクセルのユークリッド距離の最小値 |
| Cross Point | Distance to fovea ÷ Progression Rate (years) |
| 時系列進行速度 | 複数時点の中心窩距離から線形回帰（μm/年） |
| 平均余命 | 令和5年簡易生命表（厚労省・死亡率.pdf）から年齢・性別で自動参照 |

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

## 依存パッケージ

実行時依存は `requirements.txt` に集約されています（`requirements-flet.txt` は同内容への `-r` 参照）。

```
flet==0.28.3
flet-desktop==0.28.3
opencv-python==4.9.0.80
numpy==1.26.4
Pillow==10.3.0
scikit-image==0.22.0
scipy==1.13.0
matplotlib==3.8.4
tifffile==2024.2.12
fpdf2==2.8.3
```

> **注意**: Python 3.14では動作しない。必ず pyenv 経由の Python 3.12 を使うこと。

---

## トラブルシューティング

### `flet: command not found`
仮想環境が有効化されていない、または `requirements.txt` が未インストール。
```bash
source myenv/bin/activate
pip install -r requirements.txt
```

### `Pillow` のビルドエラー
Python 3.14 が使われている。`pyenv local 3.12.9` を確認してから venv を作り直す。
```bash
python3 --version  # → 3.12.x であること
```

### ポート 8551 が使用中
`run_flet.sh` は起動時にポートを自動クリーンアップします。強制再起動する場合:
```bash
GA_FORCE_RESTART=1 ./run_flet.sh
```
