# GA Segmentation Pro

眼底自発蛍光（FAF）と Optos 超広角画像から、地図状萎縮（GA）の範囲を測るデスクトップアプリです。中心窩までの距離、GAWAIN、時点をまたいだ進行を出します。

操作の手順は [ユーザーズマニュアル](docs/ユーザーズマニュアル.md) にまとめてあります。

アプリは2つあります。

| アプリ | 起動 | 使う画像 |
|---|---|---|
| GA Segmentation Pro | `./run_flet.sh`（ポート 8551） | FAF。HRA は画像幅（mm）でスケールを付ける |
| GA Segmentation Optos | `./run_optos.sh`（ポート 8552） | Optos。乳頭–中心窩 4.76 mm から時点ごとの推定 µm/px を出し、面積換算はその眼の中央値で固定する |

どちらもウィンドウが開きます。ブラウザで開くときだけ `FLET_USE_WEB=1` を付けます。Dock の名前は Flet になることがあります。ウィンドウのタイトルがアプリ名です。

## ダウンロード

ビルド済みのアプリは [Releases](https://github.com/yasuoyanagijp-maker/GA-segmentation-pro/releases) にあります。`desktop-v*` のタグを push すると、Mac（Apple Silicon）と Windows（64 bit）を作って同じ Release に載せます。

| 環境 | ファイル |
|---|---|
| Mac（Apple Silicon） | `pro-macOS-arm64.zip`、`optos-macOS-arm64.zip` |
| Windows（64 bit） | `GA-Segmentation-Pro-windows-x64.exe`、`GA-Segmentation-Optos-windows-x64.exe` |

Mac は ZIP を展開し、「インストール.command」を右クリック →「開く」で Applications に入ります。Apple の公証は使いません。手順はマニュアルの「Mac に入れる」です。Windows は exe をダブルクリックして起動します。

## 起動

Python 3.12 と、このフォルダの仮想環境が必要です。Python 3.14 では依存パッケージが入りません。

```bash
cd ~/GA-segmentation-pro
source myenv/bin/activate
./run_flet.sh      # GA Segmentation Pro
./run_optos.sh     # GA Segmentation Optos
```

初回の環境作成はマニュアルの「ソースから起動する」にあります。

## 開発

```bash
pip install -r requirements-dev.txt
pytest -q
```

計測の本体は `engine/` です。画面は `flet_ui/`（Pro）と `optos_ui/`（Optos）です。

すでに書き出した Optos の `grader.csv` を、眼ごとの固定 µm/px で面積だけやり直すとき:

```bash
python -m ga_reading recompute-scale /path/case2/OD_reading
# 第2・最終も直して採用ルールまで再実行するなら
python -m ga_reading recompute-scale /path/case2/OD_reading --rerun-adoption
```
