# GA Segmentation Pro

眼底自発蛍光（FAF）と Optos 超広角画像から、地図状萎縮（GA）の範囲を測るデスクトップアプリです。中心窩までの距離、GAWAIN、時点をまたいだ進行を出します。

操作の手順は [ユーザーズマニュアル](docs/ユーザーズマニュアル.md) にまとめてあります。

アプリは2つあります。

| アプリ | 起動 | 使う画像 |
|---|---|---|
| GA Segmentation Pro | `./run_flet.sh`（ポート 8551） | FAF。HRA は画像幅（mm）でスケールを付ける |
| GA Segmentation Optos | `./run_optos.sh`（ポート 8552） | Optos。面積換算は固定 µm/px または DICOM PixelSpacing、なければ眼の中央値 |

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
# 装置の固定スケール（全画面 4000 px なら 10.4 µm/px）
python -m ga_reading recompute-scale /path/case2/OD_reading --scale-um-px 10.4
python -m ga_reading recompute-scale /path/case2/OD_reading --from-image-width
python -m ga_reading recompute-scale /path/case2/OD_reading --dicom /path/to/export.dcm
```

起動時の環境変数でも同じ指定ができます。`OPTOS_SCALE_UM_PX=10.4`、`OPTOS_FROM_IMAGE_WIDTH=1`、`OPTOS_DICOM=/path/to.dcm`、`OPTOS_NATIVE_UM_PER_PX` / `OPTOS_NATIVE_WIDTH_PX`（既定 10.4 / 4000）。

### Optos の書き出し（推奨）

OptosAdvance の DICOM（P200DTx、PROJECTED optomap/AF）は PixelSpacing 0.0104 mm（10.4 µm/px）、4000×4000、HFOV 200° です。軸長や 2D→3D マップはありません。

読影用 PNG は **「現在の画像」** を、ズームしない・解像度を変えないで書き出してください。ビューポートを拡大した PNG は幅が時点ごとに変わり、µm/px がずれます。ずれている過去分は眼の中央値で揃えます。これから測る分は一定解像度の全画面なら 10.4 µm/px（または `10.4 × 4000 / 画像幅`）を使えます。
