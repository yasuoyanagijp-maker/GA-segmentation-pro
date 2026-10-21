# GA Segmentation Pro

眼底自発蛍光（FAF）と Optos 超広角画像から、地図状萎縮（GA）の範囲を測るデスクトップアプリです。中心窩までの距離、GAWAIN、時点をまたいだ進行を出します。

操作の手順は [ユーザーズマニュアル](docs/ユーザーズマニュアル.md) にまとめてあります。

アプリは2つあります。

| アプリ | 起動 | 使う画像 |
|---|---|---|
| GA Segmentation Pro | `./run_flet.sh`（ポート 8551） | FAF。HRA は画像幅（mm）でスケールを付ける |
| GA Segmentation Optos | `./run_optos.sh`（ポート 8552） | Optos。乳頭中心から中心窩までを 4.76 mm として、時点ごとにスケールを付ける |

どちらもウィンドウが開きます。ブラウザで開くときだけ `FLET_USE_WEB=1` を付けます。Dock の名前は Flet になることがあります。ウィンドウのタイトルがアプリ名です。

## 起動

Python 3.12 と、このフォルダの仮想環境が必要です。Python 3.14 では依存パッケージが入りません。

```bash
cd ~/GA-segmentation-pro
source myenv/bin/activate
./run_flet.sh      # GA Segmentation Pro
./run_optos.sh     # GA Segmentation Optos
```

初回の環境作成はマニュアルの「ソースから起動する」にあります。

Mac の配布 ZIP には `インストール.command` が入っています。右クリック →「開く」で Applications に入ります。Apple の公証は使いません。手順はマニュアルの「Mac に入れる」です。

## 開発

```bash
pip install -r requirements-dev.txt
pytest -q
```

計測の本体は `engine/` です。画面は `flet_ui/`（Pro）と `optos_ui/`（Optos）です。
