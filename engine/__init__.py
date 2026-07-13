"""GA Segmentation Pro - コアロジック（UI非依存）パッケージ。

Flet UI など任意のフロントエンドから同じロジックを呼び出せる分離レイヤー。

サブモジュール:
    io_utils           画像の読み込み
    segmentation       GAセグメンテーション（CLAHE→Otsu→形態学演算→近接フィルタ）
    calibration        乳頭径クリックからのスケール較正（um/px）
    crop               中心部矩形クロップ
    foveal_correction  中心窩周辺の輝度補正
    region_growing     リージョングローイング補助
    stats              GA面積・中心窩距離などの計測値算出
    gawain             GAWAIN指数（60同心円環・論文由来の重み）
    gawain_figure      GAWAIN図の生成
    life_table         令和5年簡易生命表による平均余命ルックアップ
    series_registration  複数時期の位置合わせ
    timeseries         複数時点の解析・GIF/グラフ用データ生成
    overlay            オーバーレイ画像（可視化）の生成
    export             結果のエクスポート
"""
