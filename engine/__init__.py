"""GA Segmentation Pro - コアロジック（UI非依存）パッケージ。

Streamlit / stlite / 将来のデスクトップUIなど、どのUI層からも
同じロジックを呼び出せるようにするための分離レイヤー。

サブモジュール:
    io_utils     画像の読み込み
    segmentation GAセグメンテーション（CLAHE→Otsu→形態学演算→近接フィルタ）
    gawain       GAWAIN指数（60同心円環・論文由来の重み）
    life_table   令和5年簡易生命表による平均余命ルックアップ
    registration 時系列アフィン位置合わせ（stack_reg 由来）
    timeseries   複数時点の解析・GIF/グラフ用データ生成
    calibration  乳頭径クリックからのスケール較正（um/px）
    stats        GA面積・中心窩距離などの計測値算出
    overlay      オーバーレイ画像（可視化）の生成
"""
