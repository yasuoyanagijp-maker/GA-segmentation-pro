"""GA Segmentation Pro - コアロジック（UI非依存）パッケージ。

Streamlit / stlite / 将来のデスクトップUIなど、どのUI層からも
同じロジックを呼び出せるようにするための分離レイヤー。

サブモジュール:
    io_utils     画像の読み込み
    segmentation GAセグメンテーション（CLAHE→Otsu→形態学演算→近接フィルタ）
    calibration  乳頭径クリックからのスケール較正（um/px）
    stats        GA面積・中心窩距離などの計測値算出
    overlay      オーバーレイ画像（可視化）の生成
"""
