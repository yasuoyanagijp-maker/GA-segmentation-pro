# Registration（時系列位置合わせ）の速度調査メモ

2026-07-15 調査。`engine/series_registration.py` の `compute_forward_matrix` に
ステージ別タイミングログ（logger 名 `ga.reg.timing`、プレフィックス
`[reg-timing]`）を追加し、1536×1536 の合成 FAF ペアで計測した。
`engine/timeseries.py` の `build_series_from_saved` にも時点ごとの所要時間
ログを追加済み。

## 計測結果（1ペアあたり、landmark 手法）

```
[reg-timing] zoom_sweep: 5458 ms
[reg-timing] phase_translation: 152 ms
[reg-timing] sift_similarity (init 1/3): 944 ms
[reg-timing] sift_similarity (init 2/3): 764 ms
[reg-timing] sift_similarity (init 3/3): 766 ms
[reg-timing] ecc_refine (init 1/3): 4963 ms
[reg-timing] ecc_refine (init 2/3): 9452 ms
[reg-timing] ecc_refine (init 3/3): 7661 ms
[reg-timing] ecc_pyramid: 148 ms
[reg-timing] pick_best (11 candidates): 234 ms
[reg-timing] compute_forward_matrix TOTAL (method=landmark): 30575 ms
```

内訳（cProfile）:

| ステージ | 時間 | 割合 | 備考 |
|---|---|---|---|
| `_ecc_refine_forward` ×3 | 約 22 秒 | 72% | 原寸 `findTransformECC` 200 反復 ×3 初期値 |
| `_zoom_scale_sweep_initial` | 約 5.5 秒 | 18% | うち 5 秒は `_alignment_mae_downscaled` 268 回 |
| `_sift_ransac_forward` ×3 | 約 2.5 秒 | 8% | ref 側 SIFT を毎回再計算 |
| その他（phase / pyramid / pick_best） | 約 0.5 秒 | 2% | |

`_preprocess_for_registration`（CLAHE＋Gaussian×2）が 1 回の登録で **578 回**
呼ばれている（うち 536 回は zoom sweep のスコアリング内での完全な重複計算）。

## スピードアップ案（未実装）

精度に影響しない順に:

1. **前処理・マスクのキャッシュ**（低リスク・効果大）
   `_alignment_mae_downscaled` / `_alignment_mae` / `_phase_translation_initial`
   などが毎回 `_preprocess_for_registration` と `_feature_detection_mask` を
   原寸で再計算している。`_image_registration_candidates` の冒頭で ref/mov の
   前処理画像・マスク・縮小版を 1 回だけ作って引き回せば、zoom sweep が
   約 5.5 秒 → 約 0.5 秒（計測値ベース）になる見込み。

2. **ECC の反復回数・実行方法の見直し**（最大のボトルネック）
   - 3 つの初期値それぞれで原寸 200 反復の ECC を回している（計約 22 秒）。
     良い初期値（SIFT が成功した場合）があるときは ECC は微調整なので、
     `criteria` の反復数を 200 → 50 程度に減らす、または eps を緩める。
   - まず縮小画像（例: 長辺 768）で ECC を収束させ、原寸では数反復だけ
     仕上げる 2 段構成にする（`_ecc_inverse_map` は既にピラミッドで
     150 ms 程度と速い。同じ方式を `_ecc_refine_forward` に適用する）。
   - SIFT がインライア十分で成功した初期値については ECC をスキップ、
     もしくは初期値 3 通り → スコア上位 1 通りだけ ECC する。

3. **SIFT の ref 側特徴量キャッシュ**（低リスク）
   `_sift_ransac_forward` は初期値ごとに ref/mov 両方の
   `detectAndCompute` を実行するが、ref 側（前処理画像＋マスク）は
   毎回同一。1 回計算して使い回せば SIFT 時間が約半分になる。
   `_SIFT_NFEATURES=6000` も 3000 程度への削減を検討可（精度検証必要）。

4. **zoom sweep の早期打ち切り**
   2 点 landmark（中心窩＋乳頭）からの初期値でスケールがほぼ 1.0
   （例: |scale−1| < 0.03）と分かっている場合、37+21×11 通りの
   スケール×回転スイープは不要。landmark 初期値があるときは
   fine スイープを省略するか範囲を狭める。

5. **時系列全体の並列化**（`build_series_from_saved`）
   時点ごとの登録は互いに独立（全て先頭時点が基準）なので、
   `concurrent.futures.ThreadPoolExecutor` で並列化できる
   （OpenCV は GIL を解放するのでスレッドで効く）。
   N 時点の系列でほぼ N−1 倍 → コア数分の短縮。

6. **`reg_matrix` の永続化・再利用**
   `build_series_from_saved` は `prior_matrices` があれば再計算しない
   仕組みが既にある。UI 側で一度計算した `reg_matrix` をセッション保存
   （JSON 等）し、画像・固定点が変わらない限り再構築時に渡せば、
   2 回目以降の時系列表示は数十ミリ秒になる。

想定効果: 案 1〜3 だけで 1 ペア約 30 秒 → 約 5〜8 秒、案 5 併用で
体感はさらに短縮。案 2 の反復数削減のみ精度検証（既存ベンチ
`scripts/bench_faf_registration.py` での MAE 比較）を推奨。
