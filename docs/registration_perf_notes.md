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

## スピードアップ案と実装状況

案 1〜3 は **実装済み**（2026-07-15、下記「実装結果」参照）。

1. **前処理・マスクのキャッシュ**（実装済み）
   `_PairContext` を導入し、CLAHE＋血管強調・特徴マスク・縮小版
   （max_side=320）を 1 ペアにつき 1 回だけ計算して
   `_zoom_scale_sweep_initial` / `_alignment_mae(_downscaled)` /
   `_phase_translation_initial` / ECC / SIFT / `_pick_best_matrix` で共有。
   計算内容は従来と同一（キャッシュ有無の結果一致は
   `tests/test_series_registration_cache.py` で担保）。

2. **ECC の二段化**（実装済み）
   `_ecc_refine_forward` を coarse-to-fine 化: 縮小画像（長辺 768）で
   ECC を収束させ、その結果を初期値に原寸 ECC（criteria は従来と同一の
   反復上限 200・eps 1e-5）で仕上げる。縮小段・初期値付き原寸段が失敗した
   場合は従来どおり原寸・単位行列初期値へフォールバックするので、
   従来より悪くなる経路はない。反復数削減や ECC スキップは
   精度リスクがあるため採用しない。

3. **SIFT のキャッシュ**（実装済み）
   ref 側 `detectAndCompute` は初期値によらず不変なので 1 回だけ計算。
   さらに対応点マッチング（knnMatch＋ratio test）はモデル
   （similarity/affine）に依存しないため、初期値ごとにキャッシュし
   affine 手法での二重計算も除去。RANSAC 推定はモデルごとに従来どおり。
   `_SIFT_NFEATURES` の削減は精度検証が必要なため見送り。

## 実装結果（案1〜3適用後）

合成 FAF ペア 20 ケース（スケール 0.85〜1.15・回転 ±8°・平行移動 ±40 px・
ノイズ/コントラスト変動・病変拡大を乱数生成）で、変更前後を同一入力で
A/B 比較（`landmark` / `affine` 両手法）:

| 指標 | 変更前 | 変更後 |
|---|---|---|
| 平均エンドポイント誤差（GT行列比） | 0.086 px | 0.088 px |
| 誤差 p95 / 最大 | 0.163 / 0.182 px | 0.163 / 0.182 px |
| 0.25 px 以上悪化したケース | — | 0 件 |
| 合計時間（20ケース） | 445 秒 | 200 秒（**2.2 倍**） |

単独実行（負荷なし・1536² landmark）では 28〜30 秒 → **6.5 秒（約4.5倍）**。
ステージ別では zoom_sweep 5.5 秒 → 0.6 秒、SIFT 2.5 秒 → 1.5 秒、
ECC×3 が 20 秒前後 → 4 秒前後。

## 残りの案（未実装）

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

案 4〜6 を追加する場合も、上記 A/B ベンチ（GT 行列とのエンドポイント誤差
比較）での劣化ゼロ確認を前提とする。
