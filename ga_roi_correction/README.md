# ga_roi_correction

FAF連続画像における GA(地図状萎縮)ROI の時系列自動補正パイプライン。
添付いただいたSOP・コード雛形を、実運用可能な形に実装したものです。
セグメンテーション自体はAI(深層学習)を使わず、**既存ROI(前フレーム or
初期抽出マスク)の境界を古典的画像処理で補正する**という設計です。

検証で、意図的に壊した3フレーム(過大×2、縮小×1)の真値マスクとの
Diceが 0.31→0.90 / 0.46→0.97 / 0.47→0.97 まで回復することを合成データで確認済みです
(`demo_synthetic.py` で再現可能)。

さらに「面積は正常なのに位置が大きくずれたROI」も、前後フレーム誘導 +
単調性制約(下記 1b)で Dice 0.09→0.94 まで回復することを確認済みです
(`demo_position_shift.py` で再現可能)。

> **GA Segmentation Pro アプリからの利用について**
> アプリ（④ 時系列の「マスクQC補正」）は、自前の位置合わせ
> （`engine/series_registration.py`）で整列済みの系列に対して
> `aligned_correct.correct_aligned_series` を
> `engine/series_roi_correction.py` 経由で呼び出します。そのため
> 本パッケージ内の `registration.py`（ECC位置合わせ）と
> `pipeline.correct_ga_series`（位置合わせ込みのフルパイプライン）は
> **アプリからは使用されません**（スタンドアロン利用・`demo_synthetic.py` /
> `demo_position_shift.py` 用として維持）。QCログ（SOP 11章相当）は
> アプリの CSV エクスポートに「ROI QC correction log」として出力されます。

## 1. 構成

```
ga_roi_correction/
  __init__.py
  preprocessing.py   SOP 6章: CLAHE・ノイズ低減・登録用/解析用画像の分離
  registration.py    SOP 7章: ECC(Euclidean→Affine)による基準フレーム登録
  refine.py          セグメンテーション境界改善 5手法 + 位置ずれ補正(下記2章)
  qc.py              SOP 8/10/11章: 異常判定・Dice・CSVログ
  pipeline.py         全体オーケストレーション + オーバーレイ画像出力
demo_synthetic.py    実データなしで動作検証できる合成テスト
demo_position_shift.py  位置ずれROIの検出・補正を検証する合成テスト
requirements.txt
```

## 2. セグメンテーション改善手法(AIセグメンテーション除く)5選

いずれも「ゼロから領域を抽出する」のではなく、**既存マスクの境界だけを
補正する**古典的画像処理手法です。`refine.py` に実装済みで、
`PipelineConfig(refine_method=...)` で切り替えられます。

1. **境界帯の局所適応閾値化 (`band_adaptive_threshold`)**
   現マスク境界の±数px帯だけを局所閾値で再判定。画像全体ではなく帯領域に
   限定することで、FAFの背景輝度ムラに引きずられず、境界のギザギザ・
   食い込みを補正できます。`base_mask` 引数で帯の中心を別マスク
   (例: 前後フレームから作った位置prior)に置き換えられます。

1b. **位置ずれ補正: 前後フレーム誘導 + 単調性制約 (`position_guided_monotonic`)**
   当該フレームのROIが「面積は正常なのに場所が大きくずれている」場合、
   面積ベースの異常検知では検出できません。そこで、
   (1) 登録済みの前後フレームROIから shape-based interpolation
   (符号付き距離変換の補間)で位置priorを作り(現マスクの位置は信頼しない)、
   (2) そのpriorを基準に band_adaptive_threshold で実際の画像輝度に
   境界をフィットさせ、
   (3) GAの非退行性の仮定に基づく単調性制約
   `prev_mask ⊆ result ⊆ dilate(next_mask, margin)`
   (前フレームより縮めない・次フレーム+マージンより広げない)を強制します。
   検出は `dice_with_neighbor_prior`(前後補間priorとのDice)が
   `AnomalyThresholds.min_dice_with_neighbor_prior`(既定0.3)未満かどうかで
   行い、QCログには `position_anomaly` として記録されます。
   両隣が存在し互いに整合するフレームのみが対象です(片側しか隣が無い
   端フレームでは、どちらがずれているか判定できないため対象外)。
   `PipelineConfig(enable_position_guided_refine=True)` が既定で有効。
   旧来どおりの挙動にしたい場合は `False` にします。

2. **Active Contour / Morphological Chan-Vese (`chan_vese`)**
   現マスクを初期輪郭として、領域内外の輝度均質性エネルギーで境界を
   局所最適化します。RegionFinder等の商用ソフトのcorrection機能や、
   SD-OCT/FAFのGA境界検出の文献でも使われる古典手法です。

3. **時系列シェイプ事前分布に基づく形態学的再構成 (`geodesic_prior_reconstruction`)**
   登録済みの前フレームROIを marker、現フレーム候補領域を軽く膨張させた
   ものを mask として reconstruction by dilation を行い、縮小異常
   (過小抽出)を復元します。単純な論理和(union)と違い、前フレームと
   連結していない無関係な領域までは巻き込みません。

4. **GraphCut / GrabCut (`grabcut`)**
   現マスクから確実前景・確実背景・不確実帯を作り、色類似度と空間的
   滑らかさを同時に考慮したエネルギー最小化で境界を補正します。
   局所閾値だけでは残るノイズ状の穴・突起の抑制に有効です。

5. **SLICスーパーピクセルによる境界スナッピング (`slic_boundary_snap`)**
   画像をSLICで過分割し、現マスクとの重なり率で各スーパーピクセルを
   病変/非病変に振り分けます。ピクセル単位のギザギザ境界を、画像の
   自然な輝度エッジに沿って滑らかにする後処理で、GrabCutの前処理として
   組み合わせられることも多い手法です。
   **注意**: 背景が滑らかに濃淡変化するFAF画像ではSLICの
   compactness/n_segments のチューニングが結果に大きく影響します
   (検証データでは他手法よりDiceが低く出ました)。単独よりGrabCutの
   下処理としての利用を推奨します。

いずれも学習データ不要・推論コスト小・パラメータの意味が説明可能という
共通の利点があり、規制当局や査読者への説明責任(なぜこの境界になったか)
を果たしやすい点も、AIセグメンテーションを外した意図に合致します。

## 3. 使い方

```python
import cv2
from ga_roi_correction import correct_ga_series, PipelineConfig

images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
masks  = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127 for p in mask_paths]  # 初期ROI

config = PipelineConfig(
    reference_index=0,          # 基準フレーム(通常は初回)
    refine_method="chan_vese",  # band_adaptive_threshold / chan_vese / grabcut / slic_boundary_snap
)

result = correct_ga_series(images, masks, dates=visit_dates,
                            config=config, output_dir="./qc_output")

result["qc_dataframe"]              # SOP11章のCSV相当(pandas.DataFrame)
result["corrected_masks_original"]  # 各フレーム元座標系での補正後マスク
```

`output_dir` を指定すると `roi_qc_log.csv` と各フレームの補正前後オーバーレイ
(赤=削除領域、緑=追加領域、黄=変更なし)が保存されます。

## 4. しきい値のチューニング

`qc.AnomalyThresholds` で異常判定の感度を調整できます(SOP8章に対応)。

```python
from ga_roi_correction.qc import AnomalyThresholds
th = AnomalyThresholds(rel_drop=0.05, rel_jump=0.20, first_ratio=1.5,
                        last_ratio=1.5, min_dice_with_prev=0.5, min_registration_cc=0.5,
                        min_dice_with_neighbor_prior=0.3)
config = PipelineConfig(thresholds=th)
```

既知の制約: 異常フレームが時系列上で隣接していると、「次フレーム比の
急増チェック」が正常フレーム側にも波及してフラグが立つことがあります
(検証データのframe1で発生)。これは意図した保守的挙動ですが、連続する
異常が多い系列では `rel_jump` を緩める・`needs_review` のみに留めて
自動補正はスキップする、といった運用が安全です。

## 5. アプリのAIチャットへの組み込み方針

Groq/DeepSeek系のfunction-calling(またはツール呼び出し)対応LLMに
本パイプラインを「ツール」として渡す想定です。例えば以下のような
関数ラッパーを1つ用意し、チャット側からは画像パス配列と設定だけを
渡してもらう形にすると実装がシンプルになります。

```python
def tool_correct_ga_series(image_paths: list[str], mask_paths: list[str],
                            dates: list[str] | None = None,
                            refine_method: str = "chan_vese") -> dict:
    """AIチャットから呼び出す用のツール関数。JSON化しやすい戻り値にする。"""
    images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127 for p in mask_paths]
    config = PipelineConfig(refine_method=refine_method)
    result = correct_ga_series(images, masks, dates=dates, config=config,
                                output_dir="/tmp/ga_qc_output")
    df = result["qc_dataframe"]
    return {
        "needs_review_frames": df[df.needs_review].index.tolist(),
        "summary": df[["index", "date", "raw_area", "corrected_area", "reason"]].to_dict("records"),
        "overlay_dir": "/tmp/ga_qc_output",
    }
```

チャットのシステムプロンプト側では、ツールの役割を「時系列ROIの異常検出と
補正候補の提示のみを行い、最終承認は人間が行う」ことを明記しておくと、
SOP10章の「QCと承認」の運用と整合します。

## 6. 依存パッケージ

`requirements.txt` を参照。標準的な `opencv-python(-headless)` /
`scikit-image` / `numpy` / `pandas` のみで、ARIAKEやgaseg系アプリの
既存Python環境にそのまま追加可能です。
