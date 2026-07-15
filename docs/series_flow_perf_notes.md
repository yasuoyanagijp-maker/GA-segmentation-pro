# ④ 時系列解析の動作フローと効率化調査メモ

2026-07-15 調査。「④ 時系列」ボタン押下後の処理を `flet_ui/main_view.py` /
`engine/timeseries.py` から追跡し、各段のコストを合成 FAF（1536²・5時点）で
実測した。**効率化は未実装**（提案のみ）。

## 押下後のフロー

```
④ 時系列 (on_series)
  └ _launch_series_display
      ├ ガード（確定済み・選択2枚以上）
      ├ _ensure_selected_series_dates  … EXIF/ファイル名/mtime から撮影日時を再取得（毎回ディスク読み）
      └ _series_task（別スレッド）
          ├ build_series_from_saved(register=False)
          │   ├ 撮影日順ソート（最古＝基準時点）
          │   ├ 全画像・マスクを基準解像度へリサイズ
          │   └ 各時点の stats を再計算 = ga_stats(mask, 基準scale, fovea)
          │       （面積 mm² / SQRT / 病変別 / 中心窩距離 / GAWAIN）
          ├ _apply_series_results → _rebuild_series_derived
          │   ├ 時点ごとの overlay 生成（FAF＋GAマスク＋中心窩＋黄斑円）
          │   ├ overlays_to_gif_bytes … アニメGIFを常に即時生成
          │   └ 進行速度3種を線形回帰（μm/年・mm²/年・SQRT mm/年）
          ├ _show_series_results_view → view_mode="graph"
          │   └ _refresh_graph … Flet LineChart（matplotlib不使用）＋ DataTable
          │       指標: area_mm2 / area_mm_sqrt / gawain / dist_um（ドロップダウン切替）
          │   └ refresh_results … カード（面積・拡大速度・GAWAIN・距離・余命・
          │       Cross Point）＋ series_metric_regressions の逆行警告
          └ 位置合わせトグル ON のとき
              └ _run_registration_on_series
                  ├ build_series_from_saved(register=True) … 全体を**作り直し**
                  │   ├ series_reg_cache が一致すれば行列再利用（再計算なし）
                  │   └ なければ時点ごとに compute_forward_matrix（直列）
                  ├ 確認モード（緑/マゼンタ差分・時点ごと OK / やり直し）
                  │   └ やり直し reregister_timepoint → _rebuild_series_derived
                  │       （**全時点**の overlay＋GIF を再生成）
                  └ 確認完了 → 作業/GIF/グラフ表示へ
```

## マスクの「補正」の実態

- ④ の位置合わせでは、マスクは画像と同じアフィン行列で **幾何学的に移動**
  されるだけ（`warp_mask`、二値保持のため最近傍補間）。中心窩は基準時点の
  中心窩へ厳密に一致（fovea anchoring）。
- stats は整列後マスク＋**基準画像の scale_um_px** で全時点を再計測
  （各画像固有のスケールは使わない）。
- 形状の補正（QC・refine）を行う `engine/series_roi_correction.py`
  （面積異常検出 → geodesic 再構成 / 近傍キャップ → band_adaptive_threshold /
  chan_vese / grabcut）はリポジトリに存在するが、**UI からは未接続**
  （テストのみが呼んでいる）。現状の④はマスク形状を変えない。

## 実測コスト（1536²・5時点）

| 処理 | 実測 | 備考 |
|---|---|---|
| ga_stats（1時点あたり） | 206 ms | うち GAWAIN 190 ms（60環×全画面マスク演算） |
| overlay 生成（5時点合計） | 30 ms | 軽い |
| GIF 生成（原寸5フレーム） | 1,066 ms / 6.9 MB | 表示は 520px なのに原寸で生成 |
| GIF 生成（520px 5フレーム） | 122 ms / 0.97 MB | 参考値 |
| 差分オーバーレイ（確認画面） | 2.3 ms | 問題なし |
| 画像表示 push（resize+JPEG+b64） | 3.8 ms | 問題なし |
| 位置合わせ（1時点、最適化後） | 約 6.5 s | 時点間は直列 |
| ROI補正（5時点、未接続） | 2,781 ms | NL-means denoise が支配的 |

## 効率化の提案（未実装）

1. **トグルON時の二重構築の解消**（構造改善・効果大）
   位置合わせ ON では register=False で全時点の stats・overlay・GIF を作って
   表示した直後に register=True で**全部作り直す**。中間ビルドの派生物
   （GIF・overlay・stats 再計算 ≒ 2秒＋N×0.2秒）が捨てられている。
   ON のときは直接 registration へ進むか、中間ビルドでは派生物生成を省略する。

2. **GIF の遅延生成＋表示解像度化**（低リスク・効果大）
   GIF は「GIF」ビューへ切り替えた時に初めて生成し、生成も表示幅
   （DISPLAY_WIDTH=520px）で行う: 1.07s→0.12s、ペイロード 6.9MB→0.97MB。
   PDF 出力は overlay（原寸）を使うので影響なし。確認モード中の
   `_rebuild_series_derived` からは GIF 生成を外せる（確認中は GIF 不可視）。

3. **やり直し時の増分更新**
   `on_review_redo` は 1 時点しか変わらないのに全時点の overlay＋GIF を
   再生成する。変更行だけ overlay を差し替え、GIF は確認完了時に 1 回作る。

4. **GAWAIN のベクトル化**（engine/gawain.py）
   60 環それぞれで全画面 boolean マスクを作っており 190 ms/時点。
   黄斑バウンディングボックスで切り出し、環インデックスを
   `np.digitize`＋`np.bincount` の 1 パスにすれば ~10ms 程度になる見込み。
   ④の毎回のビルド・やり直し・（将来の）ROI補正すべてに効く。

5. **時点間の位置合わせ並列化**（前回メモの案5）
   各時点の registration は独立（全て基準時点に対して）。
   ThreadPoolExecutor（OpenCV は GIL 解放）で N−1 時点を並列化。

6. **_rebuild_series_derived をワーカースレッドへ**
   現在 `_apply_series_results`（overlay＋GIF 生成 ≒ 1秒超）はイベント
   ループ上で走り、その間 UI がフリーズする。`asyncio.to_thread` へ移す。

7. **行列キャッシュの永続化**（前回メモの案6）
   `series_reg_cache` はメモリのみでアプリ再起動で消える。選択セット・
   基準・手法・日付をキーに JSON へ保存すれば、再起動後も再計算不要。

8. **軽微**: ④ 押下ごとの EXIF 再読込は mtime でキャッシュ可。
   GIF ビュー表示のたびに base64 再エンコード（5.8ms）はキャッシュ可。

9. **（機能面の指摘）ROI補正の未接続**
   `apply_roi_correction_to_series` を④の確認完了後に任意適用できるように
   する場合、NL-means（~0.5s/フレーム）が支配的なので縮小画像での denoise
   や省略オプションの検討を推奨。stats 再計算も走るため案4が効く。

推奨順: 2→3→6（UI 応答性、低リスク）、4（全経路に効く）、1→5（構造変更）。
いずれも前回同様、変更前後の同一入力比較（stats 値の一致、GIF/グラフの
見た目確認）を検証条件とする。
