# ④ 時系列解析の動作フローと効率化調査メモ

2026-07-15 調査。「④ 時系列」ボタン押下後の処理を `flet_ui/main_view.py` /
`engine/timeseries.py` から追跡し、各段のコストを合成 FAF（1536²・5時点）で
実測した。案 1・2・3・4・5・6 は**実装済み**（下記「実装状況」参照）。

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

## 効率化の提案と実装状況

1. **トグルON時の二重構築の解消**（実装済み）
   位置合わせトグル ON で④を押した場合、register=False の中間ビルド
   （直後に捨てられていた）を作らず、直接 register=True で1回だけ構築して
   確認モード（行列再利用時は結果表示）へ進む。位置合わせ前のグラフの
   一時表示はなくなる（ステータス表示「位置合わせを計算中…」に変更）。
   ④→登録の直行パスでは series 起動ガードを登録タスク完了まで保持する
   ため、登録中のファイルドロップダウン操作による series 離脱も防げる。
   トグル OFF の④、および結果表示後の「位置合わせを実行」ボタンの挙動は
   従来どおり。

2. **GIF の遅延生成＋表示解像度化**（実装済み）
   `_series_gif_bytes()` で GIF ビュー初回表示時に表示幅
   （DISPLAY_WIDTH=520px）で生成しキャッシュ。overlay 変更時は
   `series_gif=None` で無効化。原寸が必要な PDF 出力は overlay
   （原寸のまま保持）を使うため影響なし。

3. **やり直し時の増分更新**（実装済み）
   `on_review_redo` は変更行の overlay だけ差し替え、GIF キャッシュを
   無効化、進行速度3種のみ再計算（`_recompute_series_estimates`）。

4. **GAWAIN のベクトル化**（実装済み・engine/gawain.py）
   黄斑バウンディングボックスに切り出し、`searchsorted`＋`bincount` の
   1 パスで環集計。旧実装（60環×全画面マスク）との等価性は
   `tests/test_gawain.py` の参照実装比較（乱数マスク・黄斑が画像端で
   切れるケース含む）で担保。ga_stats: 206ms → **22ms/時点**。

5. **時点間の位置合わせ並列化**（実装済み・engine/timeseries.py）
   各時点は独立に基準へ整列するため ThreadPoolExecutor（上限4並列、
   メモリ配慮）で並列実行。直列実行と同一結果（時点間依存なし・決定的）
   であることを `tests/test_series_parallel.py` で担保。
   1536²×3登録: 直列 約20秒 → **14.7秒**（4コア環境。コア数が多いほど拡大）。

6. **派生処理をワーカースレッドへ**（実装済み）
   overlay 生成を `asyncio.to_thread` で実行し、イベントループの
   フリーズを解消。GIF は案2により生成自体が軽量化（~0.1秒）。

## 残り（未実装）

7. **行列キャッシュの永続化** — `series_reg_cache` はメモリのみで
   アプリ再起動で消える。選択セット・基準・手法・日付をキーに JSON へ
   保存すれば、再起動後も再計算不要。

8. **軽微** — ④ 押下ごとの EXIF 再読込は mtime でキャッシュ可。

9. **（機能面の指摘）ROI補正の未接続** — `apply_roi_correction_to_series`
   を④の確認完了後に任意適用できるようにする場合、NL-means
   （~0.5s/フレーム）が支配的なので縮小画像での denoise や省略オプション
   の検討を推奨。stats 再計算は案4により軽くなっている。
