# archive — 未使用コードの保管庫

アプリ本体（Flet UI / engine）から参照されなくなった関数の保管場所です。
2026-07 の時系列フロー整理（未使用コード監査）で、リポジトリ全体を
「UI/エンジンからの参照」「テストのみ」「参照ゼロ」で機械的に照合し、
アプリ経路で使われていないものをここへ移動しました。

**このフォルダのコードはアプリ・テストから import されません。**
復活させる場合は元のモジュールへ戻してテストを追加してください。

## deprecated_engine_helpers.py

| 関数 | 元の場所 | 状況 |
|---|---|---|
| `guess_image_date` | `engine/timeseries.py` | 旧API（date のみ）。アプリは `guess_image_datetime` を使用 |
| `resize_to_reference` | `engine/timeseries.py` | 旧API。アプリ経路は `resize_to_reference_with_scales` |
| `generate_pdf_report` | `engine/export.py` | ファイル書き出し版。アプリは `generate_pdf_report_bytes`（bytes 版）を使用 |
| `render_gawain_figure` | `engine/gawain_figure.py` | ファイル書き出し版。アプリは `render_gawain_figure_bytes` を使用 |
| `split_optos_dual_panel` | `engine/gawain_figure.py` | Optos 両眼パネル分割。未接続のまま |
| `crop_display_margins` | `engine/gawain_figure.py` | UI 余白除去。未接続のまま |
| `estimate_fovea_from_mask` | `engine/gawain_figure.py` | マスク重心による中心窩近似。未接続のまま |

なお `ga_roi_correction/`（`registration.py`・`pipeline.correct_ga_series` など）は
スタンドアロン研究用パッケージとして README・デモ付きで意図的に維持しており、
アーカイブ対象外です（詳細は `ga_roi_correction/README.md` 冒頭の注記を参照）。
