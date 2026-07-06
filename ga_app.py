"""
ga_app.py  –  GA Segmentation Pro (Streamlit)

Run:
    streamlit run ga_app.py
"""

import streamlit as st
import numpy as np
import json
import cv2
from datetime import date, datetime
from pathlib import Path
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from engine import segmentation, calibration
from engine.io_utils import load_gray, list_images, SUPPORTED_SUFFIXES
from engine.crop import apply_crop, normalize_box
from engine.stats import ga_stats as compute_ga_stats
from engine.overlay import make_overlay
from engine.life_table import (
    LIFE_TABLE_SOURCE,
    format_life_expectancy,
    life_expectancy_years,
)
from engine.timeseries import (
    analyze_time_series,
    estimate_progression_rate_um_per_year,
    guess_image_date,
    overlays_to_gif_bytes,
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title='GA Segmentation Pro',
    page_icon='👁',
    layout='wide',
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #111827; color: #f9fafb; }
[data-testid="stSidebar"] { background: #1f2937; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e5e7eb !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #374151 !important;
}
h1,h2,h3 { color: #f9fafb; }
.result-card {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.result-label { color: #9ca3af; font-size: 0.82rem; margin-bottom: 2px; }
.result-value { color: #f9fafb; font-size: 1.25rem; font-weight: 600; }
.result-value.accent { color: #ef4444; }
.result-value.cyan   { color: #22d3ee; }
.result-value.green  { color: #10b981; }
.result-value.amber  { color: #fbbf24; }
.badge {
    display: inline-block;
    background: #374151;
    color: #9ca3af;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
}
.ga-mode-hint {
    background: #1e3a5f !important;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.9rem;
    color: #bfdbfe !important;
    margin-bottom: 4px;
}
.ga-mode-hint, .ga-mode-hint * {
    color: #bfdbfe !important;
}
.ga-mode-hint--crop { border-left: 3px solid #60a5fa; }
.ga-mode-hint--disc { border-left: 3px solid #3b82f6; }
.ga-mode-hint--disc2 { border-left: 3px solid #fbbf24; }
.ga-mode-hint--disc2, .ga-mode-hint--disc2 * { color: #fde68a !important; }
.ga-mode-hint--fovea { border-left: 3px solid #22d3ee; }
.ga-mode-hint--fovea, .ga-mode-hint--fovea * { color: #bae6fd !important; }
.ga-mode-hint--done { border-left: 3px solid #10b981; background: #064e3b !important; }
.ga-mode-hint--done, .ga-mode-hint--done * { color: #a7f3d0 !important; }
[data-testid="stAppViewContainer"] [data-testid="stAlert"] p,
[data-testid="stAppViewContainer"] [data-testid="stAlert"] {
    color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

DISPLAY_WIDTH = 520

# ── Session state init ────────────────────────────────────────
defaults = dict(
    mode        = None,
    disc_pts    = [],
    fovea_xy    = None,
    scale_um_px = None,
    img_full    = None,
    img_gray    = None,
    crop_box    = None,
    crop_pts    = [],
    mask        = None,
    stats       = None,
    file_idx    = 0,
    img_files   = [],
    series_dates = {},
    series_results = None,
    series_overlays = None,
    series_gif = None,
    series_view = 'single',
    series_time_idx = 0,
    estimated_prog = None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_series():
    st.session_state.series_results = None
    st.session_state.series_overlays = None
    st.session_state.series_gif = None
    st.session_state.estimated_prog = None


def init_series_dates(files: list[Path]):
    dates = {}
    for f in files:
        key = f.name
        if key not in st.session_state.series_dates:
            dates[key] = guess_image_date(f).isoformat()
        else:
            dates[key] = st.session_state.series_dates[key]
    st.session_state.series_dates = dates


def reset_analysis():
    for k in ('disc_pts', 'fovea_xy', 'scale_um_px', 'mask', 'stats', 'mode'):
        st.session_state[k] = defaults[k]
    reset_series()


def reset_analysis_keep_series():
    for k in ('disc_pts', 'fovea_xy', 'scale_um_px', 'mask', 'stats', 'mode'):
        st.session_state[k] = defaults[k]


def cancel_interactive_mode():
    """クロップ・乳頭・中心窩のクリック操作を中断する。"""
    mode = st.session_state.mode
    if mode in ('disc1', 'disc2'):
        st.session_state.disc_pts = []
    elif mode in ('crop1', 'crop2'):
        st.session_state.crop_pts = []
    st.session_state.mode = None


def restart_disc_measurement():
    """乳頭計測を1点目からやり直す。"""
    st.session_state.disc_pts = []
    st.session_state.mode = 'disc1'
    st.session_state.scale_um_px = None


def reset_crop():
    if st.session_state.img_full is not None:
        st.session_state.img_gray = st.session_state.img_full.copy()
    st.session_state.crop_box = None
    st.session_state.crop_pts = []
    reset_analysis()


def load_current_image(path: Path):
    gray = load_gray(path)
    st.session_state.img_full = gray
    st.session_state.img_gray = gray.copy()
    st.session_state.crop_box = None
    st.session_state.crop_pts = []
    reset_analysis_keep_series()


def run_series_analysis(files: list[Path], disc_diameter_um: float):
    scale = st.session_state.scale_um_px
    fovea = st.session_state.fovea_xy
    if scale is None or fovea is None:
        st.warning('時系列解析の前に①乳頭計測と②中心窩指定を完了してください')
        return

    images = [load_gray(f) for f in files]
    dates = [
        datetime.fromisoformat(st.session_state.series_dates[f.name]).date()
        for f in files
    ]
    progress = st.progress(0, text='時系列解析を開始…')

    def _progress(p: float, msg: str):
        progress.progress(min(1.0, max(0.0, p)), text=msg)

    results = analyze_time_series(
        images, files, dates, scale, fovea,
        register=True, progress_callback=_progress,
    )
    overlays = []
    for row in results:
        s = row['stats']
        overlays.append(make_overlay(
            row['image'], row['mask'],
            fovea_xy=fovea,
            nearest_xy=s.get('nearest_xy'),
            disc_pts=st.session_state.disc_pts,
            disc_diameter_um=disc_diameter_um,
            show_macula=True,
            scale_um_px=scale,
        ))

    st.session_state.series_results = results
    st.session_state.series_overlays = overlays
    st.session_state.series_gif = overlays_to_gif_bytes(overlays)
    st.session_state.estimated_prog = estimate_progression_rate_um_per_year(results)
    st.session_state.series_time_idx = 0
    progress.empty()
    st.success(f'時系列解析完了（{len(results)} 時点）')


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 📁 フォルダ設定')
    input_dir  = st.text_input('入力フォルダ', value=str(Path.home() / 'Downloads'))
    output_dir = st.text_input('出力フォルダ', value=str(Path.home() / 'Downloads' / 'ga_output'))

    if st.button('📂 画像リストを読込', use_container_width=True):
        p = Path(input_dir)
        files = list_images(p)
        if not files:
            st.warning(
                f'画像が見つかりません: {p}\n'
                f'対応形式: {", ".join(sorted(SUPPORTED_SUFFIXES))}'
            )
        st.session_state.img_files = files
        st.session_state.file_idx  = 0
        st.session_state.img_full = None
        st.session_state.img_gray = None
        init_series_dates(files)
        reset_crop()

    files = st.session_state.img_files
    if files:
        names = [f.name for f in files]
        idx = st.selectbox('画像選択', range(len(names)),
                           format_func=lambda i: names[i],
                           index=st.session_state.file_idx,
                           key='file_selector')
        if idx != st.session_state.file_idx:
            st.session_state.file_idx = idx
            load_current_image(files[idx])

    if files and len(files) >= 2:
        st.markdown('---')
        st.markdown('### 📅 時系列（撮影日）')
        init_series_dates(files)
        for f in files:
            current = datetime.fromisoformat(
                st.session_state.series_dates[f.name]).date()
            picked = st.date_input(
                f.name,
                value=current,
                key=f'series_date_{f.name}',
            )
            st.session_state.series_dates[f.name] = picked.isoformat()
        if st.session_state.estimated_prog is not None:
            st.caption(
                f'推定進行速度（時系列）: {st.session_state.estimated_prog:.1f} μm/年')

    st.markdown('---')
    st.markdown('### 📐 較正')
    disc_diameter_um = st.number_input(
        '乳頭径 (μm)', min_value=1000, max_value=3000, value=1800, step=50,
        help='乳頭の端2点クリック時にこの直径で μm/px を算出します')

    st.markdown('---')
    st.markdown('### ⚙️ 解析パラメータ')
    segmentation.CLAHE_CLIP = st.slider('CLAHE clip limit', 1.0, 5.0, 2.0, 0.5)
    segmentation.MIN_GA_PX  = st.number_input('最小GA面積 (px)', 100, 5000, 500, 100)

    st.markdown('---')
    st.markdown('### 📊 患者情報')
    age  = st.number_input('年齢', 0, 120, 75)
    sex  = st.selectbox('性別', ['Male', 'Female'])
    life_years = life_expectancy_years(age, sex)
    st.markdown(
        f'**平均余命:** {format_life_expectancy(age, sex)}  '
        f'<span style="font-size:0.75rem;color:#9ca3af">({LIFE_TABLE_SOURCE})</span>',
        unsafe_allow_html=True,
    )
    prog_default = int(st.session_state.estimated_prog or 73)
    prog = st.number_input(
        'Progression Rate (μm/year)', 1, 500, prog_default,
        help='時系列解析後は距離の回帰から自動推定されます',
    )

# ── Load image ────────────────────────────────────────────────
files = st.session_state.img_files
current_file = files[st.session_state.file_idx] if files else None

if current_file and st.session_state.img_full is None:
    try:
        load_current_image(current_file)
    except Exception as e:
        st.error(f'画像読み込みエラー: {e}')

img_gray = st.session_state.img_gray
panel_stats = st.session_state.stats
panel_file = current_file
panel_mask = st.session_state.mask
panel_image = img_gray

# ── Header ────────────────────────────────────────────────────
st.markdown('# 👁 GA Segmentation Pro')
if current_file:
    crop_note = ''
    if st.session_state.crop_box is not None:
        x0, y0, x1, y1 = st.session_state.crop_box
        crop_note = f'　|　クロップ: {x1 - x0}×{y1 - y0} px'
    st.markdown(
        f'<span class="badge">研究用</span> `{current_file.name}`{crop_note}',
        unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────
col_img, col_panel = st.columns([1.15, 1], gap='large')

# ══════════════════════════════════════════════════
# LEFT: Image panel
# ══════════════════════════════════════════════════
with col_img:
    if img_gray is None:
        st.info('← サイドバーからフォルダを指定して画像を読み込んでください')
    else:
        h, w = img_gray.shape
        disp_scale = DISPLAY_WIDTH / w

        c0, c1, c2, c3 = st.columns(4)
        with c0:
            if st.button('⓪ 中心クロップ', use_container_width=True,
                         type='primary' if st.session_state.mode in ('crop1', 'crop2') else 'secondary'):
                st.session_state.mode = 'crop1'
                st.session_state.crop_pts = []
        with c1:
            if st.button('① 乳頭計測', use_container_width=True,
                         type='primary' if st.session_state.mode in ('disc1', 'disc2') else 'secondary'):
                st.session_state.mode = 'disc1'
                st.session_state.disc_pts = []
                st.session_state.scale_um_px = None
        with c2:
            if st.button('② 中心窩指定', use_container_width=True,
                         type='primary' if st.session_state.mode == 'fovea' else 'secondary'):
                st.session_state.mode = 'fovea'
        with c3:
            if st.button('③ 解析実行', use_container_width=True, type='primary'):
                if st.session_state.scale_um_px is None:
                    st.warning('先に① 乳頭計測をしてください')
                elif st.session_state.fovea_xy is None:
                    st.warning('先に② 中心窩を指定してください')
                else:
                    with st.spinner('セグメンテーション中…'):
                        st.session_state.mask = segmentation.segment_ga(img_gray)
                        st.session_state.stats = compute_ga_stats(
                            st.session_state.mask,
                            st.session_state.scale_um_px,
                            st.session_state.fovea_xy,
                        )

        if files and len(files) >= 2:
            if st.button('④ 時系列を位置合わせ＋解析', use_container_width=True,
                         type='primary'):
                run_series_analysis(files, disc_diameter_um)
                st.rerun()

        if st.session_state.crop_box is not None:
            if st.button('↩ クロップを解除（原画像に戻す）', use_container_width=True):
                reset_crop()
                st.rerun()

        mode  = st.session_state.mode
        scale = st.session_state.scale_um_px
        crop_done  = st.session_state.crop_box is not None
        disc_done  = scale is not None
        fovea_done = st.session_state.fovea_xy is not None
        mask_done  = st.session_state.mask is not None

        def step_html(n, label, done, active, optional=False):
            if done:
                bg, fg, icon = '#064e3b', '#6ee7b7', '✓'
            elif active:
                bg, fg, icon = '#1e3a5f', '#93c5fd', '●'
            else:
                bg, fg, icon = '#1f2937', '#6b7280', str(n)
            opt = ' <span style="font-size:0.7rem;opacity:0.7">(任意)</span>' if optional else ''
            return (
                f'<div style="display:inline-flex;align-items:center;gap:6px;'
                f'background:{bg};border-radius:8px;padding:6px 12px;margin:2px;">'
                f'<span style="color:{fg};font-weight:700">{icon}</span>'
                f'<span style="color:{fg};font-size:0.85rem">{label}{opt}</span></div>'
            )

        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
            + step_html(0, '中心を矩形クロップ', crop_done, mode in ('crop1', 'crop2'), optional=True)
            + '<span style="color:#4b5563;align-self:center">→</span>'
            + step_html(1, '乳頭の端2点をクリック', disc_done, mode in ('disc1', 'disc2'))
            + '<span style="color:#4b5563;align-self:center">→</span>'
            + step_html(2, '中心窩を1点クリック', fovea_done, mode == 'fovea')
            + '<span style="color:#4b5563;align-self:center">→</span>'
            + step_html(3, '解析実行', mask_done, False)
            + '</div>',
            unsafe_allow_html=True)

        if mode == 'crop1':
            st.markdown(
                '<div class="ga-mode-hint ga-mode-hint--crop">'
                '🔲 &nbsp;<b>クロップモード</b> &nbsp;─&nbsp; '
                '黄斑中心付近の<b>対角1点目</b>をクリック</div>',
                unsafe_allow_html=True)
        elif mode == 'crop2':
            st.markdown(
                '<div class="ga-mode-hint ga-mode-hint--crop">'
                '🔲 &nbsp;<b>クロップモード</b> &nbsp;─&nbsp; '
                '対角の<b>2点目</b>をクリックして矩形を確定</div>',
                unsafe_allow_html=True)
        elif mode == 'disc1':
            st.markdown(
                '<div class="ga-mode-hint ga-mode-hint--disc">'
                '🟡 &nbsp;<b>乳頭計測モード</b> &nbsp;─&nbsp; '
                '視神経乳頭の<b>端（1点目）</b>をクリック</div>',
                unsafe_allow_html=True)
        elif mode == 'disc2':
            st.markdown(
                '<div class="ga-mode-hint ga-mode-hint--disc2">'
                '🟡 &nbsp;<b>乳頭計測モード</b> &nbsp;─&nbsp; '
                '反対側の<b>端（2点目）</b>をクリック'
                '（1点目 = 画像上の黄丸①）</div>',
                unsafe_allow_html=True)
        elif mode == 'fovea':
            st.markdown(
                '<div class="ga-mode-hint ga-mode-hint--fovea">'
                '🔵 &nbsp;<b>中心窩指定モード</b> &nbsp;─&nbsp; '
                '中心窩（黄斑中心）を1点クリック</div>',
                unsafe_allow_html=True)
        elif disc_done:
            st.markdown(
                f'<div class="ga-mode-hint ga-mode-hint--done">'
                f'✓ &nbsp;スケール較正完了: <b>{scale:.2f} μm/px</b>'
                f' &nbsp;（乳頭径 {disc_diameter_um:.0f} μm 基準）</div>',
                unsafe_allow_html=True)

        if mode in ('crop1', 'crop2', 'disc1', 'disc2', 'fovea'):
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button('✕ 操作をキャンセル', use_container_width=True):
                    cancel_interactive_mode()
                    st.rerun()
            with bc2:
                if mode == 'disc2' and st.button('↩ 1点目からやり直す', use_container_width=True):
                    restart_disc_measurement()
                    st.rerun()
                elif mode == 'crop2' and st.button('↩ 1点目からやり直す', use_container_width=True):
                    st.session_state.crop_pts = []
                    st.session_state.mode = 'crop1'
                    st.rerun()

        series_results = st.session_state.series_results
        series_view = None
        if series_results:
            series_view = st.radio(
                '表示モード',
                ['単一時点', 'GIF', 'グラフ'],
                horizontal=True,
                key='series_view_mode',
            )

        if series_results and series_view == 'GIF':
            st.image(st.session_state.series_gif, use_container_width=True)
            st.caption('位置合わせ後の GA オーバーレイを時系列で表示')
        elif series_results and series_view == 'グラフ':
            import pandas as pd
            chart_df = pd.DataFrame({
                'date': [r['date'] for r in series_results],
                'GA Area (mm²)': [r['stats']['area_mm2'] for r in series_results],
                'GAWAIN': [r['stats']['gawain'] for r in series_results],
                'Dist to fovea (μm)': [
                    r['stats']['dist_um'] if r['stats']['dist_um'] is not None else np.nan
                    for r in series_results
                ],
            })
            st.line_chart(chart_df.set_index('date'))
            st.dataframe(chart_df, use_container_width=True, hide_index=True)
        elif series_results and series_view == '単一時点':
            labels = [
                f"{r['date']} — {r['filename']}" for r in series_results
            ]
            st.session_state.series_time_idx = st.selectbox(
                '撮影日（カレンダー）',
                range(len(series_results)),
                format_func=lambda i: labels[i],
                index=st.session_state.series_time_idx,
                key='series_time_selector',
            )
            row = series_results[st.session_state.series_time_idx]
            panel_stats = row['stats']
            panel_file = Path(row['path'])
            panel_mask = row['mask']
            panel_image = row['image']
            s = row['stats']
            overlay_rgb = st.session_state.series_overlays[st.session_state.series_time_idx]
            h, w = overlay_rgb.shape[:2]
            disp_h = int(h * (DISPLAY_WIDTH / w))
            st.image(cv2.resize(overlay_rgb, (DISPLAY_WIDTH, disp_h)), use_container_width=True)
            st.caption(f"{row['date']} | GAWAIN {s['gawain']:.1f} | GA {s['area_mm2']} mm²")
        else:
            stats = st.session_state.stats
            nearest_xy = stats['nearest_xy'] if stats else None
            overlay_rgb = make_overlay(
                img_gray,
                st.session_state.mask if st.session_state.mask is not None
                else np.zeros_like(img_gray),
                fovea_xy=st.session_state.fovea_xy,
                nearest_xy=nearest_xy,
                disc_pts=st.session_state.disc_pts,
                disc_diameter_um=disc_diameter_um,
                crop_pts=st.session_state.crop_pts if mode in ('crop1', 'crop2') else None,
                show_macula=mask_done,
                scale_um_px=scale,
            )

            disp_h = int(h * disp_scale)
            disp_rgb = cv2.resize(overlay_rgb, (DISPLAY_WIDTH, disp_h))
            pil_img = Image.fromarray(disp_rgb)

            coords = streamlit_image_coordinates(pil_img, key='main_canvas')

            if coords and mode in ('crop1', 'crop2', 'disc1', 'disc2', 'fovea'):
                ox = max(0, min(w - 1, coords['x'] / disp_scale))
                oy = max(0, min(h - 1, coords['y'] / disp_scale))

                if mode == 'crop1':
                    st.session_state.crop_pts = [(ox, oy)]
                    st.session_state.mode = 'crop2'
                    st.rerun()
                elif mode == 'crop2':
                    st.session_state.crop_pts.append((ox, oy))
                    p1, p2 = st.session_state.crop_pts
                    try:
                        box = normalize_box(p1[0], p1[1], p2[0], p2[1], w, h)
                        if st.session_state.crop_box is None:
                            st.session_state.crop_box = box
                        else:
                            ox0, oy0, _, _ = st.session_state.crop_box
                            bx0, by0, bx1, by1 = box
                            st.session_state.crop_box = (
                                ox0 + bx0, oy0 + by0, ox0 + bx1, oy0 + by1,
                            )
                        st.session_state.img_gray = apply_crop(img_gray, box)
                        st.session_state.crop_pts = []
                        st.session_state.mode = None
                        reset_analysis()
                        st.rerun()
                    except ValueError as e:
                        st.warning(str(e))
                        st.session_state.crop_pts = []
                        st.session_state.mode = 'crop1'
                elif mode == 'disc1':
                    st.session_state.disc_pts = [(ox, oy)]
                    st.session_state.mode = 'disc2'
                    st.rerun()
                elif mode == 'disc2':
                    st.session_state.disc_pts.append((ox, oy))
                    p1, p2 = st.session_state.disc_pts
                    st.session_state.scale_um_px = calibration.compute_scale_um_per_px(
                        p1, p2, disc_diameter_um)
                    st.session_state.mode = None
                    reset_series()
                    st.rerun()
                elif mode == 'fovea':
                    st.session_state.fovea_xy = (ox, oy)
                    st.session_state.mode = None
                    reset_series()
                    if st.session_state.mask is not None:
                        st.session_state.stats = compute_ga_stats(
                            st.session_state.mask,
                            st.session_state.scale_um_px,
                            st.session_state.fovea_xy,
                        )
                    st.rerun()

        st.markdown(
            '<span style="color:#ef4444">■</span> GA (hypo-AF)　'
            '<span style="color:#22d3ee">✛</span> 中心窩　'
            '<span style="color:#fbbf24">─</span> 乳頭計測　'
            '<span style="color:#60a5fa">▭</span> クロップ　'
            '<span style="color:#a78bfa">○</span> GAWAIN黄斑域 (6mm)',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# RIGHT: Atrophy Progression Analyzer panel
# ══════════════════════════════════════════════════
with col_panel:
    st.markdown('### Atrophy Progression Analyzer')

    stats = panel_stats
    dist_um  = stats['dist_um']  if stats else None
    area_px  = stats['area_px']  if stats else None
    area_mm2 = stats['area_mm2'] if stats else None
    lesions  = stats['lesions']  if stats else []
    gawain   = stats['gawain']   if stats else None
    life     = format_life_expectancy(age, sex)
    cross_yr = round(dist_um / prog, 1) if (dist_um and prog > 0) else None
    ga_before_death = (
        cross_yr is not None and cross_yr < life_years
    )

    nearest_xy = stats['nearest_xy'] if stats else None

    def card(label, value, cls=''):
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">{label}</div>'
            f'<div class="result-value {cls}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True)

    card('Age (years)', age)
    card('Sex', sex)
    card('Disc diameter (μm)', f'{disc_diameter_um:.0f}')
    card('GA Area (total)',
         f'{area_px:,} px  ({area_mm2} mm²)' if area_px else '—',
         'accent' if area_px else '')
    card('GA lesions detected',
         str(len(lesions)) if stats else '—')
    card('GAWAIN',
         f'{gawain:.1f}' if gawain is not None else '—',
         'amber' if gawain is not None else '')
    card('Current Distance to fovea (microns)',
         f'{dist_um:.0f} μm' if dist_um is not None else '—',
         'cyan' if dist_um else '')
    card('Average Progression Rate (microns/year)', f'{prog} μm/yr')
    card('平均余命（令和5年生命表）', life)
    card('Cross Point',
         f'{cross_yr} years' if cross_yr is not None else '—',
         'green' if cross_yr else '')

    if cross_yr is not None:
        if ga_before_death:
            st.markdown(
                '<div style="background:#422006;border:1px solid #92400e;border-radius:8px;'
                'padding:10px 14px;font-size:0.88rem;color:#fde68a">'
                f'⚠ Cross Point（{cross_yr}年）&lt; 平均余命（{life_years:.1f}年）'
                ' — GAが中心窩に到達する可能性が余命内にあります</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#064e3b;border:1px solid #065f46;border-radius:8px;'
                'padding:10px 14px;font-size:0.88rem;color:#a7f3d0">'
                f'Cross Point（{cross_yr}年）≥ 平均余命（{life_years:.1f}年）'
                ' — 余命内での中心窩到達リスクは低い可能性があります</div>',
                unsafe_allow_html=True)

    if lesions:
        st.markdown('#### GA病変ごとの面積')
        table_rows = [{
            'ID': l['id'],
            'Area (px)': l['area_px'],
            'Area (mm²)': l['area_mm2'],
            'Dist to fovea (μm)': l['dist_um'] if l['dist_um'] is not None else '—',
        } for l in lesions]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    if st.session_state.series_results:
        st.markdown('#### 時系列サマリー')
        series_rows = [{
            'Date': r['date'],
            'GA (mm²)': r['stats']['area_mm2'],
            'GAWAIN': r['stats']['gawain'],
            'Dist (μm)': r['stats']['dist_um'],
        } for r in st.session_state.series_results]
        st.dataframe(series_rows, use_container_width=True, hide_index=True)

    st.markdown('<br>', unsafe_allow_html=True)

    if st.button('💾 結果を保存', use_container_width=True, type='primary',
                 disabled=(stats is None)):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = panel_file.stem if panel_file else 'result'

        overlay = make_overlay(
            panel_image, panel_mask,
            fovea_xy=st.session_state.fovea_xy,
            nearest_xy=nearest_xy,
            disc_pts=st.session_state.disc_pts,
            disc_diameter_um=disc_diameter_um,
            show_macula=True,
            scale_um_px=st.session_state.scale_um_px,
        )
        cv2.imwrite(str(out / f'{stem}_overlay.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f'{stem}_mask.png'), panel_mask)

        result = dict(
            file=stem, age=age, sex=sex,
            disc_diameter_um=disc_diameter_um,
            crop_box=list(st.session_state.crop_box) if st.session_state.crop_box else None,
            scale_um_per_px=round(st.session_state.scale_um_px, 4),
            ga_area_px=area_px, ga_area_mm2=area_mm2,
            lesion_count=len(lesions),
            lesions=lesions,
            gawain=gawain,
            dist_to_fovea_um=dist_um,
            life_expectancy_years=round(life_years, 2),
            life_expectancy_source=LIFE_TABLE_SOURCE,
            progression_rate=prog,
            cross_point_yr=cross_yr,
            ga_reaches_fovea_within_life_expectancy=ga_before_death if cross_yr is not None else None,
        )
        with open(out / f'{stem}_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        st.success(f'保存完了 → {out}')
