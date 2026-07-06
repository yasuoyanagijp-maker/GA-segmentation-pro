"""
ga_app.py  –  GA Segmentation Pro (Streamlit)

Run:
    streamlit run ga_app.py
"""

import streamlit as st
import numpy as np
import json
import cv2
from pathlib import Path
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from engine import segmentation, calibration
from engine.io_utils import load_gray
from engine.stats import ga_stats as compute_ga_stats
from engine.overlay import make_overlay

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
[data-testid="stSidebar"] * { color: #f9fafb !important; }
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
</style>
""", unsafe_allow_html=True)

DISC_DIAMETER_UM = 1800
DISPLAY_WIDTH    = 520

# ── Session state init ────────────────────────────────────────
defaults = dict(
    mode        = None,
    disc_pts    = [],
    fovea_xy    = None,
    scale_um_px = None,
    img_gray    = None,
    mask        = None,
    stats       = None,
    file_idx    = 0,
    img_files   = [],
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_analysis():
    for k in ('disc_pts', 'fovea_xy', 'scale_um_px',
              'img_gray', 'mask', 'stats', 'mode'):
        st.session_state[k] = defaults[k]


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 📁 フォルダ設定')
    input_dir  = st.text_input('入力フォルダ', value=str(Path.home() / 'Downloads'))
    output_dir = st.text_input('出力フォルダ', value=str(Path.home() / 'Downloads' / 'ga_output'))

    if st.button('📂 画像リストを読込', use_container_width=True):
        p = Path(input_dir)
        files = sorted([f for ext in ('*.tif','*.tiff','*.png','*.jpg','*.jpeg')
                        for f in p.glob(ext)])
        st.session_state.img_files = files
        st.session_state.file_idx  = 0
        reset_analysis()

    files = st.session_state.img_files
    if files:
        names = [f.name for f in files]
        idx = st.selectbox('画像選択', range(len(names)),
                           format_func=lambda i: names[i],
                           index=st.session_state.file_idx,
                           key='file_selector')
        if idx != st.session_state.file_idx:
            st.session_state.file_idx = idx
            reset_analysis()

    st.markdown('---')
    st.markdown('### ⚙️ 解析パラメータ')
    segmentation.CLAHE_CLIP = st.slider('CLAHE clip limit', 1.0, 5.0, 2.0, 0.5)
    segmentation.MIN_GA_PX  = st.number_input('最小GA面積 (px)', 100, 5000, 500, 100)

    st.markdown('---')
    st.markdown('### 📊 患者情報')
    age  = st.number_input('年齢', 0, 120, 75)
    sex  = st.selectbox('性別', ['Male', 'Female'])
    life = st.text_input('平均余命', '11 years')
    prog = st.number_input('Progression Rate (μm/year)', 1, 500, 73)

# ── Load image ────────────────────────────────────────────────
files = st.session_state.img_files
current_file = files[st.session_state.file_idx] if files else None

if current_file and st.session_state.img_gray is None:
    try:
        st.session_state.img_gray = load_gray(current_file)
    except Exception as e:
        st.error(f'画像読み込みエラー: {e}')

img_gray = st.session_state.img_gray

# ── Header ────────────────────────────────────────────────────
st.markdown('# 👁 GA Segmentation Pro')
if current_file:
    st.markdown(f'<span class="badge">研究用</span> `{current_file.name}`',
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

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button('① 乳頭計測', use_container_width=True,
                         type='primary' if st.session_state.mode in ('disc1','disc2') else 'secondary'):
                st.session_state.mode     = 'disc1'
                st.session_state.disc_pts = []
                st.session_state.scale_um_px = None
        with b2:
            if st.button('② 中心窩指定', use_container_width=True,
                         type='primary' if st.session_state.mode == 'fovea' else 'secondary'):
                st.session_state.mode = 'fovea'
        with b3:
            if st.button('③ 解析実行', use_container_width=True, type='primary'):
                if st.session_state.scale_um_px is None:
                    st.warning('先に① 乳頭計測をしてください')
                elif st.session_state.fovea_xy is None:
                    st.warning('先に② 中心窩を指定してください')
                else:
                    with st.spinner('セグメンテーション中…'):
                        st.session_state.mask  = segmentation.segment_ga(img_gray)
                        st.session_state.stats = compute_ga_stats(
                            st.session_state.mask,
                            st.session_state.scale_um_px,
                            st.session_state.fovea_xy,
                        )

        mode  = st.session_state.mode
        scale = st.session_state.scale_um_px
        disc_done  = scale is not None
        fovea_done = st.session_state.fovea_xy is not None
        mask_done  = st.session_state.mask is not None

        def step_html(n, label, done, active):
            if done:
                bg, fg, icon = '#064e3b', '#6ee7b7', '✓'
            elif active:
                bg, fg, icon = '#1e3a5f', '#93c5fd', '●'
            else:
                bg, fg, icon = '#1f2937', '#6b7280', str(n)
            return (
                f'<div style="display:inline-flex;align-items:center;gap:6px;'
                f'background:{bg};border-radius:8px;padding:6px 12px;margin:2px;">'
                f'<span style="color:{fg};font-weight:700">{icon}</span>'
                f'<span style="color:{fg};font-size:0.85rem">{label}</span></div>'
            )

        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
            + step_html(1, '乳頭の端2点をクリック', disc_done,  mode in ('disc1','disc2'))
            + '<span style="color:#4b5563;align-self:center">→</span>'
            + step_html(2, '中心窩を1点クリック',   fovea_done, mode == 'fovea')
            + '<span style="color:#4b5563;align-self:center">→</span>'
            + step_html(3, '解析実行',              mask_done,  False)
            + '</div>',
            unsafe_allow_html=True)

        if mode == 'disc1':
            st.markdown(
                '<div style="background:#1e3a5f;border-left:3px solid #3b82f6;'
                'border-radius:6px;padding:8px 14px;font-size:0.9rem;color:#bfdbfe">'
                '🟡 &nbsp;<b>乳頭計測モード</b> &nbsp;─&nbsp; '
                '視神経乳頭の<b>端（1点目）</b>をクリック</div>',
                unsafe_allow_html=True)
        elif mode == 'disc2':
            st.markdown(
                '<div style="background:#1e3a5f;border-left:3px solid #fbbf24;'
                'border-radius:6px;padding:8px 14px;font-size:0.9rem;color:#fde68a">'
                '🟡 &nbsp;<b>乳頭計測モード</b> &nbsp;─&nbsp; '
                '反対側の<b>端（2点目）</b>をクリック'
                '（1点目 = 画像上の黄丸①）</div>',
                unsafe_allow_html=True)
        elif mode == 'fovea':
            st.markdown(
                '<div style="background:#0c4a6e;border-left:3px solid #22d3ee;'
                'border-radius:6px;padding:8px 14px;font-size:0.9rem;color:#bae6fd">'
                '🔵 &nbsp;<b>中心窩指定モード</b> &nbsp;─&nbsp; '
                '中心窩（黄斑中心）を1点クリック</div>',
                unsafe_allow_html=True)
        elif disc_done:
            st.markdown(
                f'<div style="background:#064e3b;border-left:3px solid #10b981;'
                f'border-radius:6px;padding:8px 14px;font-size:0.9rem;color:#a7f3d0">'
                f'✓ &nbsp;スケール較正完了: <b>{scale:.2f} μm/px</b>'
                f' &nbsp;（乳頭径 {DISC_DIAMETER_UM} μm 基準）</div>',
                unsafe_allow_html=True)

        nearest_xy = st.session_state.stats['nearest_xy'] \
                     if st.session_state.stats else None
        overlay_rgb = make_overlay(
            img_gray,
            st.session_state.mask if st.session_state.mask is not None
            else np.zeros_like(img_gray),
            fovea_xy   = st.session_state.fovea_xy,
            nearest_xy = nearest_xy,
            disc_pts   = st.session_state.disc_pts,
        )

        disp_h  = int(h * disp_scale)
        disp_rgb = cv2.resize(overlay_rgb, (DISPLAY_WIDTH, disp_h))
        pil_img  = Image.fromarray(disp_rgb)

        coords = streamlit_image_coordinates(pil_img, key='main_canvas')

        if coords and mode in ('disc1', 'disc2', 'fovea'):
            ox = max(0, min(w - 1, coords['x'] / disp_scale))
            oy = max(0, min(h - 1, coords['y'] / disp_scale))

            if mode == 'disc1':
                st.session_state.disc_pts = [(ox, oy)]
                st.session_state.mode = 'disc2'
                st.rerun()
            elif mode == 'disc2':
                st.session_state.disc_pts.append((ox, oy))
                p1, p2 = st.session_state.disc_pts
                st.session_state.scale_um_px = calibration.compute_scale_um_per_px(
                    p1, p2, DISC_DIAMETER_UM)
                st.session_state.mode = None
                st.rerun()
            elif mode == 'fovea':
                st.session_state.fovea_xy = (ox, oy)
                st.session_state.mode = None
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
            '<span style="color:#fbbf24">─</span> 乳頭計測',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# RIGHT: Atrophy Progression Analyzer panel
# ══════════════════════════════════════════════════
with col_panel:
    st.markdown('### Atrophy Progression Analyzer')

    stats    = st.session_state.stats
    dist_um  = stats['dist_um']  if stats else None
    area_px  = stats['area_px']  if stats else None
    area_mm2 = stats['area_mm2'] if stats else None
    cross_yr = round(dist_um / prog, 1) if (dist_um and prog > 0) else None

    def card(label, value, cls=''):
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">{label}</div>'
            f'<div class="result-value {cls}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True)

    card('Age (years)', age)
    card('Sex', sex)
    card('GA Area',
         f'{area_px:,} px  ({area_mm2} mm²)' if area_px else '—',
         'accent' if area_px else '')
    card('Current Distance to fovea (microns)',
         f'{dist_um:.0f} μm' if dist_um is not None else '—',
         'cyan' if dist_um else '')
    card('Average Progression Rate (microns/year)', f'{prog} μm/yr')
    card('平均余命', life)
    card('Cross Point',
         f'{cross_yr} years' if cross_yr is not None else '—',
         'green' if cross_yr else '')

    st.markdown('<br>', unsafe_allow_html=True)

    if st.button('💾 結果を保存', use_container_width=True, type='primary',
                 disabled=(stats is None)):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = current_file.stem if current_file else 'result'

        overlay = make_overlay(
            img_gray, st.session_state.mask,
            fovea_xy=st.session_state.fovea_xy,
            nearest_xy=nearest_xy,
            disc_pts=st.session_state.disc_pts,
        )
        cv2.imwrite(str(out / f'{stem}_overlay.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f'{stem}_mask.png'), st.session_state.mask)

        result = dict(
            file=stem, age=age, sex=sex,
            scale_um_per_px=round(st.session_state.scale_um_px, 4),
            ga_area_px=area_px, ga_area_mm2=area_mm2,
            dist_to_fovea_um=dist_um,
            progression_rate=prog, life_expectancy=life,
            cross_point_yr=cross_yr,
        )
        with open(out / f'{stem}_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        st.success(f'保存完了 → {out}')
