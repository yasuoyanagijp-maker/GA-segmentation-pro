"""GA Segmentation Pro — Flet UI theme (MNV_analysis_pro と同系)."""

from __future__ import annotations

import flet as ft

PRIMARY = "#00E5FF"
BG_DARK = "#050510"
PANEL_BG = "#151B2B"
TEXT = "#F9FAFB"
TEXT_MUTED = "#8B9BB4"
ACCENT_RED = "#EF4444"
ACCENT_CYAN = "#22D3EE"
ACCENT_AMBER = "#FBBF24"
ACCENT_GREEN = "#10B981"
ACCENT_GREEN_HOVER = "#059669"

BTN_SECONDARY_BG = "#1F2937"
BTN_OUTLINE = "#6B7280"
BTN_ON_PRIMARY = "#000000"
BTN_ON_GREEN = "#FFFFFF"

_BTN_SHAPE = ft.RoundedRectangleBorder(radius=8)


def apply_page_theme(page: ft.Page) -> None:
    """ダーク UI でボタン・入力の文字色が背景と同化しないようテーマを設定。"""
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=PRIMARY,
            on_primary=BTN_ON_PRIMARY,
            secondary=BTN_SECONDARY_BG,
            on_secondary=TEXT,
            surface=PANEL_BG,
            on_surface=TEXT,
        ),
    )
    page.dark_theme = page.theme


def btn_toolbar(label: str, on_click) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        label,
        on_click=on_click,
        style=ft.ButtonStyle(
            bgcolor=BTN_SECONDARY_BG,
            color=TEXT,
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
            elevation=0,
        ),
    )


def btn_primary(label: str, on_click) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        label,
        on_click=on_click,
        style=ft.ButtonStyle(
            bgcolor=PRIMARY,
            color=BTN_ON_PRIMARY,
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
        ),
    )


def btn_ok(label: str, on_click) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        label,
        on_click=on_click,
        style=ft.ButtonStyle(
            bgcolor={
                ft.ControlState.DEFAULT: ACCENT_GREEN,
                ft.ControlState.HOVERED: ACCENT_GREEN_HOVER,
            },
            color={
                ft.ControlState.DEFAULT: BTN_ON_GREEN,
                ft.ControlState.HOVERED: BTN_ON_GREEN,
            },
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
        ),
    )


def btn_outlined(label: str, on_click) -> ft.OutlinedButton:
    return ft.OutlinedButton(
        label,
        on_click=on_click,
        style=ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: TEXT,
                ft.ControlState.HOVERED: PRIMARY,
            },
            bgcolor={
                ft.ControlState.DEFAULT: "transparent",
                ft.ControlState.HOVERED: PANEL_BG,
            },
            side={
                ft.ControlState.DEFAULT: ft.BorderSide(1, BTN_OUTLINE),
                ft.ControlState.HOVERED: ft.BorderSide(1, PRIMARY),
            },
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
        ),
    )


def btn_draw(label: str, on_click) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        label,
        on_click=on_click,
        style=ft.ButtonStyle(
            bgcolor={
                ft.ControlState.DEFAULT: ACCENT_GREEN,
                ft.ControlState.HOVERED: ACCENT_GREEN_HOVER,
            },
            color={
                ft.ControlState.DEFAULT: BTN_ON_GREEN,
                ft.ControlState.HOVERED: BTN_ON_GREEN,
            },
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=18, vertical=10),
        ),
    )


def btn_erase(label: str, on_click) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        label,
        on_click=on_click,
        style=ft.ButtonStyle(
            bgcolor={
                ft.ControlState.DEFAULT: ACCENT_RED,
                ft.ControlState.HOVERED: "#DC2626",
            },
            color={
                ft.ControlState.DEFAULT: BTN_ON_GREEN,
                ft.ControlState.HOVERED: BTN_ON_GREEN,
            },
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=18, vertical=10),
        ),
    )


def btn_sidebar(label: str, on_click, *, expand: bool = False) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        label,
        on_click=on_click,
        expand=expand,
        style=ft.ButtonStyle(
            bgcolor=BTN_SECONDARY_BG,
            color=TEXT,
            shape=_BTN_SHAPE,
            padding=ft.padding.symmetric(horizontal=12, vertical=12),
            elevation=0,
        ),
    )
