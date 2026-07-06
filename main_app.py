"""GA Segmentation Pro — Flet エントリポイント。"""

import os

import flet as ft

from flet_ui.main_view import build_main_view
from flet_ui.state import GAAppState
from flet_ui import theme as T


async def main(page: ft.Page) -> None:
    page.title = "GA Segmentation Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = T.BG_DARK
    page.padding = 16
    page.spacing = 0
    page.window.width = 1400
    page.window.height = 900

    state = GAAppState()
    page.data = state
    try:
        await build_main_view(page, state)
    except Exception as exc:
        import traceback
        page.controls.clear()
        page.add(
            ft.Column([
                ft.Text("起動エラー", size=20, color=T.ACCENT_RED),
                ft.Text(str(exc), color=T.TEXT),
                ft.Text(traceback.format_exc(), size=11, color=T.TEXT_MUTED, selectable=True),
            ], scroll=ft.ScrollMode.AUTO),
        )
        page.update()
        raise


if __name__ == "__main__":
    use_web = os.environ.get("FLET_USE_WEB", "1").strip().lower() not in (
        "0", "false", "no", "native", "desktop",
    )
    port = int(os.environ.get("FLET_PORT", "8551"))
    ft.app(
        target=main,
        view=ft.AppView.WEB_BROWSER if use_web else ft.AppView.FLET_APP,
        port=port,
    )
