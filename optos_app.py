"""GA Segmentation Optos — 現行アプリとは別の入口。"""

import os

import flet as ft

from flet_ui import theme as T
from flet_ui.theme import apply_page_theme
from optos_ui.app import build_optos_view
from optos_ui.session import OptosSession


async def main(page: ft.Page) -> None:
    page.title = "GA Segmentation Optos"
    page.theme_mode = ft.ThemeMode.DARK
    apply_page_theme(page)
    page.bgcolor = T.BG_DARK
    page.padding = 16
    page.window.width = 1280
    page.window.height = 900

    session = OptosSession()
    page.data = session
    try:
        await build_optos_view(page, session)
    except Exception as exc:
        import traceback
        page.controls.clear()
        page.add(
            ft.Column(
                [
                    ft.Text("起動エラー", size=20, color=T.ACCENT_RED),
                    ft.Text(str(exc), color=T.TEXT),
                    ft.Text(traceback.format_exc(), size=11, color=T.TEXT_MUTED, selectable=True),
                ],
                scroll=ft.ScrollMode.AUTO,
            )
        )
        page.update()
        raise


if __name__ == "__main__":
    use_web = os.environ.get("FLET_USE_WEB", "1").strip().lower() not in (
        "0", "false", "no", "native", "desktop",
    )
    port = int(os.environ.get("FLET_PORT", "8552"))
    ft.app(
        target=main,
        view=ft.AppView.WEB_BROWSER if use_web else ft.AppView.FLET_APP,
        port=port,
    )
