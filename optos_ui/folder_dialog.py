"""ブラウザでは取れないフォルダパスを、Mac のダイアログで選ぶ。"""

from __future__ import annotations

import subprocess
import sys

_PROMPT = "Optos の画像フォルダを選んでください"
_SCRIPT = f'POSIX path of (choose folder with prompt "{_PROMPT}")'


def folder_from_osascript(returncode: int, stdout: str, stderr: str) -> str | None:
    """osascript の結果をパスにする。キャンセルは None。"""
    if returncode != 0:
        err = (stderr or "").strip()
        if "User canceled" in err or "-128" in err:
            return None
        raise RuntimeError(err or "フォルダを選べませんでした")
    path = (stdout or "").strip()
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path or None


def choose_folder() -> str | None:
    """この Mac 上でフォルダダイアログを出す。キャンセルは None。"""
    if sys.platform != "darwin":
        raise RuntimeError("この画面のフォルダ選択は macOS のダイアログを使います")
    proc = subprocess.run(
        ["osascript", "-e", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return folder_from_osascript(proc.returncode, proc.stdout, proc.stderr)
