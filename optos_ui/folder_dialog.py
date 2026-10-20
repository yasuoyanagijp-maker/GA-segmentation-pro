"""ブラウザでは取れないフォルダパスを、Mac のダイアログで選ぶ。"""

from __future__ import annotations

import subprocess
import sys

_PROMPT = "Optos の画像フォルダを選んでください"


def _script(kind: str, prompt: str) -> str:
    safe = prompt.replace('"', "'")
    return f'POSIX path of (choose {kind} with prompt "{safe}")'


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


def choose_folder(prompt: str = _PROMPT) -> str | None:
    """この Mac 上でフォルダダイアログを出す。キャンセルは None。"""
    return _choose("folder", prompt)


def choose_file(prompt: str) -> str | None:
    """この Mac 上でファイルダイアログを出す。キャンセルは None。"""
    return _choose("file", prompt)


def _choose(kind: str, prompt: str) -> str | None:
    if sys.platform != "darwin":
        raise RuntimeError("この画面の選択は macOS のダイアログを使います")
    proc = subprocess.run(
        ["osascript", "-e", _script(kind, prompt)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return folder_from_osascript(proc.returncode, proc.stdout, proc.stderr)
