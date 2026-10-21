#!/usr/bin/env python3
"""今の OS 向けに、ブラウザではなくウィンドウで開くアプリを作る。

macOS では Apple Silicon (arm64) の .app、Windows では 64 bit の exe。
別の OS 向けの実行ファイルは、その OS の上でこのスクリプトを実行する。
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APPS = (
    {
        "script": "optos_app.py",
        "name": "GA Segmentation Optos",
        "bundle_id": "jp.yanagi.ga-segmentation-optos",
        "dist": "dist/optos",
        "collect": ("optos_ui", "engine", "flet_ui", "ga_reading"),
        "paths": (),
    },
    {
        "script": "main_app.py",
        "name": "GA Segmentation Pro",
        "bundle_id": "jp.yanagi.ga-segmentation-pro",
        "dist": "dist/pro",
        "collect": ("engine", "flet_ui", "ga_reading", "ga_roi_correction"),
        "paths": ("ga_roi_correction",),
    },
)


def _selected(only: str) -> tuple[dict, ...]:
    if only == "optos":
        return (APPS[0],)
    if only == "pro":
        return (APPS[1],)
    return APPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "optos", "pro"), default="all")
    args = parser.parse_args()

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine != "arm64":
            sys.exit("macOS 版は Apple Silicon (arm64) でビルドしてください。")
        data_sep = ":"
        arch = ["--target-architecture", "arm64"]
    elif system == "Windows":
        data_sep = ";"
        arch = ["--target-architecture", "x86_64"]
    else:
        sys.exit(f"この OS ではビルドできません: {system}")

    for app in _selected(args.only):
        layout = "--onedir" if system == "Darwin" else "--onefile"
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--windowed",
            layout,
            *arch,
            "--name",
            app["name"],
            "--distpath",
            app["dist"],
            "--add-data",
            f"data{data_sep}data",
            "--hidden-import",
            "skimage.filters.rank",
            "--hidden-import",
            "skimage.morphology",
            "--hidden-import",
            "skimage.segmentation",
        ]
        if system == "Darwin":
            cmd.extend(["--osx-bundle-identifier", app["bundle_id"]])
        for extra in app["paths"]:
            cmd.extend(["--paths", str(ROOT / extra)])
        for module in app["collect"]:
            cmd.extend(["--collect-submodules", module])
        cmd.append(app["script"])
        subprocess.check_call(cmd, cwd=ROOT)
        if system == "Darwin":
            bundle = ROOT / app["dist"] / f"{app['name']}.app"
            strip = ROOT / "scripts" / "strip_mac_codesign_poison.sh"
            subprocess.check_call(["bash", str(strip), str(bundle)])
            subprocess.check_call(["xattr", "-cr", str(bundle)])
            subprocess.check_call(
                ["codesign", "--force", "--deep", "--sign", "-", str(bundle)]
            )
            print(bundle)


if __name__ == "__main__":
    main()
