"""GA Segmentation Optos の時点状態。画面からはこのクラスだけを動かす。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.cases import discover_cases, ordered_case_ids
from engine.io_utils import load_gray
from engine.life_table import life_expectancy_years
from engine.optos import (
    grow_to_edge,
    intact_within_mm2,
    paint_brush,
    propose_fovea_from_disc,
    provisional_scale_from_disc,
    robust_foveal_gap_um,
    scale_disagreement_note,
    scale_from_dfd,
    segment_optos,
    shift_point,
    stats_central,
    window_box,
)
from engine.region_growing import clear_mask_in_freehand_region
from optos_ui.csv_export import mark_reversals
from engine.timeseries import (
    FOVEAL_GAP_MIN_UM,
    datetime_to_iso,
    describe_foveal_approach,
    estimate_growth_mm2_per_year,
    estimate_intact_mm2_per_year,
    estimate_growth_mm_sqrt_per_year,
    estimate_progression_rate_um_per_year,
    extract_image_datetime,
    parse_user_datetime_input,
)


@dataclass
class VisitSnap:
    filename: str
    path: str
    date_iso: str
    date_source: str
    gray: np.ndarray | None = None
    disc_xy: tuple[float, float] | None = None
    fovea_xy: tuple[float, float] | None = None
    scale_um_px: float | None = None
    mask: np.ndarray | None = None
    confirmed: bool = False
    fovea_user_placed: bool = False
    mask_stale: bool = False


@dataclass
class OptosSession:
    folder: str = ""
    cases: dict[str, list[Path]] = field(default_factory=dict)
    case_id: str = ""
    files: list[Path] = field(default_factory=list)
    index: int = 0
    snaps: dict[str, VisitSnap] = field(default_factory=dict)

    gray: np.ndarray | None = None
    disc_xy: tuple[float, float] | None = None
    fovea_xy: tuple[float, float] | None = None
    scale_um_px: float | None = None
    mask: np.ndarray | None = None
    date_iso: str = ""
    date_source: str = "missing"
    confirmed: bool = False
    fovea_user_placed: bool = False
    mask_stale: bool = False

    tool: str = "disc"  # disc | fovea | pan | grow | brush_add | brush_erase | clear
    zoom: int = 1
    pan_xy: tuple[float, float] = (0.0, 0.0)
    clear_pts: list[tuple[float, float]] = field(default_factory=list)
    undo: list[np.ndarray] = field(default_factory=list)
    redo: list[np.ndarray] = field(default_factory=list)
    series_note: str = ""
    series_summary: dict | None = None
    series_visit_table: list = field(default_factory=list)
    age: str = ""
    sex: str = "Male"

    def load_folder(self, folder: str) -> str:
        self.folder = folder
        self.cases = discover_cases(Path(folder))
        ids = ordered_case_ids(self.cases)
        if not ids:
            self.case_id = ""
            self.files = []
            self._clear_current()
            return "画像がありません"
        self.snaps = {}
        self.select_case(ids[0])
        return f"{len(ids)} 症例"

    def load_paths(self, paths: list[Path], *, label: str) -> str:
        """渡された画像だけを開く。隣の masks フォルダは読まない。"""
        files = [Path(path) for path in paths]
        self.folder = str(files[0].parent) if files else ""
        self.cases = {label: files} if files else {}
        self.snaps = {}
        if not files:
            self.case_id = ""
            self.files = []
            self._clear_current()
            return "画像がありません"
        self.select_case(label)
        return f"{len(files)} 画像"

    def reading_payload(self) -> list[dict]:
        """確定済みの計測だけ。空のマスクは出さない。"""
        self._stash()
        saved = self.case_id
        rows = []
        try:
            for key, snap in self.snaps.items():
                if not snap.confirmed or snap.gray is None or snap.mask is None:
                    continue
                if not (snap.fovea_xy and snap.disc_xy and snap.scale_um_px and np.any(snap.mask > 0)):
                    continue
                self.case_id = key.split("/", 1)[0]
                row = self._snap_row(snap)
                case_id = key.split("/", 1)[0]
                filename = snap.filename if case_id in {"", "未分類"} else f"{case_id}__{snap.filename}"
                row["path"] = snap.path
                row["shape"] = tuple(snap.gray.shape)
                row["mask"] = snap.mask
                row["filename"] = filename
                row["case_id"] = case_id
                rows.append(row)
        finally:
            self.case_id = saved
        return rows

    def snap_for(self, filename: str) -> VisitSnap | None:
        return self.snaps.get(f"{self.case_id}/{filename}")

    def select_case(self, case_id: str) -> None:
        self._stash()
        self.case_id = case_id
        self.files = list(self.cases.get(case_id) or [])
        if self.files:
            self.open_index(0, stash=False)
        else:
            self._clear_current()

    def open_index(self, index: int, *, stash: bool = True) -> str:
        if not self.files:
            return "画像がありません"
        if stash:
            self._stash()
        self.index = max(0, min(index, len(self.files) - 1))
        path = self.files[self.index]
        snap = self.snap_for(path.name)
        if snap is not None and snap.gray is not None:
            self._restore(snap)
            return f"{path.name} を復元"
        return self._open_fresh(path)

    def set_date_text(self, text: str) -> str:
        dt = parse_user_datetime_input(text)
        self.date_iso = datetime_to_iso(dt)
        self.date_source = "manual"
        return self.date_iso

    def place_disc(self, xy: tuple[float, float]) -> str:
        if self.gray is None:
            return "画像がありません"
        self.disc_xy = (float(xy[0]), float(xy[1]))
        if self.fovea_xy is None:
            provisional = provisional_scale_from_disc(self.gray, self.disc_xy)
            self.fovea_xy = propose_fovea_from_disc(
                self.disc_xy, self.gray.shape[:2], scale_um_px=provisional,
            )
            self.scale_um_px = scale_from_dfd(self.disc_xy, self.fovea_xy)
            self.tool = "fovea"
            scale = f"{self.scale_um_px:.2f}"
            if provisional:
                return (
                    f"この時点のスケール {scale} µm/px。"
                    "乳頭の大きさから中心窩を 4.76 mm 先に置きました。"
                    "中心窩をクリックすると、その位置のスケールでマスクが出ます"
                )
            return (
                f"中心窩は仮位置です。合わせるとこの時点の DFD スケール"
                f"（いま {scale} µm/px）になります"
            )
        self.scale_um_px = scale_from_dfd(self.disc_xy, self.fovea_xy)
        self.confirmed = False
        self.mask_stale = True
        return (
            f"乳頭を更新しました。この時点の DFD スケールは {self.scale_um_px:.2f} µm/px です。"
            "マスクは再セグで作り直します"
        )

    def place_fovea(self, xy: tuple[float, float]) -> str:
        if self.gray is None:
            return "画像がありません"
        first_placement = not self.fovea_user_placed
        self.fovea_xy = (float(xy[0]), float(xy[1]))
        if self.disc_xy is not None:
            self.scale_um_px = scale_from_dfd(self.disc_xy, self.fovea_xy)
        self.confirmed = False
        self.fovea_user_placed = True
        self.mask_stale = True
        self.pan_xy = (0.0, 0.0)
        if first_placement and self.disc_xy is not None and self.scale_um_px:
            self.resegment()
            text = (
                f"中心窩からマスクを作りました。この時点の DFD スケールは {self.scale_um_px:.2f} µm/px です。"
                "他の時点には使いません"
            )
        elif self.scale_um_px:
            text = (
                f"中心窩を移動しました。この時点の DFD スケールは {self.scale_um_px:.2f} µm/px です。"
                "他の時点には使いません。マスクは再セグまでそのままです"
            )
        else:
            return "中心窩を移動しました。乳頭が決まるとこの時点の DFD スケールになります"
        if self.gray is not None and self.disc_xy is not None:
            rim = provisional_scale_from_disc(self.gray, self.disc_xy)
            note = scale_disagreement_note(self.scale_um_px, rim)
            if note:
                text = f"{text}。{note}"
        return text

    def resegment(self) -> str:
        if self.gray is None or self.fovea_xy is None or self.disc_xy is None or not self.scale_um_px:
            return "乳頭と中心窩が必要です"
        self._push_undo()
        x0, y0, x1, y1 = window_box(self.gray.shape[:2], self.fovea_xy, self.scale_um_px)
        crop = self.gray[y0:y1, x0:x1]
        local_f = shift_point(self.fovea_xy, (x0, y0))
        local_d = shift_point(self.disc_xy, (x0, y0))
        assert local_f is not None and local_d is not None
        local = segment_optos(crop, local_f, local_d, self.scale_um_px)
        full = np.zeros(self.gray.shape[:2], dtype=np.uint8)
        full[y0:y1, x0:x1] = local
        self.mask = full
        self.confirmed = False
        self.mask_stale = False
        self.tool = "pan"
        return "初期マスクを作り直しました"

    def close_tool(self) -> None:
        """道具を閉じる。マスクは残す。"""
        self.tool = "pan"
        self.clear_pts = []

    def edit_at(self, xy: tuple[float, float], *, record: bool = True) -> str:
        if self.mask is None or self.gray is None or not self.scale_um_px:
            return "先に初期マスクを作ります"
        x, y = int(round(xy[0])), int(round(xy[1]))
        h, w = self.mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return "画像の外です"
        if record:
            self._push_undo()
        if self.tool == "grow":
            self.mask, added = grow_to_edge(
                self.gray, self.mask, (x, y), self.scale_um_px,
            )
            self.confirmed = False
            return f"縁まで成長 +{added} px"
        if self.tool in ("brush_add", "brush_erase"):
            self.mask = paint_brush(
                self.mask, (x, y), self.scale_um_px, erase=self.tool == "brush_erase",
            )
            self.confirmed = False
            return "境界を更新"
        if record and self.undo:
            self.undo.pop()
        return ""

    def begin_clear(self, xy: tuple[float, float]) -> None:
        self.clear_pts = [(float(xy[0]), float(xy[1]))]

    def extend_clear(self, xy: tuple[float, float]) -> None:
        self.clear_pts.append((float(xy[0]), float(xy[1])))

    def finish_clear(self) -> str:
        if self.mask is None or len(self.clear_pts) < 3:
            self.clear_pts = []
            return "範囲が短すぎます"
        self._push_undo()
        self.mask, removed = clear_mask_in_freehand_region(self.mask, self.clear_pts)
        self.clear_pts = []
        self.confirmed = False
        return f"範囲内を消去 {removed} px"

    def undo_mask(self) -> str:
        if not self.undo or self.mask is None:
            return "取り消す編集がありません"
        self.redo.append(self.mask.copy())
        self.mask = self.undo.pop()
        self.confirmed = False
        return "編集を取り消しました"

    def redo_mask(self) -> str:
        if not self.redo or self.mask is None:
            return "やり直す編集がありません"
        self.undo.append(self.mask.copy())
        self.mask = self.redo.pop()
        self.confirmed = False
        return "編集をやり直しました"

    def confirm(self) -> str:
        if self.fovea_xy is None or self.disc_xy is None or self.mask is None or not self.scale_um_px:
            return "乳頭、中心窩、マスクが揃ってから確定します"
        self.confirmed = True
        self._stash()
        done = sum(1 for s in self._case_snaps() if s.confirmed)
        return f"この時点を確定 ({done}/{len(self.files)})"

    def next_instruction(self) -> str:
        """画面に出す、いま押すべき次の一手。"""
        if not self.folder:
            return "フォルダのパスを入れて「開く」を押すか、「フォルダ選択」で画像フォルダを選んでください。"
        if self.gray is None:
            return "症例と時点を選んでください。"
        if self.disc_xy is None:
            return "「乳頭」のまま、乳頭の中心を1回クリックしてください。"
        if not self.fovea_user_placed:
            return (
                "「中心窩」で、中心窩らしい場所をクリックしてください。"
                "仮の点は乳頭から 4.76 mm 先です。"
                "クリックすると、この時点のスケールでマスクが出ます。"
            )
        if self.mask_stale:
            return "「再セグ」を押してください。マスクが、この時点のスケールで作り直されます。"
        if not self.confirmed:
            return (
                "境界を直すなら「縁まで」「塗る」「消す」「範囲消去」。"
                "そのままでよければ「この時点を確定」。"
                "道具を閉じてもマスクは残ります。"
            )
        done = sum(1 for s in self._case_snaps() if s.confirmed)
        total = len(self.files)
        if done >= 2 and done >= total:
            return "全部の時点が確定済みです。「時系列を計算」を押すと進行が出ます。年齢を入れると余命も足されます。"
        if done >= 2:
            return (
                f"確定 {done}/{total}。"
                "次の時点を開いて乳頭から始めるか、「時系列を計算」でここまでの進行を見てください。"
                "年齢を入れると余命も足されます。"
            )
        return f"確定 {done}/{total}。時点の一覧で次の画像を開き、また乳頭の中心から始めてください。"

    def live_stats(self) -> dict | None:
        if (
            self.mask is None
            or self.fovea_xy is None
            or not self.scale_um_px
            or not np.any(self.mask > 0)
        ):
            return None
        return stats_central(self.mask, self.fovea_xy, self.scale_um_px)

    def _snap_row(self, snap: VisitSnap) -> dict:
        assert snap.fovea_xy and snap.disc_xy and snap.scale_um_px and snap.mask is not None
        stats = stats_central(snap.mask, snap.fovea_xy, snap.scale_um_px)
        gap = robust_foveal_gap_um(snap.mask, snap.fovea_xy, snap.scale_um_px)
        return {
            "case_id": self.case_id,
            "date": snap.date_iso,
            "filename": snap.filename,
            "confirmed": int(snap.confirmed),
            "scale_um_per_px": round(float(snap.scale_um_px), 4),
            "disc_x": round(float(snap.disc_xy[0]), 1),
            "disc_y": round(float(snap.disc_xy[1]), 1),
            "fovea_x": round(float(snap.fovea_xy[0]), 1),
            "fovea_y": round(float(snap.fovea_xy[1]), 1),
            "area_mm2": stats.get("area_mm2"),
            "area_mm_sqrt": stats.get("area_mm_sqrt"),
            "lesion_count": stats.get("lesion_count"),
            "gawain": stats.get("gawain"),
            "dist_um_nearest": stats.get("dist_um"),
            "gap_um": None if gap is None else round(gap, 1),
            "intact_1mm_mm2": round(intact_within_mm2(snap.mask, snap.fovea_xy, snap.scale_um_px), 4),
            "outside_6mm_px": stats.get("outside_quant_px"),
        }

    def visit_rows(self) -> list[dict]:
        """マスクと乳頭・中心窩がある時点。確定前も含む。"""
        self._stash()
        snaps = [
            s for s in self._case_snaps()
            if s.mask is not None and s.fovea_xy and s.disc_xy and s.scale_um_px and np.any(s.mask > 0)
        ]
        snaps.sort(key=lambda s: s.date_iso)
        return [self._snap_row(snap) for snap in snaps]

    def run_series(self) -> str:
        self._stash()
        visits = [s for s in self._case_snaps() if s.confirmed and s.mask is not None]
        visits = [s for s in visits if s.fovea_xy and s.disc_xy and s.scale_um_px and s.gray is not None]
        if len(visits) < 2:
            self.series_summary = None
            self.series_visit_table = []
            self.series_note = "確定した時点が2つ以上必要です"
            return self.series_note
        visits.sort(key=lambda s: s.date_iso)
        rows = []
        gap_rows = []
        for s in visits:
            assert s.fovea_xy and s.scale_um_px and s.mask is not None
            stats = stats_central(s.mask, s.fovea_xy, s.scale_um_px)
            gap = robust_foveal_gap_um(s.mask, s.fovea_xy, s.scale_um_px)
            intact = intact_within_mm2(s.mask, s.fovea_xy, s.scale_um_px)
            stats["intact_1mm_mm2"] = intact
            rows.append({
                "date": s.date_iso,
                "filename": s.filename,
                "stats": stats,
            })
            gap_stats = dict(stats)
            gap_stats["dist_um"] = gap
            gap_rows.append({
                "date": s.date_iso,
                "filename": s.filename,
                "stats": gap_stats,
            })
        mm2 = estimate_growth_mm2_per_year(rows)
        mm_sqrt = estimate_growth_mm_sqrt_per_year(rows)
        gap_um = estimate_progression_rate_um_per_year(gap_rows)
        latest_gap = gap_rows[-1]["stats"].get("dist_um")
        intact_slope = estimate_intact_mm2_per_year(rows)
        life_years = self._life_years()
        life = self._life_text()
        first_intact = rows[0]["stats"]["intact_1mm_mm2"]
        last_intact = rows[-1]["stats"]["intact_1mm_mm2"]
        close = latest_gap is not None and latest_gap < FOVEAL_GAP_MIN_UM
        years_to = None
        if latest_gap is not None and gap_um and gap_um > 0 and not close:
            years_to = round(float(latest_gap) / float(gap_um), 2)
        foveal_note = describe_foveal_approach(latest_gap, None if close else gap_um)
        parts = [
            f"実測 {mm2 if mm2 is not None else '—'} mm²/年",
            f"√面積 {mm_sqrt if mm_sqrt is not None else '—'} mm/年",
            foveal_note,
            (
                f"中心 1 mm の残存 {first_intact:.2f}→{last_intact:.2f} mm²"
                + (f"（{intact_slope:+.3f} mm²/年）" if intact_slope is not None else "")
            ),
        ]
        if life:
            parts.append(life)
        else:
            parts.append("余命は年齢を入れてから")
        parts.append("各時点のスケールで計算")
        self.series_summary = {
            "case_id": self.case_id,
            "n_visits": len(visits),
            "date_first": rows[0]["date"],
            "date_last": rows[-1]["date"],
            "area_mm2_per_year": mm2,
            "area_mm_sqrt_per_year": mm_sqrt,
            "gap_um_latest": None if latest_gap is None else round(float(latest_gap), 1),
            "gap_um_per_year": None if close else gap_um,
            "years_to_fovea": years_to,
            "foveal_note": foveal_note,
            "intact_1mm_first_mm2": round(float(first_intact), 4),
            "intact_1mm_last_mm2": round(float(last_intact), 4),
            "intact_1mm_mm2_per_year": intact_slope,
            "age": (self.age or "").strip(),
            "sex": self.sex,
            "life_expectancy_years": None if life_years is None else round(life_years, 2),
        }
        self.series_visit_table = mark_reversals([self._snap_row(s) for s in visits])
        self.series_note = " · ".join(parts)
        return self.series_note

    def _life_years(self) -> float | None:
        text = (self.age or "").strip()
        if not text:
            return None
        try:
            return float(life_expectancy_years(int(text), self.sex))
        except (ValueError, KeyError):
            return None

    def _life_text(self) -> str:
        years = self._life_years()
        if years is None:
            if (self.age or "").strip():
                return "余命を算出できません"
            return ""
        return f"平均余命 {years:.1f} 年"

    def _case_snaps(self) -> list[VisitSnap]:
        out = []
        for path in self.files:
            snap = self.snap_for(path.name)
            if snap is not None:
                out.append(snap)
        return out

    def _open_fresh(self, path: Path) -> str:
        self._clear_edit_state()
        dt, source = extract_image_datetime(path)
        self.date_iso = datetime_to_iso(dt) if dt is not None else ""
        self.date_source = source
        self.gray = load_gray(path)
        self.disc_xy = None
        self.fovea_xy = None
        self.scale_um_px = None
        self.mask = np.zeros(self.gray.shape[:2], dtype=np.uint8)
        self.confirmed = False
        self.tool = "disc"
        return f"{path.name} — 乳頭中心をクリック"

    def _stash(self) -> None:
        if not self.files or self.gray is None:
            return
        path = self.files[self.index]
        self.snaps[f"{self.case_id}/{path.name}"] = VisitSnap(
            filename=path.name,
            path=str(path),
            date_iso=self.date_iso,
            date_source=self.date_source,
            gray=self.gray,
            disc_xy=self.disc_xy,
            fovea_xy=self.fovea_xy,
            scale_um_px=self.scale_um_px,
            mask=None if self.mask is None else self.mask.copy(),
            confirmed=self.confirmed,
            fovea_user_placed=self.fovea_user_placed,
            mask_stale=self.mask_stale,
        )

    def _restore(self, snap: VisitSnap) -> None:
        self._clear_edit_state()
        self.gray = snap.gray
        self.disc_xy = snap.disc_xy
        self.fovea_xy = snap.fovea_xy
        self.scale_um_px = snap.scale_um_px
        self.mask = None if snap.mask is None else snap.mask.copy()
        self.date_iso = snap.date_iso
        self.date_source = snap.date_source
        self.confirmed = snap.confirmed
        self.fovea_user_placed = snap.fovea_user_placed
        self.mask_stale = snap.mask_stale
        self.tool = "pan" if self.fovea_xy is not None else "disc"

    def _clear_current(self) -> None:
        self.gray = None
        self.disc_xy = None
        self.fovea_xy = None
        self.scale_um_px = None
        self.mask = None
        self.date_iso = ""
        self.date_source = "missing"
        self.confirmed = False
        self._clear_edit_state()

    def _clear_edit_state(self) -> None:
        self.undo = []
        self.redo = []
        self.clear_pts = []
        self.fovea_user_placed = False
        self.mask_stale = False
        self.zoom = 1
        self.pan_xy = (0.0, 0.0)
        self.series_note = ""
        self.series_summary = None
        self.series_visit_table = []

    def _push_undo(self) -> None:
        if self.mask is None:
            return
        self.undo.append(self.mask.copy())
        self.redo = []
        if len(self.undo) > 30:
            del self.undo[0]
