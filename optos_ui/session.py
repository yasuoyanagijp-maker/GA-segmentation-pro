"""GA Segmentation Optos の時点状態。画面からはこのクラスだけを動かす。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.cases import discover_cases, folder_exclusions, ordered_case_ids
from engine.eye_scale import fixed_eye_scale, is_scale_outlier
from engine.io_utils import load_gray
from engine.life_table import life_expectancy_years
from engine.optos import (
    grow_to_edge,
    growth_image,
    intact_within_mm2,
    align_prior_mask,
    translate_mask,
    apply_reading_exclusions,
    baseline_appearance,
    paint_brush,
    fovea_from_reference,
    measurement_scale,
    propose_fovea_from_disc,
    series_scale_from_dfd,
    stable_disc_radius,
    provisional_scale_from_disc,
    register_fovea_from_prior,
    robust_foveal_gap_um,
    scale_disagreement_note,
    scale_from_dfd,
    segment_optos,
    shift_point,
    stats_central,
    window_box,
)
from engine.region_growing import clear_mask_in_freehand_region
from ga_reading.bundle import exclusion_log_text
from optos_ui.csv_export import area_change_warnings, gap_reopen_warnings, mark_reversals
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

logger = logging.getLogger("ga.optos.reading")


def _scale_message(source: str, scale: float | None, *, placed: str) -> str:
    if scale is None:
        return "乳頭の縁が読めず、中心窩も未確定なのでスケールがありません"
    value = f"{scale:.2f}"
    if source == "disc":
        basis = f"乳頭径 1.8 mm の {value} µm/px"
        stable = "中心窩を少し動かしても、この換算は変わりません。"
    elif source == "series":
        basis = f"1枚目との乳頭–中心窩の画素距離の比で計算した {value} µm/px"
        stable = "中心窩や乳頭を動かすと、この比で計算し直します。"
    elif source == "registration":
        basis = f"1枚目と血管を合わせた倍率で計算した {value} µm/px"
        stable = "乳頭や中心窩を少し動かしても、この換算は変わりません。"
    elif source == "dfd-held":
        basis = f"{value} µm/px（クリックの小さなずれでは据え置き）"
        stable = "この画像の距離が基準です。小さなずれでは換算を動かしません。"
    else:
        basis = f"この画像の乳頭–中心窩 4.76 mm の {value} µm/px"
        stable = "後の画像は、この距離を乳頭の大きさで合わせます。"
    if placed == "disc" and source == "series":
        return (
            f"この時点のスケールは {basis}。"
            "1枚目の乳頭から中心窩までの距離を、この画像の乳頭の大きさで点にしました。"
            f"中心窩をクリックするとマスクが出ます。{stable}"
        )
    if placed == "disc" and source == "disc":
        return (
            f"この時点のスケールは {basis}。"
            "乳頭の大きさから中心窩を 4.76 mm 先に置きました。"
            f"中心窩をクリックするとマスクが出ます。{stable}"
        )
    if placed == "disc":
        return f"この時点のスケールは {basis}。中心窩は仮位置です。クリックするとマスクが出ます。{stable}"
    if placed == "disc-kept":
        return (
            f"乳頭を更新しました。中心窩の位置はそのままです。"
            f"この時点のスケールは {basis} です。"
            f"前の輪郭は、新しい距離に合わせて拡大縮小します。{stable}"
        )
    if placed == "disc-moved":
        return f"乳頭を更新しました。この時点のスケールは {basis} です。マスクは再セグで作り直します。{stable}"
    if placed == "fovea-first":
        return f"中心窩からマスクを作りました。この時点のスケールは {basis} です。{stable}"
    return (
        f"中心窩を移動しました。この時点のスケールは {basis} です。"
        f"{stable}マスクは再セグまでそのままです"
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
    disc_radius_px: float | None = None
    mask: np.ndarray | None = None
    confirmed: bool = False
    fovea_user_placed: bool = False
    mask_stale: bool = False
    contrast_gain: float | None = None
    hypo_threshold: float | None = None
    needs_review: str = ""
    area_note: str = ""
    # 前の時点の輪郭を初期マスクにしている。時点を移しても、別の時点へ持ち越さない。
    mask_from_prior: bool = False
    # 血管で写した暫定中心窩。戻ったあとのクリックが、この位置と比べられる。
    fovea_proposal_xy: tuple[float, float] | None = None
    fovea_proposal_source: str = ""
    # 血管位置合わせでスケールを決めたときの、1枚目に対する倍率。None なら DFD 比。
    registration_zoom: float | None = None


# 血管位置合わせで写した暫定中心窩から、クリックがこれ以上離れたら確認を促す。
_FOVEA_PROPOSAL_WARN_UM = 300.0


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
    disc_radius_px: float | None = None
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
    min_lesion_mm2: float = 0.05
    mask_from_prior: bool = False
    baseline_gain: float | None = None
    # 絶対輝度ではなく、基準時点の ROI に対する低蛍光の割合（0–100）。
    baseline_threshold: float | None = None
    alignment_gray: np.ndarray | None = None
    needs_review: str = ""
    area_note: str = ""
    # 2枚目以降の暫定中心窩。source は "registration"（血管位置合わせ）か "vector"。
    fovea_proposal_xy: tuple[float, float] | None = None
    fovea_proposal_source: str = ""
    # 血管位置合わせでスケールを決めたときの、1枚目に対する倍率。None なら DFD 比。
    registration_zoom: float | None = None
    folder_exclusions: list[tuple[str, str]] = field(default_factory=list)
    load_errors: dict[str, str] = field(default_factory=dict)

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
        self.load_errors = {}
        self.folder_exclusions = folder_exclusions(Path(folder))
        self._clear_current()
        self._log_exclusions(self.folder_exclusions)
        self.select_case(ids[0])
        names = [path.name for path in self.files]
        logger.info("読影対象 %d 枚（日付の重複や間隔では外していない）: %s", len(names), ", ".join(names))
        return self._open_status(f"{len(ids)} 症例")

    def load_paths(self, paths: list[Path], *, label: str) -> str:
        """渡された画像だけを開く。隣の masks フォルダは読まない。"""
        files = [Path(path) for path in paths]
        self.folder = str(files[0].parent) if files else ""
        self.cases = {label: files} if files else {}
        self.snaps = {}
        self.load_errors = {}
        self.folder_exclusions = []
        self._clear_current()
        if not files:
            self.case_id = ""
            self.files = []
            self._clear_current()
            return "画像がありません"
        self.select_case(label)
        names = [path.name for path in self.files]
        logger.info("読影対象 %d 枚（日付の重複や間隔では外していない）: %s", len(names), ", ".join(names))
        return self._open_status(f"{len(files)} 画像")

    def reading_payload(self) -> list[dict]:
        """確定済みの計測だけ。空のマスクは出さない。"""
        self._stash()
        saved = self.case_id
        rows = []
        try:
            for key, snap in self.snaps.items():
                case_id = key.split("/", 1)[0]
                # 役割を切り替えた直前の画像が、前の役割名で残ることがある。
                if case_id in {"第2", "最終"} and case_id != saved:
                    continue
                if not snap.confirmed or snap.gray is None or snap.mask is None:
                    continue
                if not (snap.fovea_xy and snap.disc_xy and snap.scale_um_px and np.any(snap.mask > 0)):
                    continue
                self.case_id = case_id
                row = self._snap_row(snap)
                # 「第2」「最終」は画面上の役割名。画像名に付けると第1と突合できない。
                filename = snap.filename if case_id in {"", "未分類", "第2", "最終"} else f"{case_id}__{snap.filename}"
                row["path"] = snap.path
                row["shape"] = tuple(snap.gray.shape)
                row["mask"] = snap.mask
                row["filename"] = filename
                row["case_id"] = case_id
                review = snap.needs_review or ""
                if row.get("scale_outlier"):
                    flags = [part for part in review.split(";") if part]
                    if "scale_outlier" not in flags:
                        flags.append("scale_outlier")
                    review = ";".join(flags)
                row["needs_review"] = review
                rows.append(row)
        finally:
            self.case_id = saved
        return rows

    def measurement_gaps(self) -> list[tuple[str, str]]:
        """数値を書かない画像と理由。日付や間隔では外さない。"""
        self._stash()
        gaps: list[tuple[str, str]] = []
        for path in self.files:
            reason = self.load_errors.get(path.name)
            if reason:
                gaps.append((path.name, reason))
                continue
            snap = self.snap_for(path.name)
            if snap is None or not snap.confirmed:
                if snap is not None and snap.mask_stale:
                    gaps.append((path.name, "未確定（マスクが古いので再セグが必要）"))
                else:
                    gaps.append((path.name, "未確定"))
                continue
            if not (snap.fovea_xy and snap.disc_xy and snap.scale_um_px):
                gaps.append((path.name, "乳頭・中心窩・スケールが揃っていない"))
                continue
            if snap.mask is None or not np.any(snap.mask > 0):
                gaps.append((path.name, "マスクが空"))
        return gaps

    def exclusion_report(self) -> tuple[list[tuple[str, str]], str]:
        """画面と reading_exclusions.log に出す文。外した分はログにも残す。"""
        excluded = list(self.folder_exclusions) + self.measurement_gaps()
        self._log_exclusions(excluded)
        considered = len(self.files) + len(self.folder_exclusions)
        text = exclusion_log_text(considered, len(self.reading_payload()), excluded)
        return excluded, text

    def _open_status(self, head: str) -> str:
        excluded = list(self.folder_exclusions)
        excluded.extend(self.load_errors.items())
        if not excluded:
            return f"{head}。画像 {len(self.files)} 枚"
        listed = "、".join(f"{name}（{reason}）" for name, reason in excluded)
        return f"{head}。画像 {len(self.files)} 枚。外しました: {listed}"

    def _log_exclusions(self, excluded: list[tuple[str, str]]) -> None:
        for name, reason in excluded:
            logger.warning("読影から外しました %s: %s", name, reason)

    def snap_for(self, filename: str) -> VisitSnap | None:
        return self.snaps.get(f"{self.case_id}/{filename}")

    def select_case(self, case_id: str) -> None:
        self._stash()
        if case_id != self.case_id:
            self._clear_series_result()
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

    def _visit_key(self, path: Path) -> tuple:
        """検査日が早い順。日付が無いファイルは名前順で後ろ。mtime は使わない。"""
        snap = self.snap_for(path.name)
        if snap is not None and snap.date_iso:
            return (0, snap.date_iso, path.name.lower())
        dt, source = extract_image_datetime(path)
        if dt is not None and source != "mtime":
            return (0, datetime_to_iso(dt), path.name.lower())
        return (1, "", path.name.lower())

    def _anchor_path(self) -> Path | None:
        if not self.files:
            return None
        return min(self.files, key=self._visit_key)

    def _is_anchor_visit(self) -> bool:
        anchor = self._anchor_path()
        if anchor is None or not self.files:
            return False
        return self.files[self.index].name == anchor.name

    def _series_anchor(self) -> VisitSnap | None:
        """検査日がいちばん早い時点。いま開いている画像がそれなら使わない。"""
        anchor = self._anchor_path()
        if anchor is None or not self.files or self._is_anchor_visit():
            return None
        snap = self.snap_for(anchor.name)
        if snap is None or not snap.fovea_user_placed:
            return None
        if snap.fovea_xy and snap.disc_xy and snap.scale_um_px and snap.disc_radius_px:
            return snap
        return None

    def _propagate_anchor(self) -> int:
        """1枚目の距離を、乳頭が済んでいる後の時点へ写す。マスクは再セグまで古い。"""
        if not self._is_anchor_visit() or not self.files:
            return 0
        if not (
            self.fovea_user_placed
            and self.fovea_xy
            and self.disc_xy
            and self.scale_um_px
            and self.disc_radius_px
        ):
            return 0
        self._stash()
        vector = (
            float(self.fovea_xy[0]) - float(self.disc_xy[0]),
            float(self.fovea_xy[1]) - float(self.disc_xy[1]),
        )
        current = self.files[self.index].name
        changed = 0
        for path in self.files:
            if path.name == current:
                continue
            snap = self.snap_for(path.name)
            if snap is None or snap.gray is None or snap.disc_xy is None or not snap.disc_radius_px:
                continue
            if snap.fovea_xy is not None and (snap.fovea_user_placed or snap.registration_zoom):
                # クリック済み、または血管で写した位置は、1枚目の中心窩を
                # 直しても動かさない。スケールだけ 1枚目の µm/px ÷ 保存した倍率で追従する。
                fovea = snap.fovea_xy
            else:
                fovea = fovea_from_reference(
                    snap.disc_xy, vector, self.disc_radius_px, snap.disc_radius_px, snap.gray.shape[:2],
                )
            if snap.registration_zoom:
                # 血管の倍率で決めたスケールは、1枚目のスケールにだけ追従する。
                scale = float(self.scale_um_px) / float(snap.registration_zoom)
            else:
                try:
                    scale = series_scale_from_dfd(
                        self.scale_um_px, self.disc_xy, self.fovea_xy, snap.disc_xy, fovea,
                    )
                except ValueError:
                    continue
            moved = (
                snap.scale_um_px is None
                or abs(float(snap.scale_um_px) - scale) > 1e-4
                or snap.fovea_xy is None
                or abs(float(snap.fovea_xy[0]) - fovea[0]) > 0.5
                or abs(float(snap.fovea_xy[1]) - fovea[1]) > 0.5
            )
            snap.scale_um_px = scale
            snap.fovea_xy = fovea
            if moved:
                snap.confirmed = False
                snap.mask_stale = True
                changed += 1
        return changed

    def set_min_lesion(self, text: str) -> str:
        try:
            value = float(str(text).strip())
        except ValueError:
            return "最小病変は数値で指定してください"
        if value < 0:
            return "最小病変は 0 以上にしてください"
        self.min_lesion_mm2 = value
        return f"最小病変 {value:g} mm² 未満は数えません"

    def _capture_baseline(self) -> None:
        if self.gray is None or self.fovea_xy is None or self.disc_xy is None or not self.scale_um_px:
            return
        self.baseline_gain, self.baseline_threshold = baseline_appearance(
            self.gray, self.fovea_xy, self.disc_xy, self.scale_um_px,
        )

    def _shared_look(self) -> tuple[float | None, float | None]:
        if self.baseline_gain is not None and self.baseline_threshold is not None:
            return self.baseline_gain, self.baseline_threshold
        anchor = self._anchor_path()
        if anchor is None:
            return None, None
        snap = self.snaps.get(f"{self.case_id}/{anchor.name}")
        if snap is None or snap.contrast_gain is None or snap.hypo_threshold is None:
            return None, None
        return snap.contrast_gain, snap.hypo_threshold

    def _prior_measured(self) -> VisitSnap | None:
        if not self.files:
            return None
        current = self.files[self.index].name
        found: list[VisitSnap] = []
        for path in self.files:
            if path.name == current:
                continue
            snap = self.snaps.get(f"{self.case_id}/{path.name}")
            if (
                snap is None
                or not snap.confirmed
                or snap.mask is None
                or snap.disc_xy is None
                or not snap.disc_radius_px
                or not np.any(snap.mask > 0)
            ):
                continue
            found.append(snap)
        if not found:
            return None
        earlier = [snap for snap in found if snap.date_iso and self.date_iso and snap.date_iso < self.date_iso]
        pool = earlier or found
        pool.sort(key=lambda snap: snap.date_iso)
        return pool[-1]

    def _fovea_reference_snap(self) -> VisitSnap | None:
        """中心窩を写す元の時点。誤差が足し合わさらないよう、まず1枚目へ直接。"""
        anchor = self._series_anchor()
        if anchor is not None and anchor.gray is not None and anchor.fovea_xy is not None:
            return anchor
        prior = self._prior_measured()
        if prior is not None and prior.gray is not None and prior.fovea_xy is not None and prior.fovea_user_placed:
            return prior
        return None

    def _fovea_proposal_note(self) -> str:
        """血管で写した暫定位置から、クリックが大きく離れたら確認を促す文。"""
        if (
            self.fovea_proposal_xy is None
            or self.fovea_proposal_source != "registration"
            or self.fovea_xy is None
            or not self.scale_um_px
        ):
            return ""
        moved_um = float(np.hypot(
            float(self.fovea_xy[0]) - float(self.fovea_proposal_xy[0]),
            float(self.fovea_xy[1]) - float(self.fovea_proposal_xy[1]),
        )) * float(self.scale_um_px)
        if moved_um <= _FOVEA_PROPOSAL_WARN_UM:
            return ""
        return (
            f"前の時点から血管で写した中心窩より {moved_um:.0f} µm 離れています。"
            "中心窩の位置が前の時点とずれていないか確認してください"
        )

    def _offer_prior_mask(self) -> str:
        if self.gray is None or self.disc_xy is None or not self.disc_radius_px:
            return ""
        if self.mask is not None and np.any(self.mask > 0) and not self.mask_from_prior:
            return ""
        prior = self._prior_measured()
        if (
            prior is None
            or prior.mask is None
            or prior.disc_xy is None
            or prior.fovea_xy is None
            or self.fovea_xy is None
        ):
            return ""
        aligned, _preview = align_prior_mask(
            prior.mask,
            prior.disc_xy,
            prior.fovea_xy,
            self.disc_xy,
            self.fovea_xy,
            self.gray.shape[:2],
            prior_gray=prior.gray,
            gray=self.gray,
        )
        if not np.any(aligned > 0):
            return ""
        self.mask = aligned
        # 位置合わせの色分けは使わない。1枚目と同じ、グレーに緑のマスクで見せる。
        self.alignment_gray = None
        self.mask_from_prior = True
        self.mask_stale = False
        return (
            "前の時点のマスクを、1枚目と同じ緑の表示で初期輪郭にしました。"
            "中心窩を動かすと、1枚目との画素距離の比で µm/px を計算し直し、"
            "この輪郭もその倍率で拡大縮小します。"
            "縁まで、塗る、消すで直せます"
        )

    def _with_prior(self, text: str) -> str:
        extra = self._offer_prior_mask()
        if not extra:
            return text
        return f"{text}。{extra}"

    def place_disc(self, xy: tuple[float, float]) -> str:
        if self.gray is None:
            return "画像がありません"
        center, radius = stable_disc_radius(self.gray, xy)
        self.disc_xy = center
        self.disc_radius_px = radius
        anchor = self._series_anchor()
        if (
            anchor is not None
            and radius
            and anchor.disc_radius_px
            and anchor.scale_um_px
            and anchor.fovea_xy
            and anchor.disc_xy
        ):
            kept_fovea = self.fovea_user_placed and self.fovea_xy is not None
            registered_scale = False
            if not kept_fovea:
                # 乳頭ベクトルは撮影間の回転を無視するので、まず血管の
                # 位置合わせで前の時点の中心窩を写す。失敗したらベクトル。
                proposal = None
                reference = self._fovea_reference_snap()
                if reference is not None:
                    proposal = register_fovea_from_prior(
                        reference.gray, reference.fovea_xy, self.gray,
                        scale_um_px=reference.scale_um_px,
                    )
                if proposal is not None:
                    fovea, pair_zoom = proposal
                    self.fovea_xy = fovea
                    self.fovea_proposal_source = "registration"
                    # 元が1枚目でないときも、1枚目に対する倍率に直して持つ。
                    self.registration_zoom = (reference.registration_zoom or 1.0) * pair_zoom
                    if reference.scale_um_px:
                        self.scale_um_px = float(reference.scale_um_px) / pair_zoom
                        registered_scale = True
                else:
                    vector = (
                        float(anchor.fovea_xy[0]) - float(anchor.disc_xy[0]),
                        float(anchor.fovea_xy[1]) - float(anchor.disc_xy[1]),
                    )
                    self.fovea_xy = fovea_from_reference(
                        center, vector, anchor.disc_radius_px, radius, self.gray.shape[:2],
                    )
                    self.fovea_proposal_source = "vector"
                    self.registration_zoom = None
                self.fovea_proposal_xy = self.fovea_xy
                self.fovea_user_placed = False
                self.tool = "fovea"
            elif self.registration_zoom and self.scale_um_px:
                # 血管の倍率で決めたスケールは、乳頭の置き直しでは変わらない。
                registered_scale = True
            if not registered_scale:
                try:
                    self.scale_um_px = series_scale_from_dfd(
                        anchor.scale_um_px, anchor.disc_xy, anchor.fovea_xy, center, self.fovea_xy,
                    )
                except ValueError:
                    return "乳頭と中心窩が近すぎて、スケールを計算できません"
            self.confirmed = False
            self.mask_stale = True
            placed = "disc-kept" if kept_fovea else "disc"
            source = "registration" if registered_scale else "series"
            text = _scale_message(source, self.scale_um_px, placed=placed)
            if not kept_fovea and self.fovea_proposal_source == "registration":
                text += "。暫定の中心窩とスケールは、前の時点と血管を合わせて写しました"
            return self._with_prior(text)
        if self.fovea_user_placed and self.fovea_xy is not None:
            center, scale, source = measurement_scale(
                self.gray, center, self.fovea_xy, previous=self.scale_um_px,
            )
            self.disc_xy = center
            if scale is not None:
                self.scale_um_px = scale
            self.confirmed = False
            self.mask_stale = True
            text = self._with_prior(_scale_message(source, self.scale_um_px, placed="disc-moved"))
            changed = self._propagate_anchor()
            if changed:
                text += f"。後の時点 {changed} 件は換算を更新したので、再セグしてから確定し直してください"
            return text
        provisional = provisional_scale_from_disc(self.gray, self.disc_xy)
        self.fovea_xy = propose_fovea_from_disc(
            self.disc_xy, self.gray.shape[:2], scale_um_px=provisional,
        )
        self.scale_um_px = provisional
        self.tool = "fovea"
        return self._with_prior(
            "乳頭を置きました。中心窩をクリックすると、この画像の距離が基準になります。"
            "後の画像は、その距離を乳頭の大きさで合わせます"
        )

    def place_fovea(self, xy: tuple[float, float]) -> str:
        if self.gray is None:
            return "画像がありません"
        previous_fovea = self.fovea_xy
        first_placement = not self.fovea_user_placed
        self.fovea_xy = (float(xy[0]), float(xy[1]))
        source = "none"
        snapped_note = ""
        anchor = self._series_anchor()
        if (
            anchor is not None
            and self.disc_xy is not None
            and anchor.scale_um_px
            and anchor.disc_xy
            and anchor.fovea_xy
        ):
            handled = False
            if (
                self.fovea_proposal_source == "registration"
                and self.fovea_proposal_xy is not None
                and self.registration_zoom
                and self.scale_um_px
            ):
                gap_um = float(np.hypot(
                    self.fovea_xy[0] - float(self.fovea_proposal_xy[0]),
                    self.fovea_xy[1] - float(self.fovea_proposal_xy[1]),
                )) * float(self.scale_um_px)
                if gap_um <= _FOVEA_PROPOSAL_WARN_UM:
                    # クリックは「写した位置で合っている」という確認とみなして
                    # 写した位置に合わせる。±数 px のクリック差で中心窩温存の
                    # 指標（GAWAIN など）が数点動くのを防ぐ。
                    self.fovea_xy = self.fovea_proposal_xy
                    source = "registration"
                    snapped_note = (
                        f"クリックは血管で写した中心窩と {gap_um:.0f} µm 差だったので、"
                        "写した位置に置きました。スケールも血管の倍率のままです"
                    )
                    handled = True
            if not handled:
                # 位置合わせがまだ効いていなければ（乳頭時に失敗、または前の
                # クリックで手動優先に落ちた）、手動クリックを中心にして
                # 周辺の歪みを避けつつもう一度だけ合わせ直す。
                reference = self._fovea_reference_snap()
                retry = None
                if reference is not None:
                    retry = register_fovea_from_prior(
                        reference.gray, reference.fovea_xy, self.gray,
                        scale_um_px=reference.scale_um_px,
                        center_hint=self.fovea_xy,
                    )
                if retry is not None and reference.scale_um_px:
                    fovea, pair_zoom = retry
                    scale = float(reference.scale_um_px) / pair_zoom
                    gap_um = float(np.hypot(
                        self.fovea_xy[0] - fovea[0], self.fovea_xy[1] - fovea[1],
                    )) * scale
                    if gap_um <= _FOVEA_PROPOSAL_WARN_UM:
                        self.fovea_xy = fovea
                        self.fovea_proposal_xy = fovea
                        self.fovea_proposal_source = "registration"
                        self.registration_zoom = (reference.registration_zoom or 1.0) * pair_zoom
                        self.scale_um_px = scale
                        source = "registration"
                        snapped_note = (
                            f"クリックを中心に血管で合わせ直し、{gap_um:.0f} µm 差だったので"
                            "写した位置に置きました。スケールも血管の倍率で計算しました"
                        )
                        handled = True
            if not handled:
                try:
                    self.scale_um_px = series_scale_from_dfd(
                        anchor.scale_um_px, anchor.disc_xy, anchor.fovea_xy,
                        self.disc_xy, self.fovea_xy,
                    )
                except ValueError:
                    self.fovea_xy = previous_fovea
                    return "乳頭と中心窩が近すぎて、スケールを計算できません"
                self.registration_zoom = None
                source = "series"
        elif self.disc_xy is not None:
            _center, scale, source = measurement_scale(
                self.gray,
                self.disc_xy,
                self.fovea_xy,
                previous=None if first_placement else self.scale_um_px,
            )
            if scale is not None:
                self.scale_um_px = scale
        self.confirmed = False
        self.fovea_user_placed = True
        self.mask_stale = True
        self.pan_xy = (0.0, 0.0)
        if first_placement and self.disc_xy is not None and self.scale_um_px and not self.mask_from_prior:
            if self._is_anchor_visit():
                self._capture_baseline()
            self.resegment()
            text = _scale_message(source, self.scale_um_px, placed="fovea-first")
        elif self.mask_from_prior and self.scale_um_px:
            self.alignment_gray = None
            realigned = self._offer_prior_mask()
            if realigned:
                self.mask_stale = False
                placed = "fovea-first" if first_placement else "fovea-moved"
                text = (
                    _scale_message(source, self.scale_um_px, placed=placed)
                    + "前の時点の輪郭を、乳頭と中心窩の距離に合わせて拡大縮小しました。"
                    "縁まで、塗る、消すで直せます。"
                )
            elif previous_fovea is not None and self.mask is not None:
                self.mask = translate_mask(
                    self.mask,
                    self.fovea_xy[0] - float(previous_fovea[0]),
                    self.fovea_xy[1] - float(previous_fovea[1]),
                )
                self.mask_stale = False
                placed = "fovea-first" if first_placement else "fovea-moved"
                text = (
                    _scale_message(source, self.scale_um_px, placed=placed)
                    + "前の時点の輪郭を、中心窩と同じだけ動かしました。縁まで、塗る、消すで直せます。"
                )
            else:
                placed = "fovea-first" if first_placement else "fovea-moved"
                text = _scale_message(source, self.scale_um_px, placed=placed)
        elif self.scale_um_px:
            text = _scale_message(source, self.scale_um_px, placed="fovea-moved")
        else:
            return "中心窩を移動しました。乳頭が決まるとこの時点のスケールになります"
        if self.gray is not None and self.disc_xy is not None and self.fovea_xy is not None and self.scale_um_px:
            rim = provisional_scale_from_disc(self.gray, self.disc_xy)
            dfd = scale_from_dfd(self.disc_xy, self.fovea_xy)
            note = scale_disagreement_note(dfd, rim or self.scale_um_px)
            if note:
                held = "" if source == "series" else "換算は変えていません。"
                text = f"{text}。位置の確認です。{held}{note}"
        if snapped_note:
            text += "。" + snapped_note
        warn = self._fovea_proposal_note()
        if warn:
            text += "。" + warn
        changed = self._propagate_anchor()
        if changed:
            text += f"。後の時点 {changed} 件は換算を更新したので、再セグしてから確定し直してください"
        return text

    def confirm_fovea(self) -> str:
        """中心窩の位置を確定し、位置合わせの色をやめていつものマスク色に戻す。"""
        if self.gray is None or self.disc_xy is None or self.fovea_xy is None:
            return "先に乳頭と中心窩の位置を決めてください"
        if not self.fovea_user_placed and self._is_anchor_visit():
            return "1枚目は、中心窩らしい場所をクリックしてから「中心窩確定」を押してください"
        if self.mask is None or not np.any(self.mask > 0):
            self.mask_stale = True
            return "中心窩を確定できません。マスクがないので「再セグ」を押してください"
        if self.mask_stale:
            return "中心窩を確定できません。マスクが古いので「再セグ」で作り直してください"
        self.fovea_user_placed = True
        self.alignment_gray = None
        self.tool = "pan"
        warn = self._fovea_proposal_note()
        suffix = f"。{warn}" if warn else ""
        return f"中心窩を確定しました。マスクをいつもの色で表示しています{suffix}"

    def resegment(self) -> str:
        if self.gray is None or self.fovea_xy is None or self.disc_xy is None or not self.scale_um_px:
            return "乳頭と中心窩が必要です"
        self._push_undo()
        x0, y0, x1, y1 = window_box(self.gray.shape[:2], self.fovea_xy, self.scale_um_px)
        crop = self.gray[y0:y1, x0:x1]
        local_f = shift_point(self.fovea_xy, (x0, y0))
        local_d = shift_point(self.disc_xy, (x0, y0))
        assert local_f is not None and local_d is not None
        kept_prior = self.mask_from_prior
        gain, percentile = self._shared_look()
        local = segment_optos(
            crop,
            local_f,
            local_d,
            self.scale_um_px,
            min_lesion_mm2=self.min_lesion_mm2,
            contrast_gain=gain,
            hypo_percentile=percentile,
        )
        full = np.zeros(self.gray.shape[:2], dtype=np.uint8)
        full[y0:y1, x0:x1] = local
        before = int(np.count_nonzero(full))
        full = apply_reading_exclusions(full, self.gray, self.disc_xy, self.scale_um_px)
        after = int(np.count_nonzero(full))
        logger.info("除外前 %d px、除外後 %d px", before, after)
        if after == 0:
            if self.undo:
                self.mask = self.undo.pop()
            self.mask_from_prior = kept_prior
            self.mask_stale = not (self.mask is not None and np.any(self.mask > 0))
            reason = (
                f"再セグの結果が空です。前のマスクは残しています。"
                f"除外前 {before} px、除外後 {after} px。"
            )
            if before == 0:
                reason += "しきい値のあとで病変が残っていません。"
            else:
                reason += "除外マスクで病変が全部消えました。"
            logger.warning(reason)
            return reason
        self.mask_from_prior = False
        self.alignment_gray = None
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
            image = self.gray
            if self.disc_xy is not None:
                image = growth_image(self.gray, self.disc_xy, self.scale_um_px)
            self.mask, added = grow_to_edge(
                image, self.mask, (x, y), self.scale_um_px,
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

    def _area_review(self) -> tuple[str, str]:
        """直前の確定時点との面積差と、縁まで（0→正）の逆行。画面の文と needs_review。"""
        stats = self.live_stats()
        if stats is None or stats.get("area_mm2") is None or not self.date_iso or not self.files:
            return "", ""
        current_name = self.files[self.index].name
        earlier: list[tuple[str, float, VisitSnap]] = []
        for path in self.files:
            if path.name == current_name:
                continue
            snap = self.snaps.get(f"{self.case_id}/{path.name}")
            if (
                snap is None
                or not snap.confirmed
                or not snap.date_iso
                or snap.date_iso >= self.date_iso
                or snap.mask is None
                or snap.fovea_xy is None
                or not snap.scale_um_px
            ):
                continue
            used = self.measurement_scale_um_px(snap.scale_um_px)
            if used is None:
                continue
            previous = stats_central(snap.mask, snap.fovea_xy, used, snap.disc_xy)
            if previous.get("area_mm2") is None:
                continue
            earlier.append((snap.date_iso, float(previous["area_mm2"]), snap))
        if not earlier:
            return "", ""
        earlier.sort(key=lambda item: item[0])
        prev_date, prev_area, prev_snap = earlier[-1]
        notes = area_change_warnings([
            (prev_date, prev_area),
            (self.date_iso, float(stats["area_mm2"])),
        ])
        prev_used = self.measurement_scale_um_px(prev_snap.scale_um_px)
        current_used = self.measurement_scale_um_px(self.scale_um_px)
        prev_gap = robust_foveal_gap_um(prev_snap.mask, prev_snap.fovea_xy, prev_used or prev_snap.scale_um_px)
        gap = robust_foveal_gap_um(self.mask, self.fovea_xy, current_used or self.scale_um_px)
        notes += gap_reopen_warnings([
            (prev_date, prev_gap),
            (self.date_iso, gap),
        ])
        if not notes:
            return "", ""
        codes: list[str] = []
        if any("減りました" in note for note in notes):
            codes.append("area_decrease")
        if any("2倍" in note for note in notes):
            codes.append("area_doubled")
        if any("萎縮は戻らない" in note for note in notes):
            codes.append("gap_reopened")
        return " ".join(notes), ";".join(codes)

    def confirm(self) -> str:
        accepted_proposal = False
        if not self.fovea_user_placed:
            # 2枚目以降は、血管で写した中心窩を動かさずに輪郭だけ直して確定できる。
            # 1枚目の仮の中心窩は、クリックするまで確定しない。
            # ここでは印を付けない。マスク検査で戻ったあとに付ける。
            if self._series_anchor() is not None and self.fovea_xy is not None:
                accepted_proposal = True
            else:
                return "中心窩をクリックしてから確定します"
        if self.fovea_xy is None or self.disc_xy is None or self.mask is None or not self.scale_um_px:
            return "乳頭、中心窩、マスクが揃ってから確定します"
        if self.mask_stale:
            return "マスクが古いので確定できません。「再セグ」で作り直してください"
        if not np.any(self.mask > 0):
            return "マスクに領域がないので確定できません。縁まで、または塗りで囲んでください"
        if accepted_proposal:
            self.fovea_user_placed = True
            self.alignment_gray = None
        self.confirmed = True
        note, review = self._area_review()
        self.area_note = note
        self.needs_review = review
        self._stash()
        done = sum(1 for s in self._case_snaps() if s.confirmed)
        text = f"この時点を確定 ({done}/{len(self.files)})"
        if accepted_proposal:
            text += "。中心窩は写した位置のまま記録しました"
        if note:
            text += "。" + note
        return text

    def next_instruction(self) -> str:
        """画面に出す、いま押すべき次の一手。"""
        if not self.folder:
            return "フォルダのパスを入れて「開く」を押すか、「フォルダ選択」で画像フォルダを選んでください。"
        if self.gray is None:
            return "症例と時点を選んでください。"
        if self.disc_xy is None:
            return "「乳頭」のまま、乳頭の中心を1回クリックしてください。"
        if not self.fovea_user_placed:
            if self._series_anchor() is not None:
                rule = (
                    "2枚目以降は、中心窩を動かすと、1枚目の乳頭–中心窩の画素距離との比で µm/px を計算し直します。"
                    "前の時点の輪郭も、その倍率で拡大縮小します。"
                )
            else:
                rule = (
                    "1枚目は、その距離がこの眼の基準になります。"
                    "点を少し動かしても、面積の換算は変わりません。"
                )
            return (
                "「中心窩」で、中心窩らしい場所をクリックしてください。"
                + rule
                + "位置がよければ「中心窩確定」。マスクがいつもの色に戻ります。"
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
            text = "全部の時点が確定済みです。「時系列を計算」を押すと進行が出ます。年齢を入れると余命も足されます。"
        elif done >= 2:
            text = (
                f"確定 {done}/{total}。"
                "次の時点を開いて乳頭から始めるか、「時系列を計算」でここまでの進行を見てください。"
                "年齢を入れると余命も足されます。"
            )
        else:
            text = f"確定 {done}/{total}。時点の一覧で次の画像を開き、また乳頭の中心から始めてください。"
        if self.area_note:
            return f"{text} {self.area_note}"
        return text

    def _usable_scale_estimate(
        self,
        scale: float | None,
        *,
        confirmed: bool,
        fovea_user_placed: bool,
    ) -> bool:
        return bool(scale) and float(scale) > 0 and (confirmed or fovea_user_placed)

    def _scale_estimates(self) -> list[float]:
        """この眼の時点ごとの推定 µm/px。仮の乳頭スケールは入れない。"""
        values: list[float] = []
        current = self.files[self.index].name if self.files else None
        for snap in self._case_snaps():
            if current is not None and snap.filename == current:
                continue
            if self._usable_scale_estimate(
                snap.scale_um_px,
                confirmed=snap.confirmed,
                fovea_user_placed=snap.fovea_user_placed,
            ):
                values.append(float(snap.scale_um_px))
        if self._usable_scale_estimate(
            self.scale_um_px,
            confirmed=self.confirmed,
            fovea_user_placed=self.fovea_user_placed,
        ):
            values.append(float(self.scale_um_px))
        return values

    def eye_scale_um_px(self) -> float | None:
        """面積換算に使う、この眼の固定 µm/px（時点推定の中央値）。"""
        return fixed_eye_scale(self._scale_estimates())

    def measurement_scale_um_px(self, estimate: float | None = None) -> float | None:
        """計測に使う µm/px。眼の中央値が無ければ、その時点の推定。"""
        fixed = self.eye_scale_um_px()
        if fixed is not None:
            return fixed
        if estimate is not None and float(estimate) > 0:
            return float(estimate)
        return None

    def live_stats(self) -> dict | None:
        used = self.measurement_scale_um_px(self.scale_um_px)
        if (
            self.mask is None
            or self.fovea_xy is None
            or not used
            or not np.any(self.mask > 0)
        ):
            return None
        return stats_central(self.mask, self.fovea_xy, used, self.disc_xy)

    def _snap_row(self, snap: VisitSnap) -> dict:
        assert snap.fovea_xy and snap.disc_xy and snap.scale_um_px and snap.mask is not None
        estimate = float(snap.scale_um_px)
        used = self.measurement_scale_um_px(estimate)
        assert used is not None
        stats = stats_central(snap.mask, snap.fovea_xy, used, snap.disc_xy)
        gap = robust_foveal_gap_um(snap.mask, snap.fovea_xy, used)
        outlier = is_scale_outlier(estimate, used)
        return {
            "case_id": self.case_id,
            "date": snap.date_iso,
            "filename": snap.filename,
            "confirmed": int(snap.confirmed),
            "scale_um_per_px": round(float(used), 4),
            "disc_x": round(float(snap.disc_xy[0]), 1),
            "disc_y": round(float(snap.disc_xy[1]), 1),
            "fovea_x": round(float(snap.fovea_xy[0]), 1),
            "fovea_y": round(float(snap.fovea_xy[1]), 1),
            "area_mm2": stats.get("area_mm2"),
            "area_mm2_24mm": stats.get("area_mm2_24mm"),
            "area_mm_sqrt": stats.get("area_mm_sqrt"),
            "lesion_count": stats.get("lesion_count"),
            "gawain": stats.get("gawain"),
            "dist_um_nearest": stats.get("dist_um"),
            "gap_um": None if gap is None else round(gap, 1),
            "intact_1mm_mm2": round(intact_within_mm2(snap.mask, snap.fovea_xy, used), 4),
            "outside_6mm_px": stats.get("outside_quant_px"),
            "mask_px": int(np.count_nonzero(snap.mask > 0)),
            "scale_estimate_um_per_px": round(estimate, 4),
            "scale_outlier": int(outlier),
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
            used = self.measurement_scale_um_px(s.scale_um_px)
            assert used is not None
            stats = stats_central(s.mask, s.fovea_xy, used)
            gap = robust_foveal_gap_um(s.mask, s.fovea_xy, used)
            intact = intact_within_mm2(s.mask, s.fovea_xy, used)
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
        warnings = area_change_warnings([
            (row["date"], float(row["stats"]["area_mm2"]))
            for row in rows
            if row.get("date") and row["stats"].get("area_mm2") is not None
        ])
        parts.extend(warnings)
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
            "area_warnings": warnings,
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
        try:
            self.gray = load_gray(path)
        except (OSError, ValueError) as exc:
            self.load_errors[path.name] = f"画像を読めない（{exc}）"
            logger.warning("読影から外しました %s: %s", path.name, self.load_errors[path.name])
            self.gray = None
            self.disc_xy = None
            self.fovea_xy = None
            self.scale_um_px = None
            self.disc_radius_px = None
            self.mask = None
            self.confirmed = False
            self.tool = "disc"
            return f"{path.name} を読めません。{self.load_errors[path.name]}"
        self.load_errors.pop(path.name, None)
        self.disc_xy = None
        self.fovea_xy = None
        self.scale_um_px = None
        self.disc_radius_px = None
        self.mask = np.zeros(self.gray.shape[:2], dtype=np.uint8)
        self.confirmed = False
        self.mask_from_prior = False
        self.alignment_gray = None
        self.needs_review = ""
        self.area_note = ""
        self.fovea_proposal_xy = None
        self.fovea_proposal_source = ""
        self.registration_zoom = None
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
            disc_radius_px=self.disc_radius_px,
            mask=None if self.mask is None else self.mask.copy(),
            confirmed=self.confirmed,
            fovea_user_placed=self.fovea_user_placed,
            mask_stale=self.mask_stale,
            contrast_gain=self.baseline_gain,
            hypo_threshold=self.baseline_threshold,
            needs_review=self.needs_review,
            area_note=self.area_note,
            mask_from_prior=self.mask_from_prior,
            fovea_proposal_xy=self.fovea_proposal_xy,
            fovea_proposal_source=self.fovea_proposal_source,
            registration_zoom=self.registration_zoom,
        )

    def _restore(self, snap: VisitSnap) -> None:
        self._clear_edit_state()
        self.gray = snap.gray
        self.disc_xy = snap.disc_xy
        self.fovea_xy = snap.fovea_xy
        self.scale_um_px = snap.scale_um_px
        self.disc_radius_px = snap.disc_radius_px
        self.mask = None if snap.mask is None else snap.mask.copy()
        self.date_iso = snap.date_iso
        self.date_source = snap.date_source
        self.confirmed = snap.confirmed
        self.fovea_user_placed = snap.fovea_user_placed
        self.mask_stale = snap.mask_stale
        if snap.contrast_gain is not None:
            self.baseline_gain = snap.contrast_gain
            self.baseline_threshold = snap.hypo_threshold
        self.needs_review = snap.needs_review
        self.area_note = snap.area_note
        self.mask_from_prior = snap.mask_from_prior
        self.fovea_proposal_xy = snap.fovea_proposal_xy
        self.fovea_proposal_source = snap.fovea_proposal_source
        self.registration_zoom = snap.registration_zoom
        self.alignment_gray = None
        if self.fovea_xy is None:
            self.tool = "disc"
        elif not self.fovea_user_placed:
            self.tool = "fovea"
        else:
            self.tool = "pan"

    def _clear_current(self) -> None:
        self.gray = None
        self.disc_xy = None
        self.fovea_xy = None
        self.scale_um_px = None
        self.disc_radius_px = None
        self.mask = None
        self.date_iso = ""
        self.date_source = "missing"
        self.confirmed = False
        self.mask_from_prior = False
        self.alignment_gray = None
        self.needs_review = ""
        self.area_note = ""
        self._clear_edit_state()
        self._clear_series_result()

    def _clear_edit_state(self) -> None:
        self.undo = []
        self.redo = []
        self.clear_pts = []
        self.fovea_user_placed = False
        self.mask_stale = False
        self.mask_from_prior = False
        self.fovea_proposal_xy = None
        self.fovea_proposal_source = ""
        self.registration_zoom = None
        self.zoom = 1
        self.pan_xy = (0.0, 0.0)

    def _clear_series_result(self) -> None:
        """時系列の要約は症例単位。受診の切り替えでは消さない。"""
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
