"""症例グループ発見・振分のユニットテスト。"""

from pathlib import Path

from engine.cases import (
    UNCLASSIFIED_CASE_ID,
    create_case,
    discover_cases,
    move_file_to_case,
    ordered_case_ids,
)


def test_discover_cases_subfolders(tmp_path: Path):
    p1 = tmp_path / "patient1"
    p2 = tmp_path / "patient2"
    p1.mkdir()
    p2.mkdir()
    (p1 / "a.png").write_bytes(b"x")
    (p2 / "b.png").write_bytes(b"x")
    (tmp_path / "loose.png").write_bytes(b"x")
    # 非画像は無視
    (p1 / "note.txt").write_text("hi")

    cases = discover_cases(tmp_path)
    assert set(cases) == {"patient1", "patient2", UNCLASSIFIED_CASE_ID}
    assert [f.name for f in cases["patient1"]] == ["a.png"]
    assert [f.name for f in cases["patient2"]] == ["b.png"]
    assert [f.name for f in cases[UNCLASSIFIED_CASE_ID]] == ["loose.png"]
    assert ordered_case_ids(cases)[-1] == UNCLASSIFIED_CASE_ID


def test_discover_cases_root_only(tmp_path: Path):
    (tmp_path / "x.jpg").write_bytes(b"x")
    cases = discover_cases(tmp_path)
    assert list(cases.keys()) == [UNCLASSIFIED_CASE_ID]
    assert len(cases[UNCLASSIFIED_CASE_ID]) == 1


def test_create_and_move_case(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"x")
    cases = discover_cases(tmp_path)
    cases, cid = create_case(cases, "caseA")
    assert cid == "caseA"
    assert cases["caseA"] == []
    path = cases[UNCLASSIFIED_CASE_ID][0]
    cases = move_file_to_case(cases, path, UNCLASSIFIED_CASE_ID, "caseA")
    assert path in cases["caseA"]
    assert UNCLASSIFIED_CASE_ID not in cases or path not in cases.get(
        UNCLASSIFIED_CASE_ID, [],
    )
