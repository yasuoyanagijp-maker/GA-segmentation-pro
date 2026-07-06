"""life_table.py – 令和5年簡易生命表に基づく平均余命ルックアップ。

出典: 厚生労働省 令和5年簡易生命表（死亡率.pdf）
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

LIFE_TABLE_SOURCE = '令和5年（2023）簡易生命表 — 厚生労働省（死亡率.pdf）'
LIFE_TABLE_PATH = Path(__file__).resolve().parent.parent / 'data' / 'life_table_jp_r5.csv'
MIN_AGE = 0
MAX_AGE = 105

_SEX_ALIASES = {
    'male': 'male',
    'm': 'male',
    '男': 'male',
    '男性': 'male',
    'female': 'female',
    'f': 'female',
    '女': 'female',
    '女性': 'female',
}


def normalize_sex(sex: str) -> str:
    key = sex.strip().lower()
    if key in _SEX_ALIASES:
        return _SEX_ALIASES[key]
    raise ValueError(f'未対応の性別: {sex!r}（Male / Female を指定してください）')


@lru_cache(maxsize=1)
def _load_table() -> dict[str, dict[int, float]]:
    tables: dict[str, dict[int, float]] = {'male': {}, 'female': {}}
    with LIFE_TABLE_PATH.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sex = row['sex']
            age = int(row['age'])
            tables[sex][age] = float(row['life_expectancy_years'])
    return tables


def life_expectancy_years(age: int, sex: str) -> float:
    """年齢・性別から平均余命（年）を返す。"""
    sex_key = normalize_sex(sex)
    age_key = max(MIN_AGE, min(MAX_AGE, int(age)))
    table = _load_table()[sex_key]
    if age_key not in table:
        raise ValueError(f'生命表に年齢 {age_key} のデータがありません')
    return table[age_key]


def format_life_expectancy(age: int, sex: str, *, decimals: int = 1) -> str:
    """UI表示用に「12.1 年」形式で返す。"""
    years = life_expectancy_years(age, sex)
    return f'{years:.{decimals}f} 年'
