import pytest

from engine.life_table import (
    LIFE_TABLE_SOURCE,
    format_life_expectancy,
    life_expectancy_years,
    normalize_sex,
)


def test_normalize_sex_aliases():
    assert normalize_sex('Male') == 'male'
    assert normalize_sex('Female') == 'female'
    assert normalize_sex('男性') == 'male'


def test_life_expectancy_age_75_matches_life_table():
    assert life_expectancy_years(75, 'Male') == pytest.approx(12.14, abs=0.05)
    assert life_expectancy_years(75, 'Female') == pytest.approx(15.73, abs=0.05)


def test_life_expectancy_clamps_age():
    assert life_expectancy_years(150, 'Male') == life_expectancy_years(105, 'Male')


def test_format_life_expectancy():
    text = format_life_expectancy(75, 'Male')
    assert text.endswith('年')
    assert '12.' in text


def test_life_table_source_documented():
    assert '令和5年' in LIFE_TABLE_SOURCE
    assert '死亡率.pdf' in LIFE_TABLE_SOURCE
