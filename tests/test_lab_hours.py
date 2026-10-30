import datetime

import pytest

from rotten_rl.lab_hours import LAB_LAST_LAUNCH, next_time

T = datetime.datetime


@pytest.mark.parametrize("now, expected", [
    (T(2026, 10, 29, 23, 0), T(2026, 10, 30, 6, 55)),   # Thursday night: Friday
    (T(2026, 10, 30, 1, 0), T(2026, 10, 30, 6, 55)),    # early Friday: that day
    (T(2026, 10, 30, 22, 10), T(2026, 11, 2, 6, 55)),   # Friday night: Monday
    (T(2026, 10, 31, 3, 0), T(2026, 11, 2, 6, 55)),     # Saturday
    (T(2026, 11, 1, 23, 0), T(2026, 11, 2, 6, 55)),     # Sunday
    (T(2026, 11, 2, 6, 54), T(2026, 11, 2, 6, 55)),     # Monday before the cut
    (T(2026, 11, 2, 22, 0), T(2026, 11, 3, 6, 55)),     # Monday night: Tuesday
])
def test_lab_stops_next_weekday_morning(now, expected):
    assert next_time("lab", now=now) == expected


def test_lab_last_launch():
    assert next_time("lab", LAB_LAST_LAUNCH, now=T(2026, 10, 30, 22, 10)) == T(2026, 11, 2, 6, 0)


def test_fixed_time_ignores_the_day():
    assert next_time("06:55", now=T(2026, 10, 30, 22, 10)) == T(2026, 10, 31, 6, 55)
    assert next_time("14:00", now=T(2026, 10, 31, 9, 0)) == T(2026, 10, 31, 14, 0)
