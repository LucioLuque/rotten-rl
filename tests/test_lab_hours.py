import datetime

import pytest

from rotten_rl.lab_hours import (LAUNCH_EARLY, closed_spans, next_time, parse_events, window)

T = datetime.datetime

# Room events of Wednesday 4/11/2026 (local times): 13:00-14:30, 14:40-16:10, 18:00-19:30, 19:40-21:10.
WEDNESDAY = [(T(2026, 11, 4, 13, 0), T(2026, 11, 4, 14, 30)),
             (T(2026, 11, 4, 14, 40), T(2026, 11, 4, 16, 10)),
             (T(2026, 11, 4, 18, 0), T(2026, 11, 4, 19, 30)),
             (T(2026, 11, 4, 19, 40), T(2026, 11, 4, 21, 10))]


def test_weekday_reboot_is_an_event():
    # Tuesday night, no events: closes at 07:00 for the 07:30 reboot, reopens at 08:00.
    assert window([], T(2026, 11, 3, 23, 0)) == (True, T(2026, 11, 4, 7, 0), T(2026, 11, 4, 8, 0))
    assert window([], T(2026, 11, 4, 7, 10)) == (False, None, T(2026, 11, 4, 8, 0))


def test_weekend_without_events_runs_until_monday():
    assert window([], T(2026, 11, 6, 9, 0))[1] == T(2026, 11, 9, 7, 0)


def test_margins_and_late_end():
    # After the reboot the day is open until 12:30; the events merge (gaps shorter than the margins)
    # except 16:40-17:30; the one ending at 21:10 reopens at 22:20.
    spans = closed_spans(WEDNESDAY, T(2026, 11, 4, 9, 0), horizon=datetime.timedelta(hours=14))
    assert spans == [(T(2026, 11, 4, 12, 30), T(2026, 11, 4, 16, 40)),
                     (T(2026, 11, 4, 17, 30), T(2026, 11, 4, 22, 20))]


def test_short_gap_between_events():
    is_open, closes, reopens = window(WEDNESDAY, T(2026, 11, 4, 16, 45))
    assert (is_open, closes, reopens) == (True, T(2026, 11, 4, 17, 30), T(2026, 11, 4, 22, 20))


@pytest.mark.parametrize("end, reopens", [
    (T(2026, 11, 7, 20, 50), T(2026, 11, 7, 21, 20)),    # before 21:00: 30 minutes
    (T(2026, 11, 7, 21, 0), T(2026, 11, 7, 22, 20)),
    (T(2026, 11, 7, 22, 0), T(2026, 11, 7, 22, 30)),     # 22:20 would be less than 30 minutes
])
def test_reopen_after_evening_events(end, reopens):
    events = [(end - datetime.timedelta(hours=1), end)]
    assert window(events, end)[2] == reopens


def test_next_time_lab():
    assert next_time("lab", now=T(2026, 11, 3, 23, 0), events=[]) == T(2026, 11, 4, 6, 55)
    assert next_time("lab", LAUNCH_EARLY, now=T(2026, 11, 4, 9, 0), events=WEDNESDAY) == T(2026, 11, 4, 11, 30)
    now = T(2026, 11, 4, 13, 0)                          # closed: stop now
    assert next_time("lab", now=now, events=WEDNESDAY) == now


def test_fixed_times():
    assert next_time("06:55", now=T(2026, 10, 30, 22, 10)) == T(2026, 10, 31, 6, 55)
    assert next_time("14:00", now=T(2026, 10, 31, 9, 0)) == T(2026, 10, 31, 14, 0)
    assert next_time("2026-11-09 06:55") == T(2026, 11, 9, 6, 55)


ICS = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20261104T160000Z
DTEND:20261104T173000Z
SUMMARY:Busy
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=America/Argentina/Buenos_Aires:20261105T100000
DTEND;TZID=America/Argentina/Buenos_Aires:20261105T110000
END:VEVENT
BEGIN:VEVENT
STATUS:CANCELLED
DTSTART:20261106T160000Z
DTEND:20261106T173000Z
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20261107
DTEND;VALUE=DATE:20261108
END:VEVENT
END:VCALENDAR
"""


def test_parse_events(monkeypatch):
    monkeypatch.setenv("TZ", "America/Argentina/Buenos_Aires")
    import time
    time.tzset()
    assert parse_events(ICS) == [(T(2026, 11, 4, 13, 0), T(2026, 11, 4, 14, 30)),
                                 (T(2026, 11, 5, 10, 0), T(2026, 11, 5, 11, 0)),
                                 (T(2026, 11, 7), T(2026, 11, 8))]


def test_recurring_events_fail():
    with pytest.raises(ValueError):
        parse_events(ICS.replace("SUMMARY:Busy", "RRULE:FREQ=WEEKLY"))
