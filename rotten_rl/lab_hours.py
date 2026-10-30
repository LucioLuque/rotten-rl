"""
rotten_rl/lab_hours.py  —  When to stop according to the lab schedule.

Monday to Friday access ends at 07:00 and machines reboot at 07:30. Saturdays and Sundays have no
cut: runs can go straight from Friday 22:00 to Monday 07:00. So `--stop_at lab` means "the next
weekday morning": on Friday night that is Monday, not Saturday. Holidays are not detected: runs
stop anyway, which is the safe side.
"""

import datetime

LAB_STOP = "06:55"
LAB_LAST_LAUNCH = "06:00"


def next_time(spec, lab_time=LAB_STOP, now=None):
    """Next instant of `spec`: "HH:MM" (today if not past yet, else tomorrow) or "lab" (`lab_time`
    of the next Monday-to-Friday morning)."""
    now = now or datetime.datetime.now()
    hh, mm = (int(x) for x in (lab_time if spec == "lab" else spec).split(":"))
    when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if when <= now:
        when += datetime.timedelta(days=1)
    while spec == "lab" and when.weekday() >= 5:
        when += datetime.timedelta(days=1)
    return when
