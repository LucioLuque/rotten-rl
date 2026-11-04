"""
rotten_rl/lab_hours.py  —  When the lab can be used, from the room calendar.

The lab can be used at any time (day, night, weekends) except around room events. Around each
event of the public calendar the lab is closed from 30 minutes before it starts until 30 minutes
after it ends; if it ends at 21:00 or later, until 22:20. On weekdays the 07:30 reboot is treated
as an event (closed 07:00 to 08:00), because it is not known if it still happens.

`--stop_at lab` means "5 minutes before the lab closes" and `--no_launch_after lab` "1 hour before".
Both are refreshed from the calendar every 30 minutes (`Deadline`), so an event added during a run
is honoured. The calendar is downloaded and cached; if it cannot be downloaded, the cache is used
(on a machine without internet, copy the .ics to the cache path, see `cache_path`). If there is no
calendar at all, it fails: guessing could lose a night of runs.

    python -m rotten_rl.lab_hours            # state, when it closes and reopens
    python -m rotten_rl.lab_hours closes     # one field, "YYYY-MM-DD HH:MM" (for shell scripts)
"""

import datetime
import os
import re
import sys
import time
import urllib.request

CALENDAR_URL = ("https://calendar.google.com/calendar/ical/c_5c48e43e289b1191467cda48d68ca21a0614f6"
                "a404fd7bcaec66b240e4d06d77%40group.calendar.google.com/public/basic.ics")
TZ = "America/Argentina/Buenos_Aires"

MARGIN = datetime.timedelta(minutes=30)
LATE_END = datetime.time(21, 0)          # an event ending from this time on...
LATE_RETURN = datetime.time(22, 20)      # ...reopens at this time
REBOOT = datetime.time(7, 30)            # weekdays, treated as an event
STOP_EARLY = datetime.timedelta(minutes=5)       # --stop_at lab
LAUNCH_EARLY = datetime.timedelta(hours=1)       # --no_launch_after lab
REFRESH = 30 * 60                                # seconds between calendar reads in Deadline
HORIZON = datetime.timedelta(days=60)

Min = datetime.timedelta(minutes=1)


def cache_path():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.environ.get("LAB_CALENDAR", os.path.join(base, "rotten-rl", "lab_calendar.ics"))


def fetch_calendar(timeout=15, fresh=5 * 60):
    """Text of the calendar: the cache if it is less than `fresh` seconds old, else downloaded (and
    cached) or, if that fails, the cache."""
    path = cache_path()
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < fresh:
        with open(path) as f:
            return f.read()
    try:
        with urllib.request.urlopen(CALENDAR_URL, timeout=timeout) as r:
            text = r.read().decode("utf-8")
        if "BEGIN:VCALENDAR" not in text:
            raise ValueError("the answer is not a calendar")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
        return text
    except Exception as e:  # noqa: BLE001 (any failure falls back to the cache)
        if not os.path.exists(path):
            raise RuntimeError(f"cannot download the lab calendar ({e}) and there is no cache at "
                               f"{path}: copy the .ics there") from e
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        print(f"lab_hours: using the cached calendar ({age_h:.1f} h old): {e}", file=sys.stderr)
        with open(path) as f:
            return f.read()


def _to_local(value, params):
    """iCal date or date-time -> naive local datetime."""
    if "VALUE=DATE" in params and "T" not in value:
        return datetime.datetime.strptime(value, "%Y%m%d")
    if value.endswith("Z"):
        utc = datetime.datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc)
        return utc.astimezone().replace(tzinfo=None)
    naive = datetime.datetime.strptime(value, "%Y%m%dT%H%M%S")
    m = re.search(r"TZID=([^;:]+)", params)
    if m:
        from zoneinfo import ZoneInfo
        return naive.replace(tzinfo=ZoneInfo(m.group(1))).astimezone().replace(tzinfo=None)
    return naive


def parse_events(text):
    """[(start, end)] of the calendar's events, as naive local datetimes. Fails on recurring
    events (RRULE): Google exports the instances one by one, and expanding rules is not done."""
    text = re.sub(r"\r?\n[ \t]", "", text)
    events = []
    for block in text.split("BEGIN:VEVENT")[1:]:
        block = block.split("END:VEVENT")[0]
        fields = {}
        for line in block.splitlines():
            m = re.match(r"([A-Z-]+)((?:;[^:]*)?):(.*)", line)
            if m:
                fields[m.group(1)] = (m.group(2), m.group(3).strip())
        if "RRULE" in fields:
            raise ValueError("the calendar has a recurring event (RRULE), not supported")
        if fields.get("STATUS", ("", ""))[1] == "CANCELLED" or "DTSTART" not in fields:
            continue
        start = _to_local(fields["DTSTART"][1], fields["DTSTART"][0])
        if "DTEND" in fields:
            end = _to_local(fields["DTEND"][1], fields["DTEND"][0])
        else:
            all_day = "VALUE=DATE" in fields["DTSTART"][0]
            end = start + (datetime.timedelta(days=1) if all_day else datetime.timedelta(0))
        events.append((start, end))
    return events


def _reopen(end):
    """30 minutes after the end; if it ends at 21:00 or later, not before 22:20 either."""
    if end.time() >= LATE_END:
        return max(end + MARGIN, datetime.datetime.combine(end.date(), LATE_RETURN))
    return end + MARGIN


def closed_spans(events, now, horizon=HORIZON):
    """Sorted, merged [(closes, reopens)] from now until now + horizon: events with their margins
    plus the weekday reboots."""
    end_of_horizon = now + horizon
    events = list(events)
    day = now.date() - datetime.timedelta(days=1)
    while day <= end_of_horizon.date():
        if day.weekday() < 5:
            t = datetime.datetime.combine(day, REBOOT)
            events.append((t, t))
        day += datetime.timedelta(days=1)
    spans = sorted((s - MARGIN, _reopen(e)) for s, e in events)
    merged = []
    for closes, reopens in spans:
        if reopens <= now or closes > end_of_horizon:
            continue
        if merged and closes <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], reopens))
        else:
            merged.append((closes, reopens))
    return merged


def window(events, now=None):
    """(is_open, closes, reopens). Open: closes = when the current window ends, reopens = the end of
    that closed span. Closed: closes = None, reopens = when it opens again."""
    now = now or datetime.datetime.now()
    spans = closed_spans(events, now)
    if not spans:          # cannot happen with the weekday reboots, but be safe
        return True, now + HORIZON, None
    closes, reopens = spans[0]
    if closes <= now:
        return False, None, reopens
    return True, closes, reopens


def next_time(spec, early=STOP_EARLY, now=None, events=None):
    """Next instant of `spec`: "HH:MM" (today if not past yet, else tomorrow), "YYYY-MM-DD HH:MM",
    or "lab" (`early` before the lab closes; if it is closed now, `now`)."""
    now = now or datetime.datetime.now()
    if spec == "lab":
        if events is None:
            events = parse_events(fetch_calendar())
        is_open, closes, _ = window(events, now)
        return closes - early if is_open else now
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}", spec):
        return datetime.datetime.strptime(spec, "%Y-%m-%d %H:%M")
    hh, mm = (int(x) for x in spec.split(":"))
    when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if when <= now:
        when += datetime.timedelta(days=1)
    return when


class Deadline:
    """A `next_time` that, for "lab", reads the calendar again every `refresh` seconds (if it cannot,
    it keeps the last value and says so). `changed` is True after a refresh that moved it."""

    def __init__(self, spec, early=STOP_EARLY, refresh=REFRESH):
        self.spec, self.early, self.refresh = spec, early, refresh
        self.when = next_time(spec, early)
        self._read_at = time.monotonic()
        self.changed = False

    def reached(self, now=None):
        now = now or datetime.datetime.now()
        self.changed = False
        if self.spec == "lab" and time.monotonic() - self._read_at >= self.refresh:
            self._read_at = time.monotonic()
            try:
                when = next_time("lab", self.early, now=now)
            except Exception as e:  # noqa: BLE001
                print(f"lab_hours: calendar not refreshed, keeping {self.when:%Y-%m-%d %H:%M}: {e}",
                      file=sys.stderr)
            else:
                # If it is closed now, next_time gives `now`, so it stops right away.
                self.changed = when != self.when
                self.when = when
        return now >= self.when


def main(argv):
    now = datetime.datetime.now()
    is_open, closes, reopens = window(parse_events(fetch_calendar()), now)
    fmt = (lambda t: t.strftime("%Y-%m-%d %H:%M") if t else "-")
    fields = {
        "state": "open" if is_open else "closed",
        "closes": fmt(closes),
        "stop": fmt(closes - STOP_EARLY if closes else None),
        "last_launch": fmt(closes - LAUNCH_EARLY if closes else None),
        "reopens": fmt(reopens),
    }
    if argv:
        print(fields[argv[0]])
    else:
        for k, v in fields.items():
            print(f"{k:12s} {v}")


if __name__ == "__main__":
    main(sys.argv[1:])
