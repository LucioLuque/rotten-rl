"""
rotten_rl/lab_hours.py  —  Cuándo cortar según el horario del laboratorio.

De lunes a viernes el acceso termina a las 07:00 y a las 07:30 se reinician las máquinas. Sábados y
domingos no hay corte: de viernes 22:00 a lunes 07:00 se puede correr de seguido. Por eso `--stop_at
lab` significa "la próxima mañana de día hábil": el viernes a la noche es el lunes, no el sábado.
Un feriado no se detecta: se corta igual, que es lo seguro.
"""

import datetime

LAB_STOP = "06:55"
LAB_LAST_LAUNCH = "06:00"


def next_time(spec, lab_time=LAB_STOP, now=None):
    """Próximo instante de `spec`: "HH:MM" (hoy si todavía no pasó; si no, mañana) o "lab" (la
    hora `lab_time` de la próxima mañana de lunes a viernes)."""
    now = now or datetime.datetime.now()
    hh, mm = (int(x) for x in (lab_time if spec == "lab" else spec).split(":"))
    when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if when <= now:
        when += datetime.timedelta(days=1)
    while spec == "lab" and when.weekday() >= 5:
        when += datetime.timedelta(days=1)
    return when
