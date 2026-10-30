import datetime

import pytest

from rotten_rl.lab_hours import LAB_LAST_LAUNCH, next_time

T = datetime.datetime


@pytest.mark.parametrize("now, expected", [
    (T(2026, 10, 29, 23, 0), T(2026, 10, 30, 6, 55)),   # jueves a la noche: viernes
    (T(2026, 10, 30, 1, 0), T(2026, 10, 30, 6, 55)),    # viernes de madrugada: ese día
    (T(2026, 10, 30, 22, 10), T(2026, 11, 2, 6, 55)),   # viernes a la noche: lunes
    (T(2026, 10, 31, 3, 0), T(2026, 11, 2, 6, 55)),     # sábado
    (T(2026, 11, 1, 23, 0), T(2026, 11, 2, 6, 55)),     # domingo
    (T(2026, 11, 2, 6, 54), T(2026, 11, 2, 6, 55)),     # lunes antes del corte
    (T(2026, 11, 2, 22, 0), T(2026, 11, 3, 6, 55)),     # lunes a la noche: martes
])
def test_lab_corta_la_proxima_manana_habil(now, expected):
    assert next_time("lab", now=now) == expected


def test_lab_ultimo_lanzamiento():
    assert next_time("lab", LAB_LAST_LAUNCH, now=T(2026, 10, 30, 22, 10)) == T(2026, 11, 2, 6, 0)


def test_hora_fija_no_mira_el_dia():
    assert next_time("06:55", now=T(2026, 10, 30, 22, 10)) == T(2026, 10, 31, 6, 55)
    assert next_time("14:00", now=T(2026, 10, 31, 9, 0)) == T(2026, 10, 31, 14, 0)
