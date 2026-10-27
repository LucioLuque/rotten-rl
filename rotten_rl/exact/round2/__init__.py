"""La ronda de 2 cartas, resuelta con MCCFR en C++ (ver README.md de esta
carpeta). `binary()` compila el programa si hace falta (necesita g++ y make)."""

import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))


def binary() -> str:
    path = os.path.join(_DIR, "build", "round2")
    src = os.path.join(_DIR, "round2.cpp")
    if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(src):
        subprocess.run(["make", "-C", _DIR], check=True, capture_output=True)
    return path


def run(*args) -> str:
    """Corre `round2 <args>` y devuelve lo que imprime."""
    return subprocess.run([binary(), *map(str, args)], check=True, capture_output=True,
                          text=True).stdout
