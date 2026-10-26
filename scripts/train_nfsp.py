"""
scripts/train_nfsp.py  —  Alias de scripts/train.py (que por defecto usa --algo nfsp).

Queda para retomar con --resume las corridas lanzadas antes de que existiera
scripts/train.py, y para los comandos y colas viejos. Ver scripts/train.py.
"""

import os
import sys

from absl import app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

if __name__ == "__main__":
    app.run(train.main)
