"""
rotten_rl/run_logger.py

Manejo de una corrida de entrenamiento:
  - Nombre unico de experimento (para que NUNCA se pisen los resultados).
  - TensorBoard   -> curvas en vivo.
  - CSV           -> mismas metricas, para re-analizar con pandas offline.
  - config.json   -> snapshot de todos los hiperparametros (reproducibilidad).

Estructura en disco:
    runs/<run_name>/
        tb/                 event files de TensorBoard
        config.json         hiperparametros de la corrida
        train.csv           progreso de entrenamiento (reward movil, loss)
        eval_<opp>_overall.csv  una fila por evaluacion (metricas globales)
        eval_<opp>_by_h.csv     una fila por (evaluacion, h)
        eval_<opp>_by_seat.csv  win_rate por asiento, una fila por evaluacion
        checkpoints/            snapshots de las avg-nets (los escribe el script)
    con <opp> = random | heuristic | past (un juego de archivos por oponente).

Para ver las curvas:
    tensorboard --logdir runs
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(
        self,
        base_dir: str = "runs",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        # Timestamp SIEMPRE presente -> dos corridas nunca comparten carpeta,
        # aunque pases el mismo --run_name.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = run_name.strip() if run_name else "podrida"
        name = f"{base}_{ts}"
        self.run_dir = os.path.join(base_dir, name)
        os.makedirs(os.path.join(self.run_dir, "tb"), exist_ok=True)
        self.name = name

        self.writer = SummaryWriter(os.path.join(self.run_dir, "tb"))

        if config is not None:
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
            # tambien registramos los hparams en TensorBoard (pestania HParams)
            flat = {k: v for k, v in config.items()
                    if isinstance(v, (int, float, str, bool))}
            self._hparams = flat
        else:
            self._hparams = {}

        self._csv_writers: Dict[str, csv.DictWriter] = {}
        self._csv_files = {}

    # ------------------------------------------------------------------

    def _csv_row(self, fname: str, row: Dict) -> None:
        if fname not in self._csv_writers:
            path = os.path.join(self.run_dir, fname)
            f = open(path, "w", newline="")
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            self._csv_files[fname] = f
            self._csv_writers[fname] = w
        self._csv_writers[fname].writerow(row)
        self._csv_files[fname].flush()

    # ------------------------------------------------------------------

    def log_train(self, episode: int, reward_ma: float, losses) -> None:
        """Progreso continuo de entrenamiento (estilo SB3).
        `losses` es una lista con un elemento por agente; cada elemento es la
        tupla (sl_loss, rl_loss) que devuelve NFSP.loss."""
        self.writer.add_scalar("train/reward_moving_avg", reward_ma, episode)
        row = {"episode": episode, "reward_ma": reward_ma}
        for i, comp in enumerate(losses):
            if isinstance(comp, (tuple, list)):
                sl, rl = (list(comp) + [None, None])[:2]
            else:
                sl, rl = None, comp
            for tag, val in (("sl", sl), ("rl", rl)):
                v = float(val) if val is not None else float("nan")
                self.writer.add_scalar(f"train/loss_agent_{i}_{tag}", v, episode)
                row[f"loss_agent_{i}_{tag}"] = v
        self._csv_row("train.csv", row)

    def log_eval(self, episode: int, metrics: Dict, opponent: str = "random") -> None:
        """Metricas de evaluacion contra `opponent`: overall, por-h, por
        asiento y calibracion. Cada oponente tiene sus propias curvas
        (eval_<opp>/...) y sus propios CSV (eval_<opp>_overall.csv, ...)."""
        o = metrics["overall"]
        for k, v in o.items():
            self.writer.add_scalar(f"eval_{opponent}/{k}", v, episode)
        self._csv_row(f"eval_{opponent}_overall.csv", {"episode": episode, **o})

        # por tamanio de ronda h
        for h, m in metrics["by_h"].items():
            for k, v in m.items():
                self.writer.add_scalar(f"eval_{opponent}_by_h/{k}/h{h}", v, episode)
            self._csv_row(f"eval_{opponent}_by_h.csv", {"episode": episode, "h": h, **m})

        # por asiento
        by_seat = metrics.get("win_rate_by_seat", {})
        for s, v in by_seat.items():
            self.writer.add_scalar(f"eval_{opponent}_by_seat/win_rate/seat{s}", v, episode)
        if by_seat:
            self._csv_row(f"eval_{opponent}_by_seat.csv",
                          {"episode": episode, **{f"seat{s}": v for s, v in by_seat.items()}})

        # calibracion E[tricks | bid=b]
        for b, v in metrics["calibration"].items():
            self.writer.add_scalar(f"calibration_{opponent}/bid_{b}", v, episode)

        self.writer.flush()

    # ------------------------------------------------------------------

    def close(self, final_metrics: Optional[Dict] = None) -> None:
        # Guarda hparams + metrica final en la pestania HParams de TensorBoard.
        if self._hparams and final_metrics is not None:
            flat_metrics = {f"hparam/{k}": float(v)
                            for k, v in final_metrics["overall"].items()
                            if isinstance(v, (int, float))}
            try:
                self.writer.add_hparams(self._hparams, flat_metrics)
            except Exception:
                pass
        self.writer.close()
        for f in self._csv_files.values():
            f.close()
