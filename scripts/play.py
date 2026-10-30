"""
scripts/play.py

Partida jugable de La Podrida con Pygame, contra cualquier rival.

Sos el jugador de abajo. El rival puede ser la heurística o un modelo entrenado.
Con --perfil se guarda cada decisión tuya en data/perfiles/<perfil>/, para armar
ese perfil después (docs/perfiles/). Cada partida usa un reparto nuevo y vas
cambiando de asiento; al guardar no se puede ver las manos ni usar el autoplay.

Instalación:
    pip install -e ".[play]"   # desde la raiz del repo

Ejecución:
    python scripts/play.py                                     # practicar contra la heurística
    python scripts/play.py --perfil lucio                      # guardando tus decisiones
    python scripts/play.py --perfil lucio --rival runs/lab/.../<corrida>   # contra un modelo
    python scripts/play.py --perfil lucio --situaciones 150    # jugadas elegidas, una por vez

Controles:
    CLICK en un botón de canto  -> cantás esa cantidad
    CLICK en una carta          -> la jugás (solo las legales responden)
    CLICK en "Siguiente ronda"  -> arranca la ronda siguiente
    ESPACIO                     -> avanza (siguiente ronda / destraba la pausa)
    F11 o F                     -> pantalla completa / volver a ventana
    A                           -> autoplay: un bot juega por vos (sin --perfil)
    V                           -> mostrar/ocultar las manos de los rivales (sin --perfil)
    R                           -> partida nueva, con el reparto siguiente
    ESC                         -> salir

La ventana se puede redimensionar arrastrando el borde o maximizando: todo el
layout se recalcula solo.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import pyspiel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
import rotten_rl.openspiel_game  # noqa: E402,F401  registra el juego "podrida"
from rotten_rl.engine import Card  # noqa: E402
from rotten_rl.evaluation import heuristic_opponent  # noqa: E402
from rotten_rl.perfiles.partida import HumanMatch, SessionLog, SpotSession, next_deal  # noqa: E402
from rotten_rl.perfiles.situaciones import pick_spots  # noqa: E402
from rotten_rl.renderer import PodridaPygameRenderer, RoundRecord  # noqa: E402
from tournament import make_agent  # noqa: E402

PLAYER_NAMES = ["Vos", "Ana", "Beto", "Caro", "Dani", "Eze", "Fito"]

BOT_DELAY = 0.65       # segundos entre jugadas de bots
TRICK_PAUSE = 1.3      # segundos que queda visible la baza completa
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "perfiles")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

class PodridaApp:
    def __init__(self, args) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.font.init()

        # Abre ocupando casi todo el monitor. Se puede redimensionar y pasar a
        # pantalla completa con F11: el layout se recalcula con cada cambio.
        try:
            dw, dh = pygame.display.get_desktop_sizes()[0]
        except Exception:
            info = pygame.display.Info()
            dw, dh = info.current_w, info.current_h

        self.renderer = PodridaPygameRenderer(
            render_mode=args.render_mode,
            width=int(dw * 0.92),
            height=int(dh * 0.88),
            resizable=True,
        )
        self.pyspiel_game = pyspiel.load_game("podrida", {"players": args.jugadores})
        rng = np.random.RandomState()
        self.rival_name, self.rival = make_agent(args.rival, self.pyspiel_game, "cpu", rng)
        self.helper = heuristic_opponent(rng)
        self.reparto_seed = args.reparto_seed
        self.saving = args.perfil is not None
        folder = os.path.join(DATA_DIR, args.perfil or "")
        self.log = SessionLog(folder, args.perfil, self.rival_name) if self.saving else None
        self.reparto = next_deal(folder, self.reparto_seed) if self.saving else 0

        self.autoplay = False
        self.reveal_all = False
        self.mouse_pos = (0, 0)
        self.running = True
        self.spots = None
        if args.situaciones:
            self.saving = True
            self.spots = SpotSession(self.pyspiel_game,
                                     pick_spots(self.pyspiel_game, args.situaciones, args.reparto_seed),
                                     self.log)
        self.new_match()

    def new_match(self) -> None:
        if self.spots is not None:
            return self.show_spot()
        n = self.pyspiel_game.num_players()
        self.seat = self.reparto % n         # vas rotando de asiento
        self.match = HumanMatch(self.pyspiel_game, self.rival, self.seat, self.reparto_seed,
                                self.reparto, self.log)
        self.reparto += 1
        others = iter(PLAYER_NAMES[1:])
        for i, player in enumerate(self.game.players):
            player.name = PLAYER_NAMES[0] if i == self.seat else next(others)
        self.history: List[RoundRecord] = []
        self.finished = False
        self.round_recorded = False
        self.freeze_until = 0.0
        self.next_bot_time = time.time() + BOT_DELAY
        self.message = f"Ronda 1: {self.game.cards_per_player} cartas. A cantar."

    def show_spot(self) -> None:
        """Muestra la situación que toca. Sin rival: al contestar pasa directo a la siguiente."""
        spots = self.spots
        self.history = []
        self.freeze_until = 0.0
        self.round_recorded = False
        self.finished = spots.finished
        if spots.finished:
            self.message = spots.finished_message
            self.running = False
            return
        self.match = spots
        self.seat = spots.seat
        others = iter(PLAYER_NAMES[1:])
        for i, player in enumerate(self.game.players):
            player.name = PLAYER_NAMES[0] if i == self.seat else next(others)
        played = [c for trick in self.game.completed_tricks for _, c in trick]
        seen = " ".join(self.card_label(c) for c in played)
        self.message = f"Situación {spots.index + 1} de {len(spots.spots)}" + (
            f". Ya jugadas: {seen}" if seen else "")

    # ------------------------------------------------------------------

    @property
    def game(self):
        return self.match.engine

    @property
    def frozen(self) -> bool:
        return time.time() < self.freeze_until

    def card_label(self, card: Card) -> str:
        return f"{card.rank}{self.renderer.SUIT_SYMBOLS.get(card.suit, '')}"

    # ------------------------------------------------------------------

    def run(self) -> None:
        pygame = self.pygame
        while self.running and not self.renderer.closed:
            for event in pygame.event.get():
                self.handle_event(event)

            self.step_bots()
            self.draw()

        self.renderer.close()

    # ------------------------------------------------------------------

    def handle_event(self, event) -> None:
        pygame = self.pygame

        if event.type == pygame.QUIT:
            self.running = False
            return

        if event.type == pygame.MOUSEMOTION:
            self.mouse_pos = event.pos
            return

        if event.type == pygame.VIDEORESIZE:
            self.renderer.set_size(event.w, event.h)
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.handle_click(event.pos)
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.renderer.fullscreen:
                    self.renderer.toggle_fullscreen()
                else:
                    self.running = False
            elif event.key in (pygame.K_F11, pygame.K_f):
                self.renderer.toggle_fullscreen()
            elif event.key == pygame.K_v and not self.saving:
                self.reveal_all = not self.reveal_all
                self.message = "Manos rivales visibles" if self.reveal_all else "Manos rivales ocultas"
            elif event.key == pygame.K_a and not self.saving:
                self.autoplay = not self.autoplay
                self.message = "Autoplay activado: juega un bot por vos" if self.autoplay else "Autoplay pausado"
            elif event.key == pygame.K_r:
                self.new_match()
                self.message = "Partida nueva. " + self.message
            elif event.key == pygame.K_SPACE:
                self.freeze_until = 0.0
                if self.game.phase == "done":
                    self.advance_round()

    def handle_click(self, pos) -> None:
        game = self.game

        if game.phase == "done" or self.finished:
            if self.renderer.button_at(pos) == "next_round":
                self.advance_round()
            return

        if self.frozen:
            self.freeze_until = 0.0
            return

        if not self.match.human_turn:
            return

        if game.phase == "bidding":
            bid = self.renderer.bid_at(pos)
            if bid is not None:
                self.apply(bid + 52)
            return

        if game.phase == "playing":
            card = self.renderer.card_at(pos)
            if card is not None:
                self.apply(game.card_to_id(card))

    # ------------------------------------------------------------------

    def apply(self, action_id: int) -> None:
        game = self.game
        idx = game.current_player_index
        name = game.players[idx].name
        tricks_before = len(game.completed_tricks)

        if not self.match.apply(action_id):
            self.message = "Movimiento inválido"
            return
        if self.spots is not None:
            return self.show_spot()
        self.after_move(action_id, name, tricks_before)

    def after_move(self, action_id: int, name: str, tricks_before: int) -> None:
        game = self.game
        if action_id >= 52:
            self.message = f"{name} cantó {action_id - 52}"
        else:
            self.message = f"{name} jugó {self.card_label(game.id_to_card(action_id))}"

        if len(game.completed_tricks) > tricks_before:
            last = game.completed_tricks[-1]
            winner = game.players[game.resolve_trick(list(last))].name
            self.message = f"Baza para {winner}"
            self.freeze_until = time.time() + TRICK_PAUSE

        if self.match.round_over and not self.round_recorded:
            self.record_round()
            self.round_recorded = True
            hit = [p.name for p in game.players if p.bid == p.tricks_won]
            self.message = (
                f"Ronda {len(self.history)} terminada. "
                f"Cumplieron: {', '.join(hit) if hit else 'nadie'}. {self.standings()}"
            )

        self.next_bot_time = time.time() + BOT_DELAY

    def record_round(self) -> None:
        game = self.game
        self.history.append(
            RoundRecord(
                round_number=len(self.history) + 1,
                cards_per_player=game.cards_per_player,
                bids=[p.bid or 0 for p in game.players],
                tricks=[p.tricks_won for p in game.players],
                points=[p.last_round_score for p in game.players],
                totals=[p.score for p in game.players],
            )
        )

    def standings(self) -> str:
        ranking = sorted(self.game.players, key=lambda p: -p.score)
        return " · ".join(f"{p.name} {p.score}" for p in ranking)

    def advance_round(self) -> None:
        if self.match.finished:
            self.finished = True
            winner = max(self.game.players, key=lambda p: p.score)
            self.message = f"Partida terminada. Ganó {winner.name} con {winner.score} puntos. (R para jugar otra)"
            return
        self.match.next_round()
        self.round_recorded = False
        self.freeze_until = 0.0
        self.next_bot_time = time.time() + BOT_DELAY
        self.message = f"Ronda {len(self.history) + 1}: {self.game.cards_per_player} cartas. A cantar."

    def step_bots(self) -> None:
        match = self.match
        if self.spots is not None:
            return
        if match.round_over or self.frozen or time.time() < self.next_bot_time:
            return
        if match.human_turn and not self.autoplay:
            return

        game = self.game
        name = game.players[game.current_player_index].name
        tricks_before = len(game.completed_tricks)
        state = match.state
        act = self.helper if match.human_turn else self.rival
        action = act(state, state.legal_actions())
        match.apply(action)
        self.after_move(action, name, tricks_before)

    # ------------------------------------------------------------------

    def draw(self):
        game = self.game
        human_turn = self.match.human_turn and not self.autoplay and not self.frozen
        return self.renderer.render(
            game,
            perspective_player=self.seat,
            reveal_all=self.reveal_all,
            message=self.message,
            history=self.history,
            round_number=(self.match.state._round_ptr + 1 if self.spots is not None
                          else len(self.history) + (0 if self.match.round_over else 1)),
            total_rounds=len(self.pyspiel_game.ladder),
            show_bid_buttons=human_turn and game.phase == "bidding",
            freeze_last_trick=self.frozen or game.phase == "done",
            next_round_button=game.phase == "done" and not self.finished,
            mouse_pos=self.mouse_pos,
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Jugar a La Podrida contra un rival.")
    p.add_argument("--jugadores", type=int, default=2)
    p.add_argument("--rival", default="heuristic",
                   help="heuristic, random o una corrida (<corrida> o <corrida>@<ep>)")
    p.add_argument("--perfil", default=None, help="guardar tus decisiones para este perfil")
    p.add_argument("--reparto_seed", type=int, default=0, help="serie de repartos a usar")
    p.add_argument("--situaciones", type=int, default=0,
                   help="en vez de partidas, esta cantidad de jugadas elegidas (requiere --perfil)")
    p.add_argument("--render_mode", default="human", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.situaciones and not args.perfil:
        p.error("--situaciones necesita --perfil, para guardar las respuestas")
    return args


def main() -> None:
    try:
        import pygame  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Pygame no está instalado. Instalalo con: pip install pygame") from exc

    PodridaApp(parse_args()).run()


if __name__ == "__main__":
    main()
