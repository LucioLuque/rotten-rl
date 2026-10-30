"""
scripts/play.py

Playable match of La Podrida with Pygame, against any rival.

You are the bottom player. The rival can be the heuristic, a profile or a trained model. With
--profile each of your decisions is saved to data/profiles/<profile>/, to build that profile later
(docs/profiles/). Each match uses a new deal and you rotate seats; while saving, you cannot see the
other hands or use autoplay.

Install:
    pip install -e ".[play]"   # from the repo root

Run:
    python scripts/play.py                                      # practice against the heuristic
    python scripts/play.py --profile lucio --spots 150          # chosen plays, one at a time
    python scripts/play.py --profile lucio                      # whole matches, saving decisions
    python scripts/play.py --profile lucio --rival lucio        # against the learned profile
    python scripts/play.py --profile lucio --rival runs/lab/.../<run>   # against a model

Controls:
    CLICK a bid button          -> bid that amount
    CLICK a card                -> play it (only legal cards respond)
    CLICK "Siguiente ronda"     -> start the next round
    SPACE                       -> advance (next round / skip the pause)
    F11 or F                    -> full screen / back to window
    A                           -> autoplay: a bot plays for you (without --profile)
    V                           -> show/hide rival hands (without --profile)
    R                           -> new match, with the next deal
    ESC                         -> quit

The window can be resized: the whole layout is recomputed.
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
from rotten_rl.profiles.match import HumanMatch, SessionLog, SpotSession, next_deal  # noqa: E402
from rotten_rl.profiles.spots import answered, pick_spots  # noqa: E402
from rotten_rl.renderer import PodridaPygameRenderer, RoundRecord  # noqa: E402
from tournament import make_agent  # noqa: E402

PLAYER_NAMES = ["Vos", "Ana", "Beto", "Caro", "Dani", "Eze", "Fito"]

BOT_DELAY = 0.65       # seconds between bot plays
TRICK_PAUSE = 1.3      # seconds the full trick stays visible
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

class PodridaApp:
    def __init__(self, args) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.font.init()

        # Opens filling most of the monitor. It can be resized and go full screen
        # with F11: the layout is recomputed on each change.
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
        self.pyspiel_game = pyspiel.load_game("podrida", {"players": args.players})
        rng = np.random.RandomState()
        self.rival_name, self.rival = make_agent(args.rival, self.pyspiel_game, "cpu", rng)
        self.helper = heuristic_opponent(rng)
        self.deal_seed = args.deal_seed
        self.saving = args.profile is not None
        folder = os.path.join(DATA_DIR, args.profile or "")
        self.log = SessionLog(folder, args.profile, self.rival_name) if self.saving else None
        self.deal = next_deal(folder, self.deal_seed) if self.saving else 0

        self.autoplay = False
        self.reveal_all = False
        self.mouse_pos = (0, 0)
        self.running = True
        self.spots = None
        self.status = f"contra {self.rival_name}" + (f" · guardando en {args.profile}" if self.saving else "")
        if args.spots:
            self.saving = True
            self.status = f"guardando en {args.profile}"
            self.renderer.splash("Eligiendo las situaciones más importantes…")
            picked = pick_spots(self.pyspiel_game, args.spots, args.deal_seed,
                                skip=answered(folder))
            self.spots = SpotSession(self.pyspiel_game, picked, self.log)
        self.new_match()

    def new_match(self) -> None:
        if self.spots is not None:
            return self.show_spot()
        n = self.pyspiel_game.num_players()
        self.seat = self.deal % n         # seats rotate
        self.match = HumanMatch(self.pyspiel_game, self.rival, self.seat, self.deal_seed,
                                self.deal, self.log)
        self.deal += 1
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
        """Shows the current spot. No rival: answering goes straight to the next one."""
        spots = self.spots
        self.history = []
        self.freeze_until = 0.0
        self.round_recorded = False
        self.finished = spots.finished
        if spots.finished:
            self.message = spots.finished_message
            return
        self.match = spots
        self.seat = spots.seat
        others = iter(PLAYER_NAMES[1:])
        for i, player in enumerate(self.game.players):
            player.name = PLAYER_NAMES[0] if i == self.seat else next(others)
        self.message = "Cantá." if self.game.phase == "bidding" else "Elegí una carta."

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

        button = self.renderer.button_at(pos)
        if button == "banner":
            if self.spots is not None:
                self.running = False
            else:
                self.new_match()
            return
        if game.phase == "done" or self.finished:
            if button == "next_round":
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
            self.message = f"Partida terminada. Ganó {winner.name} con {winner.score} puntos."
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

    def banner(self):
        if self.spots is not None:
            if not self.spots.finished:
                return None
            return ("Listo", f"Respondiste {len(self.spots.spots)} situaciones. Ya quedaron guardadas.",
                    "Cerrar")
        if not self.finished:
            return None
        ranking = sorted(self.game.players, key=lambda p: -p.score)
        won = ranking[0].name == PLAYER_NAMES[0]
        title = "¡Ganaste!" if won else f"Ganó {ranking[0].name}"
        return title, " · ".join(f"{p.name} {p.score}" for p in ranking), "Otra partida"

    def draw(self):
        game = self.game
        human_turn = self.match.human_turn and not self.autoplay and not self.frozen
        spot = self.spots is not None
        over = self.match.round_over and not spot
        return self.renderer.render(
            game,
            perspective_player=self.seat,
            reveal_all=self.reveal_all,
            message=self.message,
            history=self.history,
            round_number=(self.match.state._round_ptr + 1 if spot
                          else len(self.history) + (0 if self.match.round_over else 1)),
            total_rounds=len(self.pyspiel_game.ladder),
            show_bid_buttons=human_turn and game.phase == "bidding",
            freeze_last_trick=self.frozen or game.phase == "done",
            next_round_button=over and not self.finished,
            mouse_pos=self.mouse_pos,
            next_label="Ver resultado  →" if self.match.finished else "Siguiente ronda  →",
            spot_progress=(f"{min(self.spots.index + 1, len(self.spots.spots))} / {len(self.spots.spots)}"
                           if spot else None),
            played_cards=([c for trick in game.completed_tricks for _, c in trick] if spot else None),
            status=self.status,
            banner=self.banner(),
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Play La Podrida against a rival.")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--rival", default="heuristic",
                   help="heuristic, random, a profile or a run (<run> or <run>@<ep>)")
    p.add_argument("--profile", default=None, help="save your decisions for this profile")
    p.add_argument("--deal_seed", type=int, default=0, help="series of deals to use")
    p.add_argument("--spots", type=int, default=0,
                   help="instead of matches, this many chosen plays (needs --profile)")
    p.add_argument("--render_mode", default="human", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.spots and not args.profile:
        p.error("--spots needs --profile, to save the answers")
    return args


def main() -> None:
    try:
        import pygame  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Pygame no está instalado. Instalalo con: pip install pygame") from exc

    PodridaApp(parse_args()).run()


if __name__ == "__main__":
    main()
