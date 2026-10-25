"""
demo_visual.py

Partida jugable de La Podrida con Pygame.

Sos el jugador de abajo. Los demás son bots con una heurística simple.

Instalación:
    pip install pygame

Ejecución:
    python demo_visual.py

Controles:
    CLICK en un botón de canto  -> cantás esa cantidad
    CLICK en una carta          -> la jugás (solo las legales responden)
    CLICK en "Siguiente ronda"  -> arranca la ronda siguiente
    ESPACIO                     -> avanza (siguiente ronda / destraba la pausa)
    F11 o F                     -> pantalla completa / volver a ventana
    A                           -> autoplay: un bot juega por vos
    V                           -> mostrar/ocultar las manos de los rivales
    R                           -> nueva partida
    ESC                         -> salir

La ventana se puede redimensionar arrastrando el borde o maximizando: todo el
layout se recalcula solo.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

from podrida import MAX_HAND_SIZE, Card, InvalidMoveError, PodridaGame, build_round_ladder
from visual_renderer import PodridaPygameRenderer, RoundRecord


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

NUM_PLAYERS = 7        # probado de 2 a 10; con 7 entra bien en pantalla completa
HUMAN_INDEX = 0
MAX_CARDS = MAX_HAND_SIZE   # escalera 1..7..1 = 13 rondas (la partida completa)
MUST_TRUMP_WHEN_VOID = True   # obligación de fallar: sin palo de salida, va triunfo sí o sí
PLAYER_NAMES = ["Vos", "Ana", "Beto", "Caro", "Dani", "Eze", "Fito", "Gus", "Hugo", "Ivo"]

BOT_DELAY = 0.65       # segundos entre jugadas de bots
TRICK_PAUSE = 1.3      # segundos que queda visible la baza completa


# ----------------------------------------------------------------------
# Bots
# ----------------------------------------------------------------------

def _beats(candidate: Card, best: Card, lead_suit: str, trump: Optional[str]) -> bool:
    if trump is not None and candidate.suit == trump and best.suit != trump:
        return True
    if best.suit == trump and candidate.suit != trump:
        return False
    if candidate.suit == best.suit:
        return candidate.value > best.value
    return False


def _current_best(game: PodridaGame) -> Optional[Card]:
    if not game.current_trick:
        return None
    lead_suit = game.current_trick[0][1].suit
    best = game.current_trick[0][1]
    for _, card in game.current_trick[1:]:
        if _beats(card, best, lead_suit, game.trump_suit):
            best = card
    return best


def bot_bid(game: PodridaGame, idx: int) -> int:
    """Estimación grosera de bazas ganables."""
    legal = game.legal_bids(idx)
    hand = game.players[idx].hand
    trump = game.trump_suit

    estimate = 0.0
    for card in hand:
        if card.suit == trump:
            estimate += 0.85 if card.value >= 11 else 0.45
        elif card.rank == "A":
            estimate += 0.80
        elif card.rank == "K":
            estimate += 0.50
        elif card.rank == "Q":
            estimate += 0.25

    target = int(round(estimate))
    return min(legal, key=lambda b: (abs(b - target), b))


def bot_play(game: PodridaGame, idx: int) -> Card:
    legal = game.legal_cards(idx)
    player = game.players[idx]
    wants_trick = player.tricks_won < (player.bid or 0)
    trump = game.trump_suit

    if not game.current_trick:
        return max(legal, key=lambda c: c.value) if wants_trick else min(legal, key=lambda c: c.value)

    best = _current_best(game)
    lead_suit = game.current_trick[0][1].suit
    winners = [c for c in legal if best is not None and _beats(c, best, lead_suit, trump)]

    if wants_trick and winners:
        return min(winners, key=lambda c: (c.suit == trump, c.value))

    losers = [c for c in legal if c not in winners]
    pool = losers or legal
    return min(pool, key=lambda c: (c.suit == trump, c.value))


# ----------------------------------------------------------------------
# Partida completa (escalera de rondas + historial)
# ----------------------------------------------------------------------

class Match:
    def __init__(
        self,
        num_players: int = NUM_PLAYERS,
        max_cards: Optional[int] = MAX_CARDS,
        seed: Optional[int] = None,
    ) -> None:
        names = PLAYER_NAMES[:num_players]
        self.game = PodridaGame(
            num_players=num_players,
            player_names=names,
            seed=seed,
            must_trump_when_void=MUST_TRUMP_WHEN_VOID,
        )
        self.ladder = build_round_ladder(num_players, max_cards or MAX_HAND_SIZE)
        self.round_index = -1
        self.history: List[RoundRecord] = []
        self.finished = False

    @property
    def round_number(self) -> int:
        return self.round_index + 1

    @property
    def total_rounds(self) -> int:
        return len(self.ladder)

    def start_next_round(self) -> None:
        if self.round_index + 1 >= len(self.ladder):
            self.finished = True
            return
        self.round_index += 1
        self.game.start_round(num_cards=self.ladder[self.round_index])

    def record_round(self) -> None:
        game = self.game
        self.history.append(
            RoundRecord(
                round_number=self.round_number,
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


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

class PodridaApp:
    def __init__(self) -> None:
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
            render_mode="human",
            width=int(dw * 0.92),
            height=int(dh * 0.88),
            resizable=True,
        )
        self.match = Match()
        self.match.start_next_round()

        self.message = "Ronda 1. Cantá cuántas bazas vas a hacer."
        self.autoplay = False
        self.reveal_all = False
        self.freeze_until = 0.0
        self.next_bot_time = time.time() + BOT_DELAY
        self.round_recorded = False
        self.mouse_pos = (0, 0)
        self.running = True

    # ------------------------------------------------------------------

    @property
    def game(self) -> PodridaGame:
        return self.match.game

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
            elif event.key == pygame.K_v:
                self.reveal_all = not self.reveal_all
                self.message = "Manos rivales visibles" if self.reveal_all else "Manos rivales ocultas"
            elif event.key == pygame.K_a:
                self.autoplay = not self.autoplay
                self.message = "Autoplay activado: juega un bot por vos" if self.autoplay else "Autoplay pausado"
            elif event.key == pygame.K_r:
                self.match = Match()
                self.match.start_next_round()
                self.round_recorded = False
                self.freeze_until = 0.0
                self.message = "Partida nueva."
            elif event.key == pygame.K_SPACE:
                self.freeze_until = 0.0
                if self.game.phase == "done":
                    self.advance_round()

    def handle_click(self, pos) -> None:
        game = self.game

        if game.phase == "done" or self.match.finished:
            if self.renderer.button_at(pos) == "next_round":
                self.advance_round()
            return

        if self.frozen:
            self.freeze_until = 0.0
            return

        if game.current_player_index != HUMAN_INDEX:
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

        try:
            game.apply_action(action_id, idx)
        except InvalidMoveError as exc:
            self.message = f"Movimiento inválido: {exc}"
            return

        if action_id >= 52:
            self.message = f"{name} cantó {action_id - 52}"
        else:
            self.message = f"{name} jugó {self.card_label(game.id_to_card(action_id))}"

        if len(game.completed_tricks) > tricks_before:
            last = game.completed_tricks[-1]
            winner = game.players[game.resolve_trick(list(last))].name
            self.message = f"Baza para {winner}"
            self.freeze_until = time.time() + TRICK_PAUSE

        if game.phase == "done" and not self.round_recorded:
            self.match.record_round()
            self.round_recorded = True
            hit = [p.name for p in game.players if p.bid == p.tricks_won]
            self.message = (
                f"Ronda {self.match.round_number} terminada. "
                f"Cumplieron: {', '.join(hit) if hit else 'nadie'}. {self.match.standings()}"
            )

        self.next_bot_time = time.time() + BOT_DELAY

    def advance_round(self) -> None:
        if self.match.round_index + 1 >= len(self.match.ladder):
            self.match.finished = True
            winner = max(self.game.players, key=lambda p: p.score)
            self.message = f"Partida terminada. Ganó {winner.name} con {winner.score} puntos. (R para jugar otra)"
            return
        self.match.start_next_round()
        self.round_recorded = False
        self.freeze_until = 0.0
        self.next_bot_time = time.time() + BOT_DELAY
        self.message = f"Ronda {self.match.round_number}: {self.game.cards_per_player} cartas. A cantar."

    def step_bots(self) -> None:
        game = self.game
        if game.phase not in {"bidding", "playing"} or self.frozen:
            return
        if time.time() < self.next_bot_time:
            return

        idx = game.current_player_index
        if idx == HUMAN_INDEX and not self.autoplay:
            return

        if game.phase == "bidding":
            self.apply(52 + bot_bid(game, idx))
        else:
            self.apply(game.card_to_id(bot_play(game, idx)))

    # ------------------------------------------------------------------

    def draw(self) -> None:
        game = self.game
        human_turn = game.current_player_index == HUMAN_INDEX and not self.autoplay and not self.frozen

        self.renderer.render(
            game,
            perspective_player=HUMAN_INDEX,
            reveal_all=self.reveal_all,
            message=self.message,
            history=self.match.history,
            round_number=self.match.round_number,
            total_rounds=self.match.total_rounds,
            show_bid_buttons=human_turn and game.phase == "bidding",
            freeze_last_trick=self.frozen or game.phase == "done",
            next_round_button=game.phase == "done" and not self.match.finished,
            mouse_pos=self.mouse_pos,
        )


def main() -> None:
    try:
        import pygame  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Pygame no está instalado. Instalalo con: pip install pygame") from exc

    PodridaApp().run()


if __name__ == "__main__":
    main()
