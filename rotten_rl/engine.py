"""
rotten_rl/engine.py

Motor simple y testeable para el juego de cartas "La Podrida" / Oh Hell.
No depende de Gymnasium ni de PettingZoo: la idea es que este archivo sea el
núcleo del juego y después lo envuelvas en un ambiente de RL.

Reglas implementadas, por defecto:
- Mazo francés de 52 cartas.
- Cada ronda reparte N cartas por jugador.
- Se revela una carta para definir el palo de triunfo.
- Fase 1: cada jugador canta cuántas bazas cree que hará.
- El último jugador en cantar no puede hacer que la suma total de cantos sea
  igual a la cantidad de bazas de la ronda.
- Fase 2: se juegan bazas.
- Hay que seguir el palo de salida si se tiene una carta de ese palo.
- Si no se tiene el palo de salida, hay obligación de fallar: se debe jugar
  triunfo si se tiene alguno (regla por defecto, must_trump_when_void=True).
- Recién si no se tiene ni el palo de salida ni triunfo, se juega cualquier carta.
- Gana la baza la carta más alta del palo de triunfo; si no hay triunfo en la
  baza, gana la carta más alta del palo de salida.
- La partida estándar es una escalera de 13 rondas: 1,2,3,4,5,6,7,6,5,4,3,2,1
  (ver build_round_ladder).
- Puntaje: si el jugador cumple su canto, suma 10 + 2*bazas_ganadas; si no,
  suma bazas_ganadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


SUITS: Tuple[str, ...] = ("Hearts", "Diamonds", "Clubs", "Spades")
RANKS: Tuple[str, ...] = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
RANK_VALUE: Dict[str, int] = {rank: value for value, rank in enumerate(RANKS, start=2)}

CARD_ACTIONS = 52
BID_OFFSET = 52

# La partida es una escalera fija 1,2,...,7,...,2,1 (13 rondas).
# Fijar el máximo en 7 mantiene el espacio de acciones constante (52 cartas +
# cantos 0..7 = 60) para cualquier cantidad de jugadores, que es lo que necesita
# un agente que tiene que generalizar entre 2 y 7 jugadores.
MAX_HAND_SIZE = 7


def build_round_ladder(num_players: int, max_hand: int = MAX_HAND_SIZE) -> List[int]:
    """
    Escalera de rondas: sube de 1 hasta max_hand y vuelve a bajar hasta 1.

    Con 7 jugadores entran 7 cartas cada uno (49 de 52), así que la escalera
    completa 1..7..1 vale para 2 a 7 jugadores. Con más jugadores se recorta
    sola a lo que permita el mazo.
    """
    top = min(max_hand, 51 // num_players)
    if top < 1:
        raise ValueError("Demasiados jugadores para repartir siquiera una carta.")
    return list(range(1, top + 1)) + list(range(top - 1, 0, -1))


@dataclass(frozen=True, order=False)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"Rango inválido: {self.rank!r}. Rangos válidos: {RANKS}")
        if self.suit not in SUITS:
            raise ValueError(f"Palo inválido: {self.suit!r}. Palos válidos: {SUITS}")

    @property
    def value(self) -> int:
        return RANK_VALUE[self.rank]

    def to_tuple(self) -> Tuple[str, str]:
        return (self.rank, self.suit)

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


CardLike = Union[Card, Tuple[str, str], int]
PlayerRef = Union[int, str, "Player"]


@dataclass
class Player:
    name: str
    hand: List[Card] = field(default_factory=list)
    bid: Optional[int] = None
    tricks_won: int = 0
    score: int = 0
    last_round_score: int = 0

    def reset_for_round(self) -> None:
        self.hand.clear()
        self.bid = None
        self.tricks_won = 0
        self.last_round_score = 0


class InvalidMoveError(ValueError):
    """Error para acciones ilegales dentro de la partida."""


class PodridaGame:
    def __init__(
        self,
        num_players: int = 4,
        player_names: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        max_cards_per_player: Optional[int] = None,
        must_trump_when_void: bool = True,
    ) -> None:
        if num_players < 2:
            raise ValueError("La Podrida necesita al menos 2 jugadores.")
        if num_players > 10:
            raise ValueError("Con un mazo de 52 cartas no conviene usar más de 10 jugadores.")

        self.rng = random.Random(seed)
        self.num_players = num_players
        # True  -> si no tenés el palo de salida, estás obligado a jugar triunfo.
        # False -> regla estándar de Oh Hell: si no seguís el palo, tirás cualquiera.
        self.must_trump_when_void = must_trump_when_void
        # Por defecto 7: el tope de la escalera, no lo que entre en el mazo.
        # Así action_space_size no cambia con num_players.
        self.max_cards_per_player = max_cards_per_player or MAX_HAND_SIZE

        if player_names is None:
            player_names = [f"player_{i}" for i in range(num_players)]
        if len(player_names) != num_players:
            raise ValueError("La cantidad de nombres debe coincidir con num_players.")

        self.players: List[Player] = [Player(name=name) for name in player_names]
        self.deck: List[Card] = []
        self.discard_pile: List[Card] = []
        self.played_cards: List[Card] = []

        self.rounds_played = 0
        self.cards_per_player = 0
        self.dealer_index = 0
        self.first_player_index = 0
        self.current_player_index = 0

        self.phase: str = "not_started"  # not_started | bidding | playing | done
        self.trump_card: Optional[Card] = None
        self.trump_suit: Optional[str] = None
        self.current_trick: List[Tuple[int, Card]] = []
        self.completed_tricks: List[List[Tuple[int, Card]]] = []
        self.trick_number = 0

    # ------------------------------------------------------------------
    # Conversión entre cartas y acciones discretas
    # ------------------------------------------------------------------

    @staticmethod
    def create_deck() -> List[Card]:
        return [Card(rank, suit) for suit in SUITS for rank in RANKS]

    @staticmethod
    def card_to_id(card: CardLike) -> int:
        card_obj = PodridaGame.normalize_card(card)
        suit_idx = SUITS.index(card_obj.suit)
        rank_idx = RANKS.index(card_obj.rank)
        return suit_idx * len(RANKS) + rank_idx

    @staticmethod
    def id_to_card(card_id: int) -> Card:
        if not 0 <= card_id < CARD_ACTIONS:
            raise ValueError(f"card_id inválido: {card_id}. Debe estar entre 0 y 51.")
        suit_idx, rank_idx = divmod(card_id, len(RANKS))
        return Card(RANKS[rank_idx], SUITS[suit_idx])

    @staticmethod
    def normalize_card(card: CardLike) -> Card:
        if isinstance(card, Card):
            return card
        if isinstance(card, int):
            return PodridaGame.id_to_card(card)
        if isinstance(card, tuple) and len(card) == 2:
            return Card(card[0], card[1])
        raise TypeError(f"No se puede interpretar como carta: {card!r}")

    # ------------------------------------------------------------------
    # Inicio y administración de ronda
    # ------------------------------------------------------------------

    def reset_game(self, keep_players: bool = True) -> None:
        """Reinicia toda la partida, incluyendo puntajes."""
        if keep_players:
            for player in self.players:
                player.score = 0
                player.reset_for_round()
        else:
            self.players = [Player(name=f"player_{i}") for i in range(self.num_players)]

        self.deck = []
        self.discard_pile = []
        self.played_cards = []
        self.rounds_played = 0
        self.cards_per_player = 0
        self.dealer_index = 0
        self.first_player_index = 0
        self.current_player_index = 0
        self.phase = "not_started"
        self.trump_card = None
        self.trump_suit = None
        self.current_trick = []
        self.completed_tricks = []
        self.trick_number = 0

    def start_round(self, num_cards: int, dealer_index: Optional[int] = None) -> Dict[str, object]:
        """
        Inicia una ronda nueva de N cartas.

        Devuelve un diccionario con información útil para debug/render.
        """
        if num_cards <= 0:
            raise ValueError("num_cards debe ser mayor que 0.")
        if num_cards * self.num_players >= 52:
            raise ValueError(
                "No quedan cartas suficientes para repartir y revelar triunfo. "
                "Usá menos cartas por jugador o menos jugadores."
            )

        if dealer_index is not None:
            self.dealer_index = dealer_index % self.num_players

        for player in self.players:
            player.reset_for_round()

        self.deck = self.create_deck()
        self.rng.shuffle(self.deck)
        self.discard_pile = []
        self.played_cards = []
        self.current_trick = []
        self.completed_tricks = []
        self.trick_number = 0
        self.cards_per_player = num_cards

        self.deal_cards(num_cards)

        self.trump_card = self.deck.pop()
        self.trump_suit = self.trump_card.suit
        self.discard_pile.append(self.trump_card)

        self.first_player_index = (self.dealer_index + 1) % self.num_players
        self.current_player_index = self.first_player_index
        self.phase = "bidding"

        return self.public_state()

    def deal_cards(self, num_cards: int) -> None:
        """Reparte num_cards a cada jugador desde el mazo actual."""
        needed = num_cards * self.num_players
        if len(self.deck) < needed:
            raise ValueError("No hay suficientes cartas en el mazo para repartir.")

        for _ in range(num_cards):
            for player_offset in range(self.num_players):
                player_index = (self.first_player_index + player_offset) % self.num_players
                self.players[player_index].hand.append(self.deck.pop())

        for player in self.players:
            player.hand.sort(key=lambda c: (SUITS.index(c.suit), RANKS.index(c.rank)))

    # ------------------------------------------------------------------
    # Referencias de jugadores y estado
    # ------------------------------------------------------------------

    def player_index(self, player: Optional[PlayerRef] = None) -> int:
        if player is None:
            return self.current_player_index
        if isinstance(player, int):
            if not 0 <= player < self.num_players:
                raise IndexError(f"Índice de jugador fuera de rango: {player}")
            return player
        if isinstance(player, str):
            for idx, p in enumerate(self.players):
                if p.name == player:
                    return idx
            raise KeyError(f"No existe un jugador llamado {player!r}.")
        if isinstance(player, Player):
            for idx, p in enumerate(self.players):
                if p is player:
                    return idx
            raise ValueError("El jugador no pertenece a esta partida.")
        raise TypeError(f"Referencia de jugador inválida: {player!r}")

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    def public_state(self) -> Dict[str, object]:
        return {
            "phase": self.phase,
            "rounds_played": self.rounds_played,
            "cards_per_player": self.cards_per_player,
            "dealer_index": self.dealer_index,
            "first_player_index": self.first_player_index,
            "current_player_index": self.current_player_index,
            "current_player_name": self.current_player.name,
            "trump_card": self.trump_card,
            "trump_suit": self.trump_suit,
            "current_trick": list(self.current_trick),
            "trick_number": self.trick_number,
            "bids": [p.bid for p in self.players],
            "tricks_won": [p.tricks_won for p in self.players],
            "scores": [p.score for p in self.players],
            "last_round_scores": [p.last_round_score for p in self.players],
            "hand_sizes": [len(p.hand) for p in self.players],
        }

    def observation_for_player(self, player: PlayerRef) -> Dict[str, object]:
        """
        Observación legal desde la perspectiva de un jugador.
        No incluye las cartas ocultas de los rivales.
        """
        idx = self.player_index(player)
        return {
            **self.public_state(),
            "player_index": idx,
            "player_name": self.players[idx].name,
            "hand": list(self.players[idx].hand),
            "legal_bids": self.legal_bids(idx) if self.phase == "bidding" and idx == self.current_player_index else [],
            "legal_cards": self.legal_cards(idx) if self.phase == "playing" and idx == self.current_player_index else [],
            "legal_action_ids": self.legal_action_ids(idx) if idx == self.current_player_index else [],
        }

    # ------------------------------------------------------------------
    # Fase de canto
    # ------------------------------------------------------------------

    def _bids_made(self) -> int:
        return sum(player.bid is not None for player in self.players)

    def _is_last_bidder(self) -> bool:
        return self._bids_made() == self.num_players - 1

    def legal_bids(self, player: Optional[PlayerRef] = None) -> List[int]:
        idx = self.player_index(player)
        if self.phase != "bidding":
            return []
        if idx != self.current_player_index:
            return []
        if self.players[idx].bid is not None:
            return []

        legal = list(range(self.cards_per_player + 1))

        if self._is_last_bidder():
            total_so_far = sum(p.bid or 0 for p in self.players if p.bid is not None)
            forbidden_bid = self.cards_per_player - total_so_far
            if forbidden_bid in legal:
                legal.remove(forbidden_bid)

        return legal

    def call(self, player: PlayerRef, calling_number: int) -> Dict[str, object]:
        """
        Registra el canto de un jugador.

        Ejemplo:
            game.call(0, 2)
        """
        idx = self.player_index(player)
        if self.phase != "bidding":
            raise InvalidMoveError("No se puede cantar porque la partida no está en fase de canto.")
        if idx != self.current_player_index:
            raise InvalidMoveError(f"No es el turno de {self.players[idx].name} para cantar.")
        if calling_number not in self.legal_bids(idx):
            raise InvalidMoveError(
                f"Canto ilegal: {calling_number}. Canto legales: {self.legal_bids(idx)}"
            )

        self.players[idx].bid = calling_number

        if self._bids_made() == self.num_players:
            self.phase = "playing"
            self.current_player_index = self.first_player_index
        else:
            self.current_player_index = (self.current_player_index + 1) % self.num_players

        return self.public_state()

    # ------------------------------------------------------------------
    # Fase de juego de cartas
    # ------------------------------------------------------------------

    @property
    def lead_suit(self) -> Optional[str]:
        if not self.current_trick:
            return None
        return self.current_trick[0][1].suit

    def legal_cards(self, player: Optional[PlayerRef] = None) -> List[Card]:
        """
        Cartas jugables según la regla de seguimiento de palo.

        Prioridad:
        1. Si tenés el palo de salida, estás obligado a jugarlo.
        2. Si no lo tenés y must_trump_when_void=True, estás obligado a fallar
           con triunfo (si tenés alguno).
        3. Si no tenés ni el palo de salida ni triunfo, jugás lo que quieras.
        """
        idx = self.player_index(player)
        if self.phase != "playing":
            return []
        if idx != self.current_player_index:
            return []

        hand = self.players[idx].hand
        if not hand:
            return []

        lead_suit = self.lead_suit
        if lead_suit is None:
            return list(hand)

        same_suit_cards = [card for card in hand if card.suit == lead_suit]
        if same_suit_cards:
            return same_suit_cards

        if self.must_trump_when_void and self.trump_suit is not None:
            trump_cards = [card for card in hand if card.suit == self.trump_suit]
            if trump_cards:
                return trump_cards

        return list(hand)

    def is_card_legal(self, player: PlayerRef, card: CardLike) -> bool:
        card_obj = self.normalize_card(card)
        return card_obj in self.legal_cards(player)

    def play_card(self, player: PlayerRef, card: CardLike) -> Dict[str, object]:
        idx = self.player_index(player)
        card_obj = self.normalize_card(card)

        if self.phase != "playing":
            raise InvalidMoveError("No se puede jugar una carta porque la partida no está en fase de juego.")
        if idx != self.current_player_index:
            raise InvalidMoveError(f"No es el turno de {self.players[idx].name} para jugar.")
        if card_obj not in self.players[idx].hand:
            raise InvalidMoveError(f"{self.players[idx].name} no tiene la carta {card_obj}.")
        if card_obj not in self.legal_cards(idx):
            raise InvalidMoveError(
                f"Carta ilegal: {card_obj}. Cartas legales: {[str(c) for c in self.legal_cards(idx)]}"
            )

        self.players[idx].hand.remove(card_obj)
        self.current_trick.append((idx, card_obj))
        self.played_cards.append(card_obj)

        if len(self.current_trick) == self.num_players:
            winner_index = self.resolve_trick(self.current_trick)
            self.players[winner_index].tricks_won += 1
            self.completed_tricks.append(list(self.current_trick))
            self.current_trick.clear()
            self.trick_number += 1

            if self.trick_number == self.cards_per_player:
                self._score_round()
                self.phase = "done"
                self.rounds_played += 1
                self.dealer_index = (self.dealer_index + 1) % self.num_players
            else:
                self.current_player_index = winner_index
        else:
            self.current_player_index = (self.current_player_index + 1) % self.num_players

        return self.public_state()

    def resolve_trick(self, trick: Optional[List[Tuple[int, Card]]] = None) -> int:
        """Devuelve el índice del jugador ganador de una baza."""
        if trick is None:
            trick = self.current_trick
        if not trick:
            raise ValueError("No hay cartas en la baza para resolver.")

        lead_suit = trick[0][1].suit
        trump_cards = [(idx, card) for idx, card in trick if card.suit == self.trump_suit]

        if trump_cards:
            winner_idx, _ = max(trump_cards, key=lambda item: item[1].value)
            return winner_idx

        lead_cards = [(idx, card) for idx, card in trick if card.suit == lead_suit]
        winner_idx, _ = max(lead_cards, key=lambda item: item[1].value)
        return winner_idx

    # ------------------------------------------------------------------
    # Puntaje y finalización
    # ------------------------------------------------------------------

    def _score_round(self) -> None:
        for player in self.players:
            if player.bid is None:
                raise RuntimeError("No se puede puntuar una ronda si faltan cantos.")

            if player.tricks_won == player.bid:
                player.last_round_score = 10 + 2*player.tricks_won
            else:
                player.last_round_score = player.tricks_won

            player.score += player.last_round_score

    def is_round_done(self) -> bool:
        return self.phase == "done"

    def scores(self) -> Dict[str, int]:
        return {player.name: player.score for player in self.players}

    def round_scores(self) -> Dict[str, int]:
        return {player.name: player.last_round_score for player in self.players}

    def winner(self) -> Optional[Player]:
        if not self.players:
            return None
        return max(self.players, key=lambda p: p.score)

    # ------------------------------------------------------------------
    # Interfaz útil para RL: acciones discretas + máscaras
    # ------------------------------------------------------------------

    @property
    def action_space_size(self) -> int:
        # 0..51: jugar cartas. 52..52+max_cards_per_player: cantar 0..N.
        return CARD_ACTIONS + self.max_cards_per_player + 1

    def legal_action_ids(self, player: Optional[PlayerRef] = None) -> List[int]:
        idx = self.player_index(player)
        if idx != self.current_player_index:
            return []

        if self.phase == "bidding":
            return [BID_OFFSET + bid for bid in self.legal_bids(idx)]
        if self.phase == "playing":
            return [self.card_to_id(card) for card in self.legal_cards(idx)]
        return []

    def action_mask(self, player: Optional[PlayerRef] = None) -> List[bool]:
        mask = [False] * self.action_space_size
        for action_id in self.legal_action_ids(player):
            if 0 <= action_id < len(mask):
                mask[action_id] = True
        return mask

    def apply_action(self, action_id: int, player: Optional[PlayerRef] = None) -> Dict[str, object]:
        """
        Aplica una acción discreta, pensada para RL.

        Acciones:
        - 0..51: jugar carta.
        - 52..52+N: cantar N.
        """
        idx = self.player_index(player)
        if idx != self.current_player_index:
            raise InvalidMoveError(f"No es el turno de {self.players[idx].name}.")
        if action_id not in self.legal_action_ids(idx):
            raise InvalidMoveError(
                f"Acción ilegal: {action_id}. Acciones legales: {self.legal_action_ids(idx)}"
            )

        if self.phase == "bidding":
            bid = action_id - BID_OFFSET
            return self.call(idx, bid)

        if self.phase == "playing":
            card = self.id_to_card(action_id)
            return self.play_card(idx, card)

        raise InvalidMoveError(f"No se puede aplicar acción en fase {self.phase!r}.")

    # ------------------------------------------------------------------
    # Bots mínimos para debug
    # ------------------------------------------------------------------

    def random_legal_action(self, player: Optional[PlayerRef] = None) -> int:
        legal = self.legal_action_ids(player)
        if not legal:
            raise InvalidMoveError("No hay acciones legales disponibles.")
        return self.rng.choice(legal)

    def play_random_until_done(self) -> Dict[str, int]:
        """Juega toda la ronda con acciones legales aleatorias. Útil para testear el motor."""
        while not self.is_round_done():
            action = self.random_legal_action(self.current_player_index)
            self.apply_action(action, self.current_player_index)
        return self.scores()

    # ------------------------------------------------------------------
    # Render textual simple
    # ------------------------------------------------------------------

    def render(self, perspective_player: Optional[PlayerRef] = None, reveal_all: bool = False) -> str:
        lines: List[str] = []
        lines.append(f"Fase: {self.phase}")
        lines.append(f"Triunfo: {self.trump_card} ({self.trump_suit})")
        lines.append(f"Turno: {self.current_player.name} [{self.current_player_index}]")
        lines.append(f"Baza actual: {[(self.players[i].name, str(c)) for i, c in self.current_trick]}")
        lines.append("")

        perspective_idx = self.player_index(perspective_player) if perspective_player is not None else None

        for idx, player in enumerate(self.players):
            if reveal_all or perspective_idx == idx:
                hand = ", ".join(str(card) for card in player.hand)
            else:
                hand = f"{len(player.hand)} cartas ocultas"

            lines.append(
                f"{idx}. {player.name} | bid={player.bid} | bazas={player.tricks_won} | "
                f"score={player.score} | mano: {hand}"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    game = PodridaGame(num_players=4, seed=42)
    game.start_round(num_cards=5)
    print(game.render(reveal_all=True))
    print("\nJugando ronda random...\n")
    game.play_random_until_done()
    print(game.render(reveal_all=True))
    print("Puntajes:", game.scores())
