"""
rotten_rl/renderer.py

Render visual con Pygame para el motor de La Podrida definido en `engine.py`.

Todo el layout se calcula a partir del tamaño de la ventana: no hay posiciones
en píxeles fijos. La ventana es redimensionable y soporta pantalla completa, y
los asientos se reparten sobre una elipse, así que funciona igual con 2 que con
10 jugadores.

Instalación:
    pip install pygame

Uso mínimo:
    from rotten_rl.engine import PodridaGame
    from rotten_rl.renderer import PodridaPygameRenderer

    game = PodridaGame(num_players=4, seed=42)
    game.start_round(num_cards=5)

    renderer = PodridaPygameRenderer(render_mode="human")
    renderer.render(game, perspective_player=0)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .engine import BID_OFFSET, CARD_ACTIONS, Card, PlayerRef, PodridaGame, RANKS, SUITS


Color = Tuple[int, int, int]
Point = Tuple[int, int]
RectTuple = Tuple[int, int, int, int]


@dataclass
class RoundRecord:
    """Fila del historial de rondas."""

    round_number: int
    cards_per_player: int
    bids: List[int]
    tricks: List[int]
    points: List[int]
    totals: List[int]


@dataclass(frozen=True)
class CardStyle:
    """Proporciones de carta. Los tamaños reales los calcula el layout."""

    width: int = 92
    height: int = 132
    radius: int = 10
    gap: int = 18


@dataclass
class SeatLayout:
    """Geometría de un asiento ya resuelta para el tamaño de ventana actual."""

    player_index: int
    offset: int
    anchor: Point
    plate: Any
    fan: Any
    block: Any
    upper: bool
    visible: bool
    fan_step: int


class PodridaPygameRenderer:
    """
    Renderer visual para La Podrida.

    Modos:
    - render_mode="human": abre una ventana de Pygame (redimensionable).
    - render_mode="rgb_array": devuelve un array RGB (H, W, 3) para Gym/PettingZoo.

    El renderer solo lee el estado público de PodridaGame, nunca lo modifica.
    """

    # -------------------- paleta --------------------
    BG: Color = (17, 38, 31)
    FELT: Color = (28, 92, 66)
    FELT_DARK: Color = (18, 62, 45)
    FELT_EDGE: Color = (12, 44, 32)
    PANEL: Color = (23, 27, 32)
    PANEL_ALT: Color = (32, 37, 44)
    PANEL_LINE: Color = (58, 65, 74)
    TEXT: Color = (238, 241, 238)
    MUTED: Color = (152, 162, 168)
    GOLD: Color = (228, 184, 88)
    GREEN: Color = (110, 210, 145)
    RED_UI: Color = (232, 96, 96)
    BLUE: Color = (96, 160, 232)

    CARD_FACE: Color = (250, 249, 245)
    CARD_EDGE: Color = (196, 194, 186)
    CARD_BACK: Color = (38, 62, 122)
    CARD_BACK_LINE: Color = (92, 126, 200)
    SUIT_RED: Color = (188, 44, 48)
    SUIT_BLACK: Color = (30, 32, 38)

    RED_SUITS = {"Hearts", "Diamonds"}

    SUIT_SYMBOLS: Dict[str, str] = {
        "Hearts": "♥",
        "Diamonds": "♦",
        "Clubs": "♣",
        "Spades": "♠",
    }

    MIN_WIDTH = 900
    MIN_HEIGHT = 620

    def __init__(
        self,
        width: int = 1400,
        height: int = 880,
        render_mode: str = "human",
        fps: int = 60,
        card_style: Optional[CardStyle] = None,
        caption: str = "La Podrida",
        resizable: bool = True,
        fullscreen: bool = False,
    ) -> None:
        if render_mode not in {"human", "rgb_array"}:
            raise ValueError("render_mode debe ser 'human' o 'rgb_array'.")

        self.width = max(self.MIN_WIDTH, width)
        self.height = max(self.MIN_HEIGHT, height)
        self.render_mode = render_mode
        self.fps = fps
        self.card_style = card_style or CardStyle()
        self.caption = caption
        self.resizable = resizable

        self._pygame: Optional[Any] = None
        self._screen: Optional[Any] = None
        self._surface: Optional[Any] = None
        self._clock: Optional[Any] = None
        self._fonts: Dict[Tuple[int, bool], Any] = {}
        self._card_cache: Dict[Tuple[Any, ...], Any] = {}
        self._pip_cache: Dict[Tuple[Any, ...], Any] = {}
        self._back_cache: Dict[Tuple[int, int], Any] = {}
        self._text_cache: Dict[Tuple[Any, ...], Any] = {}
        self._shadow_cache: Dict[Tuple[int, int], Any] = {}
        self._veil_cache: Dict[Tuple[int, int], Any] = {}
        self._bg_cache: Optional[Any] = None
        self._bg_key: Optional[Tuple[int, int]] = None
        self._closed = False
        self._fullscreen = fullscreen
        self._windowed_size = (self.width, self.height)

        # Zonas clickeables, se recalculan en cada render.
        self.card_hitboxes: List[Tuple[Any, Card, bool]] = []
        self.bid_hitboxes: List[Tuple[Any, int]] = []
        self.button_hitboxes: List[Tuple[Any, str]] = []

        self.ui_scale = 1.0
        self._compute_layout(4)

    # ------------------------------------------------------------------
    # Infraestructura y manejo de ventana
    # ------------------------------------------------------------------

    def _require_pygame(self) -> Any:
        if self._pygame is not None:
            return self._pygame

        try:
            import pygame  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pygame no está instalado. Instalalo con: pip install pygame") from exc

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self._pygame = pygame
        self._clock = pygame.time.Clock()

        if self.render_mode == "human":
            self._apply_display_mode()
        else:
            self._surface = pygame.Surface((self.width, self.height))

        return pygame

    def _apply_display_mode(self) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        flags = 0
        if self._fullscreen:
            flags |= pygame.FULLSCREEN
        elif self.resizable:
            flags |= pygame.RESIZABLE
        self._screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption(self.caption)

    def desktop_size(self) -> Tuple[int, int]:
        """Tamaño del monitor, para abrir la ventana a medida."""
        pygame = self._require_pygame()
        try:
            sizes = pygame.display.get_desktop_sizes()
            if sizes:
                return sizes[0]
        except Exception:
            pass
        info = pygame.display.Info()
        return (info.current_w, info.current_h)

    def set_size(self, width: int, height: int) -> None:
        """Reajusta el render a un tamaño nuevo (usar al recibir VIDEORESIZE)."""
        width = max(self.MIN_WIDTH, int(width))
        height = max(self.MIN_HEIGHT, int(height))
        if (width, height) == (self.width, self.height):
            return

        self.width, self.height = width, height
        if not self._fullscreen:
            self._windowed_size = (width, height)

        # Los tamaños de carta y de fuente cambian: hay que tirar los cachés.
        self._clear_caches()

        if self._pygame is not None:
            if self.render_mode == "human":
                self._apply_display_mode()
            else:
                self._surface = self._pygame.Surface((self.width, self.height))

    def toggle_fullscreen(self) -> None:
        """Alterna entre ventana y pantalla completa del monitor actual."""
        pygame = self._require_pygame()
        if self._fullscreen:
            self._fullscreen = False
            w, h = self._windowed_size
        else:
            self._fullscreen = True
            w, h = self.desktop_size()

        self.width, self.height = max(self.MIN_WIDTH, w), max(self.MIN_HEIGHT, h)
        self._clear_caches()
        if self.render_mode == "human":
            self._apply_display_mode()

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    def _font(self, size: int, bold: bool = False) -> Any:
        """Las medidas se piden en 'píxeles de diseño' y se escalan a la ventana."""
        scaled = max(9, int(round(size * self.ui_scale)))
        key = (scaled, bold)
        if key not in self._fonts:
            pygame = self._require_pygame()
            self._fonts[key] = pygame.font.SysFont(
                "dejavusans,arial,helvetica,segoeui", scaled, bold=bold
            )
        return self._fonts[key]

    def _px(self, value: float) -> int:
        return max(1, int(round(value * self.ui_scale)))

    def _clear_caches(self) -> None:
        for cache in (
            self._card_cache, self._back_cache, self._pip_cache,
            self._fonts, self._text_cache, self._shadow_cache, self._veil_cache,
        ):
            cache.clear()
        self._bg_cache = None
        self._bg_key = None

    def close(self) -> None:
        if self._pygame is not None and not self._closed:
            self._pygame.quit()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # Hit testing (para jugar con el mouse)
    # ------------------------------------------------------------------

    def card_at(self, pos: Point, only_legal: bool = True) -> Optional[Card]:
        for rect, card, legal in reversed(self.card_hitboxes):
            if rect.collidepoint(pos):
                if only_legal and not legal:
                    return None
                return card
        return None

    def bid_at(self, pos: Point) -> Optional[int]:
        for rect, bid in self.bid_hitboxes:
            if rect.collidepoint(pos):
                return bid
        return None

    def button_at(self, pos: Point) -> Optional[str]:
        for rect, name in self.button_hitboxes:
            if rect.collidepoint(pos):
                return name
        return None

    # ------------------------------------------------------------------
    # Layout: todo sale de (width, height, num_players)
    # ------------------------------------------------------------------

    def _compute_layout(self, num_players: int) -> None:
        W, H = self.width, self.height

        # Escala tipográfica: crece con el monitor pero con techo, para que en
        # pantallas grandes se agranden las cartas y no solo las letras.
        self.ui_scale = max(0.72, min(1.9, min(W / 1400.0, H / 880.0)))

        self.header_height = int(min(120, max(52, H * 0.086)))
        self.panel_width = int(min(500, max(286, W * 0.255)))
        self.hand_zone_height = int(min(320, max(146, H * 0.235)))
        self.message_height = self._px(38)

        tx, ty, tw, th = self._table_rect

        # Con más jugadores, todo achica un poco para que entren los asientos.
        crowd = max(0.68, 1.0 - max(0, num_players - 4) * 0.05)
        self.crowd = crowd

        self.hand_card_h = int(min(220, max(84, self.hand_zone_height * 0.60)))
        self.hand_card_w = int(self.hand_card_h * 0.70)

        self.trick_card_h = int(max(62, min(180, th * 0.27) * crowd))
        self.trick_card_w = int(self.trick_card_h * 0.70)

        self.fan_card_h = int(max(34, min(96, th * 0.125) * crowd))
        self.fan_card_w = int(self.fan_card_h * 0.70)

        self.plate_w = int(max(126, min(300, tw * 0.205) * max(0.76, crowd)))
        self.plate_h = int(max(44, self.plate_w * 0.30))

        self.trump_card_h = int(min(150, max(62, th * 0.20)))
        self.trump_card_w = int(self.trump_card_h * 0.70)

        # Cuánto espacio hay realmente entre dos asientos vecinos de la elipse.
        # Si el bloque (placa + abanico) no entra, se achica antes de dibujar.
        rx, ry = tw * 0.405, th * 0.40
        ang = 2 * math.pi / max(2, num_players)
        chord = min(
            math.hypot(
                rx * (math.cos(a) - math.cos(a + ang)),
                ry * (math.sin(a) - math.sin(a + ang)),
            )
            for a in (math.pi / 2 + ang * i for i in range(num_players))
        )
        budget = max(self._px(38), chord * 0.94)
        block_h = self.plate_h + self.fan_card_h + self._px(8)
        if block_h > budget:
            k = budget / block_h
            self.fan_card_h = max(self._px(22), int(self.fan_card_h * k))
            self.fan_card_w = int(self.fan_card_h * 0.70)
            self.plate_h = max(self._px(34), int(self.plate_h * max(0.72, k)))
            self.plate_w = min(self.plate_w, int(chord * 1.75))

    @property
    def _main_rect(self) -> RectTuple:
        """Área de mesa: todo menos el encabezado y el panel derecho."""
        return (
            0,
            self.header_height,
            self.width - self.panel_width,
            self.height - self.header_height,
        )

    @property
    def _table_rect(self) -> RectTuple:
        x, y, w, h = self._main_rect
        margin = self._px(32)
        top = y + self._px(16)
        bottom = y + h - self.hand_zone_height - self._px(12)
        return (x + margin, top, w - 2 * margin, max(self._px(160), bottom - top))

    def _trump_rect(self) -> Any:
        pygame = self._require_pygame()
        tx, ty, tw, th = self._table_rect
        return pygame.Rect(
            tx + self._px(4),
            ty + self._px(22),
            self.trump_card_w,
            self.trump_card_h,
        )

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def render(
        self,
        game: PodridaGame,
        perspective_player: Optional[PlayerRef] = 0,
        reveal_all: bool = False,
        message: Optional[str] = None,
        history: Optional[Sequence[RoundRecord]] = None,
        round_number: Optional[int] = None,
        total_rounds: Optional[int] = None,
        show_bid_buttons: bool = False,
        freeze_last_trick: bool = False,
        next_round_button: bool = False,
        mouse_pos: Optional[Point] = None,
    ) -> Optional[Any]:
        """
        Dibuja el estado actual del juego.

        Args:
            game: instancia de PodridaGame.
            perspective_player: jugador que se sienta abajo y ve sus cartas.
            reveal_all: si True, muestra todas las manos.
            message: texto de la barra inferior.
            history: lista de RoundRecord para la tabla de historial.
            round_number / total_rounds: para el encabezado ("Ronda 3/9").
            show_bid_buttons: dibuja los botones de canto (turno humano).
            freeze_last_trick: muestra la última baza completa en el centro.
            next_round_button: dibuja el botón "Siguiente ronda".
            mouse_pos: posición del mouse para efectos de hover.
        """
        if self._closed:
            return None

        pygame = self._require_pygame()
        surface = self._screen if self.render_mode == "human" else self._surface
        if surface is None:
            raise RuntimeError("No se pudo inicializar la superficie de render.")

        self._compute_layout(game.num_players)

        self.card_hitboxes = []
        self.bid_hitboxes = []
        self.button_hitboxes = []

        try:
            perspective_idx = game.player_index(perspective_player) if perspective_player is not None else 0
        except Exception:
            perspective_idx = 0

        seats = self._seat_layout(game, perspective_idx)

        surface.blit(self._background(), (0, 0))
        self._draw_header(surface, game, round_number, total_rounds, perspective_idx)
        self._draw_table(surface, game)
        self._draw_seats(surface, game, seats, reveal_all)
        self._draw_trick(surface, game, seats, freeze_last_trick)
        self._draw_hand(surface, game, perspective_idx, mouse_pos)
        self._draw_scoreboard(surface, game, history, perspective_idx)
        self._draw_message_bar(surface, game, message)

        if show_bid_buttons:
            self._draw_bid_buttons(surface, game, mouse_pos)
        if next_round_button:
            self._draw_next_round_button(surface, mouse_pos)

        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.fps)
            return None

        array = pygame.surfarray.array3d(surface)
        try:
            import numpy as np  # type: ignore

            return np.transpose(array, (1, 0, 2))
        except ModuleNotFoundError:
            return array

    def _background(self) -> Any:
        """
        Fondo estático (fieltro, paneles, separadores) cacheado en una surface.
        Redibujar la elipse de la mesa en cada frame cuesta carísimo en 4K.
        """
        pygame = self._require_pygame()
        key = (self.width, self.height)
        if self._bg_cache is not None and self._bg_key == key:
            return self._bg_cache

        bg = pygame.Surface((self.width, self.height))
        bg.fill(self.BG)

        x, y, w, h = self._table_rect
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.ellipse(bg, self.FELT_EDGE, rect)
        pygame.draw.ellipse(bg, self.FELT_DARK, rect.inflate(-self._px(12), -self._px(12)))
        pygame.draw.ellipse(bg, self.FELT, rect.inflate(-self._px(26), -self._px(26)))
        pygame.draw.ellipse(bg, self.FELT_DARK, rect.inflate(-self._px(26), -self._px(26)), 2)

        mx, my, mw, mh = self._main_rect
        zone = pygame.Rect(mx, my + mh - self.hand_zone_height, mw, self.hand_zone_height - self.message_height)
        pygame.draw.rect(bg, (14, 30, 25), zone)
        pygame.draw.line(bg, (10, 22, 18), (zone.left, zone.top), (zone.right, zone.top), 2)

        pygame.draw.rect(bg, self.PANEL, (0, 0, self.width, self.header_height))
        pygame.draw.line(bg, self.PANEL_LINE, (0, self.header_height - 1), (self.width, self.header_height - 1))

        px0 = self.width - self.panel_width
        pygame.draw.rect(bg, self.PANEL, (px0, self.header_height, self.panel_width, self.height - self.header_height))
        pygame.draw.line(bg, self.PANEL_LINE, (px0, self.header_height), (px0, self.height))

        bar_top = self.height - self.message_height
        pygame.draw.rect(bg, self.PANEL, (mx, bar_top, mw, self.message_height))
        pygame.draw.line(bg, self.PANEL_LINE, (mx, bar_top), (mx + mw, bar_top))

        self._bg_cache = bg
        self._bg_key = key
        return bg

    # ------------------------------------------------------------------
    # Encabezado
    # ------------------------------------------------------------------

    def _draw_header(
        self,
        surface: Any,
        game: PodridaGame,
        round_number: Optional[int],
        total_rounds: Optional[int],
        perspective_idx: int = 0,
    ) -> None:
        title = self._text(
            surface, "LA PODRIDA",
            (self._px(24), int(self.header_height * 0.30)),
            26, self.GOLD, bold=True,
        )

        phase_label = {
            "not_started": "sin iniciar",
            "bidding": "cantos",
            "playing": "jugando",
            "done": "ronda terminada",
        }.get(game.phase, game.phase)

        chips: List[Tuple[str, str]] = []
        if round_number is not None:
            label = f"{round_number}" + (f"/{total_rounds}" if total_rounds else "")
            chips.append(("Ronda", label))
        chips.append(("Cartas", str(game.cards_per_player)))
        if game.cards_per_player:
            baza = min(game.trick_number + 1, game.cards_per_player)
            chips.append(("Baza", f"{baza}/{game.cards_per_player}"))
        chips.append(("Fase", phase_label))
        if game.phase in {"bidding", "playing"}:
            chips.append(("Turno", game.current_player.name))
            chips.append(("Tu lugar", self._position_label(game, perspective_idx)))
        if game.trump_suit:
            chips.append(("Triunfo", self.SUIT_SYMBOLS.get(game.trump_suit, game.trump_suit)))
        if game.phase == "playing" and game.lead_suit:
            chips.append(("Salida", self.SUIT_SYMBOLS.get(game.lead_suit, game.lead_suit)))

        x = title.right + self._px(34)
        label_y = int(self.header_height * 0.22)
        value_y = int(self.header_height * 0.46)
        limit = self.width - self._px(16)

        for label, value in chips:
            color = self.TEXT
            if label == "Triunfo":
                color = (240, 120, 120) if game.trump_suit in self.RED_SUITS else self.TEXT
            elif label == "Salida":
                color = (240, 120, 120) if game.lead_suit in self.RED_SUITS else self.BLUE

            lw = self._font(12, True).size(label.upper())[0]
            vw = self._font(20, True).size(value)[0]
            if x + max(lw, vw) > limit:
                break
            self._text(surface, label.upper(), (x, label_y), 12, self.MUTED, bold=True)
            self._text(surface, value, (x, value_y), 20, color, bold=True)
            x += max(lw, vw) + self._px(30)

    @staticmethod
    def _position_label(game: PodridaGame, player_idx: int) -> str:
        """En qué lugar canta o juega el jugador: en el canto arranca la mano; en la jugada, quien
        sale en la baza en curso (si todavía no salió nadie, quien tiene el turno)."""
        if game.phase == "bidding":
            first = game.first_player_index
        else:
            first = game.current_trick[0][0] if game.current_trick else game.current_player_index
        return f"{(player_idx - first) % game.num_players + 1}º de {game.num_players}"

    # ------------------------------------------------------------------
    # Mesa
    # ------------------------------------------------------------------

    def _draw_table(self, surface: Any, game: PodridaGame) -> None:
        if game.trump_card is not None:
            trump = self._trump_rect()
            self._text(surface, "TRIUNFO", (trump.x, trump.y - self._px(18)), 12, self.GOLD, bold=True)
            self._blit_card(surface, game.trump_card, trump.topleft, trump.size, highlight=True)

    # ------------------------------------------------------------------
    # Asientos
    # ------------------------------------------------------------------

    def _seat_layout(self, game: PodridaGame, perspective_idx: int) -> List[SeatLayout]:
        """
        Reparte los asientos sobre una elipse, con el jugador de perspectiva
        abajo. Después empuja hacia adentro los que pisen la carta de triunfo.
        """
        n = game.num_players
        tx, ty, tw, th = self._table_rect
        cx, cy = tx + tw / 2.0, ty + th / 2.0
        rx, ry = tw * 0.405, th * 0.40

        seats: List[SeatLayout] = []
        for offset in range(n):
            player_idx = (perspective_idx + offset) % n
            angle = math.pi / 2 + 2 * math.pi * offset / n
            ax = cx + rx * math.cos(angle)
            ay = cy + ry * math.sin(angle)
            seats.append(self._make_seat(game, player_idx, offset, (ax, ay)))

        trump = self._trump_rect().inflate(self._px(10), self._px(10))
        for i, seat in enumerate(seats):
            if not seat.visible:
                continue
            tries = 0
            while seat.block.colliderect(trump) and tries < 40:
                ax, ay = seat.anchor
                dx, dy = cx - ax, cy - ay
                dist = math.hypot(dx, dy) or 1.0
                step = self._px(6)
                seat = self._make_seat(
                    game, seat.player_index, seat.offset,
                    (ax + dx / dist * step, ay + dy / dist * step),
                )
                tries += 1
            seats[i] = seat

        self._relax_seats(game, seats)
        return seats

    def _relax_seats(self, game: PodridaGame, seats: List[SeatLayout], iterations: int = 40) -> None:
        """
        Separa a los que se hayan quedado pegados. Con muchos jugadores en una
        ventana chica, el reparto sobre la elipse más el recorte contra los
        bordes puede dejar dos bloques tocándose; acá se empujan.
        """
        n = len(seats)
        trump = self._trump_rect().inflate(self._px(10), self._px(10))
        for _ in range(iterations):
            moved = False
            for i in range(n):
                seat = seats[i]
                if not seat.visible or not seat.block.colliderect(trump):
                    continue
                overlap = seat.block.clip(trump)
                push = min(overlap.width, overlap.height) + 1.0
                ax, ay = seat.anchor
                vx, vy = ax - trump.centerx, ay - trump.centery
                dist = math.hypot(vx, vy) or 1.0
                seats[i] = self._make_seat(
                    game, seat.player_index, seat.offset,
                    (ax + vx / dist * push, ay + vy / dist * push),
                )
                moved = True
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = seats[i], seats[j]
                    if not (a.visible and b.visible):
                        continue
                    if not a.block.colliderect(b.block):
                        continue
                    overlap = a.block.clip(b.block)
                    push = min(overlap.width, overlap.height) / 2.0 + 1.0
                    ax, ay = a.anchor
                    bx, by = b.anchor
                    vx, vy = ax - bx, ay - by
                    dist = math.hypot(vx, vy) or 1.0
                    seats[i] = self._make_seat(
                        game, a.player_index, a.offset, (ax + vx / dist * push, ay + vy / dist * push)
                    )
                    seats[j] = self._make_seat(
                        game, b.player_index, b.offset, (bx - vx / dist * push, by - vy / dist * push)
                    )
                    moved = True
            if not moved:
                return

    def _make_seat(self, game: PodridaGame, player_idx: int, offset: int, anchor: Tuple[float, float]) -> SeatLayout:
        pygame = self._require_pygame()
        tx, ty, tw, th = self._table_rect
        pw, ph = self.plate_w, self.plate_h
        fh, fw = self.fan_card_h, self.fan_card_w
        gap = self._px(8)

        # La placa y su abanico tienen que entrar enteros en la mesa.
        ax = min(max(anchor[0], tx + pw / 2 + self._px(4)), tx + tw - pw / 2 - self._px(4))
        ay = min(
            max(anchor[1], ty + fh + ph / 2 + gap + self._px(4)),
            ty + th - ph / 2 - fh - gap - self._px(4),
        )
        ax, ay = int(ax), int(ay)

        plate = pygame.Rect(ax - pw // 2, ay - ph // 2, pw, ph)
        upper = ay < ty + th * 0.48
        n_cards = len(game.players[player_idx].hand)

        if n_cards:
            step = min(fw + self._px(4), max(self._px(11), int(pw * 0.94 / n_cards)))
            fan_w = step * (n_cards - 1) + fw
            fan_y = plate.top - gap - fh if upper else plate.bottom + gap
            fan = pygame.Rect(ax - fan_w // 2, fan_y, fan_w, fh)
            block = plate.union(fan)
        else:
            step = 0
            fan = pygame.Rect(ax, plate.bottom, 0, 0)
            block = plate.copy()

        return SeatLayout(
            player_index=player_idx,
            offset=offset,
            anchor=(ax, ay),
            plate=plate,
            fan=fan,
            block=block,
            upper=upper,
            visible=offset != 0,
            fan_step=step,
        )

    def _draw_seats(self, surface: Any, game: PodridaGame, seats: Sequence[SeatLayout], reveal_all: bool) -> None:
        for seat in seats:
            if seat.visible:
                self._draw_seat(surface, game, seat, reveal_all)

    def _draw_seat(self, surface: Any, game: PodridaGame, seat: SeatLayout, reveal_all: bool) -> None:
        pygame = self._require_pygame()
        player = game.players[seat.player_index]
        is_current = seat.player_index == game.current_player_index and game.phase in {"bidding", "playing"}
        is_dealer = seat.player_index == game.dealer_index
        rect = seat.plate

        pygame.draw.rect(surface, self.PANEL_ALT if is_current else self.PANEL, rect, border_radius=self._px(12))
        pygame.draw.rect(
            surface, self.GOLD if is_current else self.PANEL_LINE, rect,
            width=self._px(3) if is_current else 1, border_radius=self._px(12),
        )

        pad = self._px(9)
        # Tamaños en unidades de diseño: _font ya multiplica por ui_scale.
        fs_name = self._fit_font(rect.height * 0.27, 11, 18)
        fs_value = self._fit_font(rect.height * 0.25, 11, 17)
        fs_label = self._fit_font(rect.height * 0.18, 9, 12)

        self._text(
            surface, self._ellipsize(player.name, fs_name, rect.width - self._px(42), bold=True),
            (rect.x + pad, rect.y + int(rect.height * 0.10)), fs_name,
            self.TEXT if is_current else self.MUTED, bold=True,
        )
        if is_dealer:
            side = int(rect.height * 0.30)
            badge = pygame.Rect(rect.right - side - self._px(6), rect.y + int(rect.height * 0.10), side, side)
            pygame.draw.rect(surface, self.GOLD, badge, border_radius=self._px(5))
            self._text(surface, "D", badge.center, fs_label, self.PANEL, bold=True, center=True)

        bid = "–" if player.bid is None else str(player.bid)
        won_color = self.GREEN if player.bid is not None and player.tricks_won == player.bid else self.TEXT
        row_y = rect.y + int(rect.height * 0.50)
        self._stat_chip(surface, (rect.x + pad, row_y), "canto", bid, self.BLUE, fs_label, fs_value)
        self._stat_chip(surface, (rect.x + int(rect.width * 0.52), row_y), "bazas",
                        str(player.tricks_won), won_color, fs_label, fs_value)

        cards = list(player.hand)
        if not cards:
            return
        for i, card in enumerate(cards):
            pos = (seat.fan.x + i * seat.fan_step, seat.fan.y)
            size = (self.fan_card_w, self.fan_card_h)
            if reveal_all:
                self._blit_card(surface, card, pos, size)
            else:
                self._blit_back(surface, pos, size)

    def _fit_font(self, target_px: float, lo: int, hi: int) -> int:
        """Convierte un alto en píxeles reales a unidades de diseño (pre ui_scale)."""
        return int(max(lo, min(hi, target_px / max(0.01, self.ui_scale))))

    def _stat_chip(
        self,
        surface: Any,
        pos: Point,
        label: str,
        value: str,
        color: Color,
        label_size: int = 12,
        value_size: int = 17,
    ) -> None:
        x, y = pos
        offset = max(1, int((value_size - label_size) * self.ui_scale * 0.45))
        w = self._text(surface, label, (x, y + offset), label_size, self.MUTED).width
        self._text(surface, value, (x + w + self._px(5), y), value_size, color, bold=True)

    # ------------------------------------------------------------------
    # Baza
    # ------------------------------------------------------------------

    def _draw_trick(
        self,
        surface: Any,
        game: PodridaGame,
        seats: Sequence[SeatLayout],
        freeze_last_trick: bool,
    ) -> None:
        pygame = self._require_pygame()
        trick: Sequence[Tuple[int, Card]] = game.current_trick
        winner_idx: Optional[int] = None

        if not trick and freeze_last_trick and game.completed_tricks:
            trick = game.completed_tricks[-1]
            try:
                winner_idx = game.resolve_trick(list(trick))
            except Exception:
                winner_idx = None

        tx, ty, tw, th = self._table_rect
        cx, cy = tx + tw // 2, ty + th // 2

        if not trick:
            self._text(surface, "esperando la primera carta", (cx, cy), 16, (120, 160, 140), center=True)
            return

        by_player = {seat.player_index: seat for seat in seats}
        obstacles = [seat.block for seat in seats if seat.visible]
        obstacles.append(self._trump_rect())
        cw, ch = self.trick_card_w, self.trick_card_h
        lead_suit = trick[0][1].suit

        for i, (player_idx, card) in enumerate(trick):
            seat = by_player.get(player_idx)
            ax, ay = seat.anchor if seat is not None else (cx, cy)

            # Se acerca la carta al centro hasta que deje de pisar cualquier asiento.
            rect = pygame.Rect(0, 0, cw, ch)
            f = 0.46
            while f >= 0.10:
                rect.center = (int(cx + (ax - cx) * f), int(cy + (ay - cy) * f))
                pad = self._px(7)
                if not any(rect.colliderect(o.inflate(pad, pad)) for o in obstacles):
                    break
                f -= 0.02
            if game.num_players == 2:    # enfrentados: lado a lado, si no se pisan
                side = -1 if ay < cy else 1
                rect.center = (cx + side * int(cw * 0.62), cy)

            is_lead = i == 0
            self._blit_card(
                surface, card, rect.topleft, (cw, ch),
                highlight=(player_idx == winner_idx),
                outline=self.BLUE if is_lead else None,
            )
            color = self.GOLD if player_idx == winner_idx else self.MUTED
            self._text(
                surface, game.players[player_idx].name,
                (rect.centerx, rect.bottom + self._px(5)), 13, color, center=True, bold=True,
            )
            if is_lead:
                tag = f"SALIDA {self.SUIT_SYMBOLS.get(lead_suit, lead_suit)}"
                self._text(
                    surface, tag, (rect.centerx, rect.bottom + self._px(21)),
                    12, self.BLUE, center=True, bold=True,
                )

        if winner_idx is not None:
            self._text(
                surface, f"baza para {game.players[winner_idx].name}",
                (cx, ty + th - self._px(20)), 17, self.GOLD, center=True, bold=True,
            )

    # ------------------------------------------------------------------
    # Mano del jugador de perspectiva
    # ------------------------------------------------------------------

    def _draw_hand(
        self,
        surface: Any,
        game: PodridaGame,
        perspective_idx: int,
        mouse_pos: Optional[Point],
    ) -> None:
        pygame = self._require_pygame()
        mx, my, mw, mh = self._main_rect
        zone = pygame.Rect(
            mx, my + mh - self.hand_zone_height,
            mw, self.hand_zone_height - self.message_height,
        )

        player = game.players[perspective_idx]
        is_current = perspective_idx == game.current_player_index and game.phase in {"bidding", "playing"}

        plate_w = min(self._px(210), int(mw * 0.20))
        plate = pygame.Rect(mx + self._px(14), zone.top + self._px(16), plate_w, self._px(74))
        pygame.draw.rect(surface, self.PANEL_ALT if is_current else self.PANEL, plate, border_radius=self._px(12))
        pygame.draw.rect(
            surface, self.GOLD if is_current else self.PANEL_LINE, plate,
            width=self._px(3) if is_current else 1, border_radius=self._px(12),
        )
        name = player.name + ("" if player.name.lower() == "vos" else " (vos)")
        self._text(
            surface, self._ellipsize(name, 17, plate.width - self._px(46), bold=True),
            (plate.x + self._px(12), plate.y + self._px(9)), 17, self.TEXT, bold=True,
        )
        bid = "–" if player.bid is None else str(player.bid)
        won_color = self.GREEN if player.bid is not None and player.tricks_won == player.bid else self.TEXT
        self._stat_chip(surface, (plate.x + self._px(12), plate.y + self._px(38)), "canto", bid, self.BLUE)
        self._stat_chip(
            surface, (plate.x + int(plate.width * 0.50), plate.y + self._px(38)),
            "bazas", str(player.tricks_won), won_color,
        )
        if perspective_idx == game.dealer_index:
            badge = pygame.Rect(plate.right - self._px(32), plate.y + self._px(8), self._px(24), self._px(20))
            pygame.draw.rect(surface, self.GOLD, badge, border_radius=self._px(6))
            self._text(surface, "D", badge.center, 14, self.PANEL, bold=True, center=True)

        cards = list(player.hand)
        area_x = plate.right + self._px(20)
        area_w = mw - (area_x - mx) - self._px(16)

        if not cards:
            self._text(surface, "sin cartas", (area_x, zone.centery), 18, self.MUTED)
            return

        if is_current and game.phase == "playing":
            hint, hint_color = self._play_hint(game, cards)
            self._text(surface, hint, (area_x, zone.top + self._px(5)), 15, hint_color, bold=True)

        legal_cards = set()
        if is_current and game.phase == "playing":
            try:
                legal_cards = set(game.legal_cards(perspective_idx))
            except Exception:
                legal_cards = set()

        cw, ch = self.hand_card_w, self.hand_card_h
        step = cw + self._px(10)
        if len(cards) * step > area_w:
            step = max(self._px(24), (area_w - cw) // max(1, len(cards) - 1))
        total_w = step * (len(cards) - 1) + cw
        start_x = area_x + max(0, (area_w - total_w) // 2)
        base_y = zone.top + self._px(26)
        if base_y + ch > zone.bottom - self._px(4):
            base_y = max(zone.top + self._px(22), zone.bottom - ch - self._px(4))

        hovered = -1
        if mouse_pos is not None:
            for i in range(len(cards)):
                hit_w = step if i < len(cards) - 1 else cw
                if pygame.Rect(start_x + i * step, base_y, hit_w, ch).collidepoint(mouse_pos):
                    hovered = i

        playable_phase = is_current and game.phase == "playing"
        for i, card in enumerate(cards):
            legal = (not playable_phase) or card in legal_cards
            lift = self._px(14) if (i == hovered and legal and playable_phase) else 0
            pos = (start_x + i * step, base_y - lift)
            self._blit_card(
                surface, card, pos, (cw, ch),
                dim=playable_phase and not legal,
                highlight=playable_phase and legal,
            )
            hit_w = step if i < len(cards) - 1 else cw
            self.card_hitboxes.append((pygame.Rect(pos[0], pos[1], hit_w, ch), card, legal))

    def _play_hint(self, game: PodridaGame, cards: Sequence[Card]) -> Tuple[str, Color]:
        """Explica en una línea a qué está obligado el jugador en esta baza."""
        lead = game.lead_suit
        trump = game.trump_suit
        if lead is None:
            return ("Salís vos: podés jugar cualquier carta", self.MUTED)

        lead_sym = self.SUIT_SYMBOLS.get(lead, lead)
        if any(c.suit == lead for c in cards):
            return (f"Salida {lead_sym}: tenés que seguir el palo", self.BLUE)

        must_trump = getattr(game, "must_trump_when_void", False)
        if must_trump and trump is not None and any(c.suit == trump for c in cards):
            trump_sym = self.SUIT_SYMBOLS.get(trump, trump)
            return (f"No tenés {lead_sym}: estás obligado a fallar con triunfo {trump_sym}", self.GOLD)

        return (f"No tenés {lead_sym} ni triunfo: jugás la que quieras", self.MUTED)

    # ------------------------------------------------------------------
    # Botones
    # ------------------------------------------------------------------

    def _draw_bid_buttons(self, surface: Any, game: PodridaGame, mouse_pos: Optional[Point]) -> None:
        pygame = self._require_pygame()
        try:
            bids = game.legal_bids(game.current_player_index)
        except Exception:
            bids = []
        if not bids:
            return

        mx, my, mw, mh = self._main_rect
        gap = self._px(9)
        bw = self._px(54)
        bh = self._px(48)
        max_total = mw - self._px(80)
        if len(bids) * (bw + gap) > max_total:
            bw = max(self._px(30), (max_total - gap * len(bids)) // len(bids))

        total = len(bids) * bw + (len(bids) - 1) * gap
        # El último no puede hacer que los cantos sumen justo las bazas de la ronda.
        banned = None
        if len(bids) == game.cards_per_player:
            banned = next(b for b in range(game.cards_per_player + 1) if b not in bids)
        note = f"No podés cantar {banned}: sumaría {game.cards_per_player}" if banned is not None else None
        title = "¿Cuántas bazas cantás?"
        text_w = max(self._font(17, True).size(title)[0],
                     self._font(14).size(note)[0] if note else 0)
        extra = self._px(20) if note else 0
        panel = pygame.Rect(0, 0, max(total, text_w) + self._px(36), bh + self._px(62) + extra)
        panel.center = (mx + mw // 2, my + mh - self.hand_zone_height - self._px(66) - extra // 2)

        pygame.draw.rect(surface, self.PANEL, panel, border_radius=self._px(14))
        pygame.draw.rect(surface, self.GOLD, panel, width=self._px(2), border_radius=self._px(14))
        self._text(
            surface, title, (panel.centerx, panel.y + self._px(13)),
            17, self.TEXT, center=True, bold=True,
        )
        if note:
            self._text(surface, note, (panel.centerx, panel.y + self._px(33)), 14, self.MUTED, center=True)

        x = panel.centerx - total // 2
        y = panel.y + self._px(40) + extra
        for bid in bids:
            rect = pygame.Rect(x, y, bw, bh)
            hover = mouse_pos is not None and rect.collidepoint(mouse_pos)
            pygame.draw.rect(surface, self.GOLD if hover else self.PANEL_ALT, rect, border_radius=self._px(10))
            pygame.draw.rect(surface, self.GOLD, rect, width=self._px(2), border_radius=self._px(10))
            self._text(surface, str(bid), rect.center, 23, self.PANEL if hover else self.TEXT, center=True, bold=True)
            self.bid_hitboxes.append((rect, bid))
            x += bw + gap

    def _draw_next_round_button(self, surface: Any, mouse_pos: Optional[Point]) -> None:
        pygame = self._require_pygame()
        mx, my, mw, mh = self._main_rect
        rect = pygame.Rect(0, 0, self._px(240), self._px(52))
        rect.center = (mx + mw // 2, my + mh - self.hand_zone_height - self._px(56))
        hover = mouse_pos is not None and rect.collidepoint(mouse_pos)
        pygame.draw.rect(surface, self.GOLD if hover else self.PANEL, rect, border_radius=self._px(12))
        pygame.draw.rect(surface, self.GOLD, rect, width=self._px(2), border_radius=self._px(12))
        self._text(
            surface, "Siguiente ronda  →", rect.center, 20,
            self.PANEL if hover else self.GOLD, center=True, bold=True,
        )
        self.button_hitboxes.append((rect, "next_round"))

    # ------------------------------------------------------------------
    # Marcador
    # ------------------------------------------------------------------

    def _draw_scoreboard(
        self,
        surface: Any,
        game: PodridaGame,
        history: Optional[Sequence[RoundRecord]],
        perspective_idx: int,
    ) -> None:
        pygame = self._require_pygame()
        px0 = self.width - self.panel_width
        pad = self._px(16)
        x0 = px0 + pad
        inner_w = self.panel_width - 2 * pad
        y = self.header_height + self._px(14)

        # ---------- Ronda actual ----------
        self._text(surface, "RONDA ACTUAL", (x0, y), 13, self.GOLD, bold=True)
        total_bids = sum(p.bid for p in game.players if p.bid is not None)
        pending = sum(1 for p in game.players if p.bid is None)
        cantado = f"cantado {total_bids} / {game.cards_per_player}"
        if pending:
            cantado += f"  ({pending} sin cantar)"
        self._text(surface, cantado, (px0 + self.panel_width - pad, y + 1), 12, self.MUTED, align="right")
        y += self._px(22)

        col_name = x0
        col_bid = x0 + int(inner_w * 0.50)
        col_won = x0 + int(inner_w * 0.66)
        col_pts = x0 + int(inner_w * 0.83)
        col_tot = x0 + inner_w

        self._text(surface, "JUGADOR", (col_name, y), 11, self.MUTED, bold=True)
        self._text(surface, "CANTO", (col_bid, y), 11, self.MUTED, bold=True, align="center")
        self._text(surface, "BAZAS", (col_won, y), 11, self.MUTED, bold=True, align="center")
        self._text(surface, "PTS", (col_pts, y), 11, self.MUTED, bold=True, align="center")
        self._text(surface, "TOTAL", (col_tot, y), 11, self.MUTED, bold=True, align="right")
        y += self._px(18)
        pygame.draw.line(surface, self.PANEL_LINE, (x0, y), (x0 + inner_w, y))
        y += self._px(5)

        row_h = max(self._px(26), self._font(16, True).get_linesize() + self._px(7))
        round_over = game.phase == "done"
        name_w = int(inner_w * 0.46)

        for idx, player in enumerate(game.players):
            row = pygame.Rect(x0 - self._px(5), y - self._px(2), inner_w + self._px(10), row_h - self._px(2))
            if idx == game.current_player_index and game.phase in {"bidding", "playing"}:
                pygame.draw.rect(surface, (44, 52, 62), row, border_radius=self._px(6))
            elif idx == perspective_idx:
                pygame.draw.rect(surface, (30, 36, 43), row, border_radius=self._px(6))

            suffix = " (vos)" if idx == perspective_idx and player.name.lower() != "vos" else ""
            self._text(
                surface, self._ellipsize(player.name + suffix, 15, name_w, bold=idx == perspective_idx),
                (col_name, y + self._px(2)), 15,
                self.TEXT if idx == perspective_idx else self.MUTED, bold=idx == perspective_idx,
            )

            bid_txt = "–" if player.bid is None else str(player.bid)
            self._text(surface, bid_txt, (col_bid, y + self._px(2)), 16, self.BLUE, align="center", bold=True)

            if player.bid is None:
                won_color = self.MUTED
            elif player.tricks_won == player.bid:
                won_color = self.GREEN
            elif player.tricks_won > player.bid:
                won_color = self.RED_UI
            else:
                won_color = self.TEXT
            self._text(surface, str(player.tricks_won), (col_won, y + self._px(2)), 16, won_color,
                       align="center", bold=True)

            pts = player.last_round_score if round_over else self._projected_points(player)
            pts_color = self.GREEN if round_over and player.bid == player.tricks_won else (
                self.MUTED if not round_over else self.TEXT
            )
            self._text(surface, f"+{pts}" if round_over else f"({pts})", (col_pts, y + self._px(2)),
                       15, pts_color, align="center")

            self._text(surface, str(player.score), (col_tot, y + self._px(1)), 17, self.GOLD,
                       align="right", bold=True)
            y += row_h

        y += self._px(4)
        pygame.draw.line(surface, self.PANEL_LINE, (x0, y), (x0 + inner_w, y))
        y += self._px(12)

        # ---------- Historial ----------
        n = game.num_players
        self._text(surface, "HISTORIAL", (x0, y), 13, self.GOLD, bold=True)
        self._text(surface, "canto · bazas · pts", (px0 + self.panel_width - pad, y + 1), 11,
                   self.MUTED, align="right")
        y += self._px(20)

        r_col = self._px(26)
        col_w = (inner_w - r_col) / n
        hist_size = 12 if n <= 4 else (11 if n <= 6 else 10)
        pair_dx = col_w * 0.25
        pts_dx = col_w * 0.27

        self._text(surface, "R", (x0, y), 11, self.MUTED, bold=True)
        for i, player in enumerate(game.players):
            cx_col = x0 + r_col + col_w * (i + 0.5)
            self._text(surface, self._ellipsize(player.name, 11, int(col_w) - 2, bold=True),
                       (int(cx_col), y), 11, self.MUTED, align="center", bold=True)
        y += self._px(16)
        pygame.draw.line(surface, self.PANEL_LINE, (x0, y), (x0 + inner_w, y))
        y += self._px(4)

        records = list(history or [])
        hist_row = max(self._px(20), self._font(hist_size).get_linesize() + self._px(5))
        totals_y = self.height - self._px(30)
        max_rows = max(0, int((totals_y - self._px(14) - y) // hist_row))
        if len(records) > max_rows:
            records = records[-max_rows:]

        if not records:
            self._text(surface, "todavía no terminó ninguna ronda", (x0, y + self._px(5)), 12, self.MUTED)

        for r_i, record in enumerate(records):
            row = pygame.Rect(x0 - self._px(5), y - self._px(2), inner_w + self._px(10), hist_row - self._px(2))
            if r_i % 2 == 0:
                pygame.draw.rect(surface, self.PANEL_ALT, row, border_radius=self._px(4))
            self._text(surface, str(record.round_number), (x0 + r_col // 2, y), 13, self.MUTED, align="center")
            for i in range(n):
                cx_col = x0 + r_col + col_w * (i + 0.5)
                bid = record.bids[i] if i < len(record.bids) else 0
                won = record.tricks[i] if i < len(record.tricks) else 0
                pts = record.points[i] if i < len(record.points) else 0
                hit = bid == won
                self._text(surface, f"{bid}·{won}", (int(cx_col - pair_dx), y + 1), hist_size,
                           self.GREEN if hit else self.MUTED, align="center")
                self._text(surface, f"+{pts}", (int(cx_col + pts_dx), y + 1), hist_size,
                           self.TEXT if hit else self.MUTED, align="center")
            y += hist_row

        # Totales al pie.
        pygame.draw.line(surface, self.PANEL_LINE, (x0, totals_y - self._px(7)), (x0 + inner_w, totals_y - self._px(7)))
        self._text(surface, "TOT", (x0, totals_y), 12, self.GOLD, bold=True)
        for i, player in enumerate(game.players):
            cx_col = x0 + r_col + col_w * (i + 0.5)
            self._text(surface, str(player.score), (int(cx_col), totals_y - self._px(2)), 16, self.GOLD,
                       align="center", bold=True)

    @staticmethod
    def _projected_points(player: Any) -> int:
        """Puntos que sumaría el jugador si la ronda terminara ahora."""
        if player.bid is None:
            return 0
        return 10 + player.tricks_won if player.tricks_won == player.bid else player.tricks_won

    # ------------------------------------------------------------------
    # Barra de mensajes
    # ------------------------------------------------------------------

    def _draw_message_bar(self, surface: Any, game: PodridaGame, message: Optional[str]) -> None:
        pygame = self._require_pygame()
        mx, my, mw, mh = self._main_rect
        bar = pygame.Rect(mx, self.height - self.message_height, mw, self.message_height)

        text = message or ""
        if not text:
            if game.phase == "bidding":
                text = f"Turno de canto: {game.current_player.name}"
            elif game.phase == "playing":
                text = f"Turno de jugar: {game.current_player.name}"
            elif game.phase == "done":
                text = "Ronda terminada"
            else:
                text = "Partida sin iniciar"

        color = self.GOLD if game.phase == "done" else self.TEXT
        text = self._ellipsize(text, 16, mw - self._px(34))
        self._text(surface, text, (mx + self._px(16), bar.centery), 16, color, valign="center")

    # ------------------------------------------------------------------
    # Texto
    # ------------------------------------------------------------------

    def _ellipsize(self, text: str, size: int, max_width: int, bold: bool = False) -> str:
        font = self._font(size, bold)
        text = str(text)
        if font.size(text)[0] <= max_width:
            return text
        while text and font.size(text + "…")[0] > max_width:
            text = text[:-1]
        return text + "…"

    def _text(
        self,
        surface: Any,
        text: str,
        pos: Point,
        size: int = 16,
        color: Color = TEXT,
        bold: bool = False,
        center: bool = False,
        align: str = "left",
        valign: str = "top",
    ) -> Any:
        key = (str(text), size, bold, color)
        rendered = self._text_cache.get(key)
        if rendered is None:
            rendered = self._font(size, bold).render(str(text), True, color)
            if len(self._text_cache) > 700:
                self._text_cache.clear()
            self._text_cache[key] = rendered
        rect = rendered.get_rect()
        if center:
            rect.center = pos
        else:
            if align == "right":
                rect.topright = pos
            elif align == "center":
                rect.midtop = pos
            else:
                rect.topleft = pos
            if valign == "center":
                rect.centery = pos[1]
        surface.blit(rendered, rect)
        return rect

    # ------------------------------------------------------------------
    # Dibujo de cartas
    # ------------------------------------------------------------------

    def _blit_card(
        self,
        surface: Any,
        card: Card,
        pos: Point,
        size: Optional[Tuple[int, int]] = None,
        highlight: bool = False,
        dim: bool = False,
        outline: Optional[Color] = None,
    ) -> Any:
        pygame = self._require_pygame()
        w, h = size or (self.card_style.width, self.card_style.height)
        rect = pygame.Rect(pos[0], pos[1], w, h)

        self._blit_shadow(surface, rect)
        surface.blit(self._card_surface(card, w, h), rect.topleft)

        if dim:
            surface.blit(self._veil_surface(w, h), rect.topleft)
        if outline is not None and not highlight:
            pygame.draw.rect(surface, outline, rect, width=3, border_radius=int(min(w, h) * 0.09))
        if highlight:
            pygame.draw.rect(surface, self.GOLD, rect, width=3, border_radius=int(min(w, h) * 0.09))
        return rect

    def _blit_back(self, surface: Any, pos: Point, size: Optional[Tuple[int, int]] = None) -> Any:
        pygame = self._require_pygame()
        w, h = size or (self.card_style.width, self.card_style.height)
        rect = pygame.Rect(pos[0], pos[1], w, h)
        self._blit_shadow(surface, rect)
        surface.blit(self._back_surface(w, h), rect.topleft)
        return rect

    def _blit_shadow(self, surface: Any, rect: Any) -> None:
        pygame = self._require_pygame()
        key = (rect.width, rect.height)
        shadow = self._shadow_cache.get(key)
        if shadow is None:
            shadow = pygame.Surface(key, pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0, 0, 0, 90), shadow.get_rect(),
                             border_radius=int(min(key) * 0.09))
            self._shadow_cache[key] = shadow
        surface.blit(shadow, (rect.x + 3, rect.y + 4))

    def _veil_surface(self, w: int, h: int) -> Any:
        pygame = self._require_pygame()
        key = (w, h)
        veil = self._veil_cache.get(key)
        if veil is None:
            veil = pygame.Surface(key, pygame.SRCALPHA)
            pygame.draw.rect(veil, (10, 16, 14, 150), veil.get_rect(),
                             border_radius=int(min(w, h) * 0.09))
            self._veil_cache[key] = veil
        return veil

    # -------------------- pips vectoriales --------------------

    @staticmethod
    def _heart_points(cx: float, cy: float, size: float, flip: bool = False) -> List[Point]:
        pts: List[Point] = []
        steps = 60
        for i in range(steps):
            t = 2 * math.pi * i / steps
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            sx = cx + x * size / 34.0
            sy = cy - y * size / 34.0 * (-1 if flip else 1)
            pts.append((sx, sy))
        return pts

    def _pip_surface(self, suit: str, size: int, color: Color, upside_down: bool = False) -> Any:
        """Devuelve una superficie con el símbolo del palo, dibujado vectorialmente."""
        pygame = self._require_pygame()
        size = max(6, int(size))
        key = (suit, size, color, upside_down)
        if key in self._pip_cache:
            return self._pip_cache[key]

        ss = 4  # supersampling para bordes suaves
        s = size * ss
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        cx = cy = s / 2

        if suit == "Hearts":
            pygame.draw.polygon(surf, color, self._heart_points(cx, cy + s * 0.03, s * 0.98))
        elif suit == "Spades":
            pygame.draw.polygon(surf, color, self._heart_points(cx, cy - s * 0.06, s * 0.92, flip=True))
            stem = [
                (cx - s * 0.055, cy + s * 0.10),
                (cx + s * 0.055, cy + s * 0.10),
                (cx + s * 0.20, cy + s * 0.44),
                (cx - s * 0.20, cy + s * 0.44),
            ]
            pygame.draw.polygon(surf, color, stem)
        elif suit == "Diamonds":
            pygame.draw.polygon(surf, color, [
                (cx, cy - s * 0.47),
                (cx + s * 0.34, cy),
                (cx, cy + s * 0.47),
                (cx - s * 0.34, cy),
            ])
        elif suit == "Clubs":
            r = s * 0.21
            pygame.draw.circle(surf, color, (int(cx), int(cy - s * 0.20)), int(r))
            pygame.draw.circle(surf, color, (int(cx - s * 0.235), int(cy + s * 0.13)), int(r))
            pygame.draw.circle(surf, color, (int(cx + s * 0.235), int(cy + s * 0.13)), int(r))
            stem = [
                (cx - s * 0.05, cy + s * 0.05),
                (cx + s * 0.05, cy + s * 0.05),
                (cx + s * 0.19, cy + s * 0.45),
                (cx - s * 0.19, cy + s * 0.45),
            ]
            pygame.draw.polygon(surf, color, stem)

        if upside_down:
            surf = pygame.transform.rotate(surf, 180)

        pip = pygame.transform.smoothscale(surf, (size, size))
        self._pip_cache[key] = pip
        return pip

    def _index_surface(self, card: Card, H: int, color: Color, rank_ratio: float, pip_ratio: float) -> Any:
        """Índice de la carta: el número/letra con el palo justo debajo."""
        pygame = self._require_pygame()
        rank_img = self._font(int(H * rank_ratio), bold=True).render(card.rank, True, color)
        pip = self._pip_surface(card.suit, int(H * pip_ratio), color)
        gap = int(H * 0.015)

        w = max(rank_img.get_width(), pip.get_width())
        h = rank_img.get_height() + gap + pip.get_height()
        corner = pygame.Surface((w, h), pygame.SRCALPHA)
        corner.blit(rank_img, ((w - rank_img.get_width()) // 2, 0))
        corner.blit(pip, ((w - pip.get_width()) // 2, rank_img.get_height() + gap))
        return corner

    def _card_surface(self, card: Card, w: int, h: int) -> Any:
        """
        Cara de la carta: solo el índice (número + palo debajo) en las esquinas.
        El centro queda en blanco: con los pips centrales, el 10 y las cartas
        chicas se amontonaban y no se leía nada.
        """
        key = (card.rank, card.suit, w, h)
        if key in self._card_cache:
            return self._card_cache[key]

        pygame = self._require_pygame()
        ss = 2
        W, H = w * ss, h * ss
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        radius = int(min(W, H) * 0.09)

        pygame.draw.rect(surf, self.CARD_FACE, (0, 0, W, H), border_radius=radius)
        pygame.draw.rect(surf, self.CARD_EDGE, (0, 0, W, H), width=max(2, ss), border_radius=radius)

        color = self.SUIT_RED if card.suit in self.RED_SUITS else self.SUIT_BLACK
        margin_x, margin_y = int(W * 0.10), int(H * 0.055)

        if h >= 76:
            # Carta grande: índice arriba a la izquierda y repetido invertido abajo.
            corner = self._index_surface(card, H, color, rank_ratio=0.18, pip_ratio=0.125)
            surf.blit(corner, (margin_x, margin_y))
            surf.blit(
                pygame.transform.rotate(corner, 180),
                (W - margin_x - corner.get_width(), H - margin_y - corner.get_height()),
            )
        else:
            # Carta chica: un solo índice, más grande, para que se lea igual.
            corner = self._index_surface(card, H, color, rank_ratio=0.30, pip_ratio=0.21)
            surf.blit(corner, (margin_x, margin_y))

        final = pygame.transform.smoothscale(surf, (w, h))
        self._card_cache[key] = final
        return final

    def _back_surface(self, w: int, h: int) -> Any:
        key = (w, h)
        if key in self._back_cache:
            return self._back_cache[key]

        pygame = self._require_pygame()
        ss = 2
        W, H = w * ss, h * ss
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        radius = int(min(W, H) * 0.09)

        pygame.draw.rect(surf, self.CARD_FACE, (0, 0, W, H), border_radius=radius)
        inset = int(min(W, H) * 0.055)
        inner = pygame.Rect(inset, inset, W - 2 * inset, H - 2 * inset)
        pygame.draw.rect(surf, self.CARD_BACK, inner, border_radius=int(radius * 0.8))

        # Trama diagonal.
        pattern = pygame.Surface((W, H), pygame.SRCALPHA)
        step = max(6, int(min(W, H) * 0.11))
        for i in range(-H, W, step):
            pygame.draw.line(pattern, self.CARD_BACK_LINE, (i, 0), (i + H, H), max(1, ss))
            pygame.draw.line(pattern, self.CARD_BACK_LINE, (i + H, 0), (i, H), max(1, ss))
        clip = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(clip, (255, 255, 255, 255), inner, border_radius=int(radius * 0.8))
        pattern.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(pattern, (0, 0))

        pygame.draw.rect(surf, self.CARD_EDGE, (0, 0, W, H), width=max(2, ss), border_radius=radius)

        final = pygame.transform.smoothscale(surf, (w, h))
        self._back_cache[key] = final
        return final


# Alias corto por comodidad.
VisualRenderer = PodridaPygameRenderer
