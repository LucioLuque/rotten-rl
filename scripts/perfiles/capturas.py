"""
scripts/perfiles/capturas.py  —  Revisa la interfaz sin ventana: guarda capturas de cada mesa y
tamaño, jugando con clicks de verdad. Sirve para validar cambios visuales (docs/perfiles/).

    python scripts/perfiles/capturas.py --salida /tmp/capturas
"""

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import play  # noqa: E402
import pygame  # noqa: E402

SIZES = [(1920, 1080), (1366, 768), (1100, 650)]


def click(app, pos):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))


def shot(app, path):
    image = app.draw()
    pygame.image.save(pygame.surfarray.make_surface(image.swapaxes(0, 1)), path)


def play_match(app, rng, out, tag):
    """Juega una partida con clicks y captura cada fase de la ronda de 7 cartas."""
    taken = set()
    while not app.finished:
        match = app.match
        app.freeze_until = app.next_bot_time = 0
        game = app.game
        app.draw()
        if match.round_over:
            if len(app.history) == 7:
                shot(app, f"{out}/{tag}_fin_ronda.png")
            button = [r for r, name in app.renderer.button_hitboxes if name == "next_round"]
            click(app, button[0].center)
        elif not match.human_turn:
            app.step_bots()
        else:
            if game.cards_per_player == 7 and game.phase not in taken:
                taken.add(game.phase)
                shot(app, f"{out}/{tag}_{game.phase}.png")
            if game.phase == "bidding":
                options = [r for r, _ in app.renderer.bid_hitboxes]
            else:
                options = [r for r, _, legal in app.renderer.card_hitboxes if legal]
            click(app, options[rng.randint(len(options))].center)
    shot(app, f"{out}/{tag}_final.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", required=True)
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    rng = np.random.RandomState(0)
    for players in (2, 3, 5, 7):
        for width, height in SIZES:
            app = play.PodridaApp(play.parse_args(["--render_mode", "rgb_array",
                                                   "--jugadores", str(players)]))
            app.renderer.set_size(width, height)
            play_match(app, rng, a.salida, f"n{players}_{width}")
            print(f"{players} jugadores, {width}x{height}: ok")


if __name__ == "__main__":
    main()
