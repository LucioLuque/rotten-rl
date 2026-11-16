# Búsqueda: PIMC

Etapa A del plan del 16/11 (`docs/plan.md`): medir mejor. La heurística ya no distingue entre agentes
buenos y no sabemos cuántas de las partidas que pierde el agente son mala suerte y cuántas errores.
PIMC sirve para las dos cosas: como **rival más fuerte** y como **analizador de decisiones**. Más
adelante (etapa G), como jugador.

## Qué es PIMC

*Perfect Information Monte Carlo*. En cada decisión, el jugador no ve las cartas de los demás. PIMC:

1. **Inventa** muchos repartos posibles de las cartas que no ve, coherentes con lo que sabe: sus
   cartas, las que ya salieron, el triunfo, cuántas cartas tiene cada uno y de qué palos se sabe que
   no tiene (porque no siguió el palo).
2. Para cada reparto inventado y cada jugada legal, **simula el resto de la ronda** con la red de PPO
   en todos los asientos.
3. Juega la jugada que **en promedio** le dio más puntos.

Es lo que usan los mejores programas de bridge y de Skat. Límites conocidos: no deduce nada de los
cantos (solo usa los palos que faltan) y en cada reparto inventado juega como si ese fuera el real
("strategy fusion"). En Wizard la búsqueda dentro del agente no lo mejoró (`docs/bibliografia.md`),
por eso primero va como rival y analizador.

## Código

- `rotten_rl/search/pimc.py`: `sample_hands` (reparto coherente; rechaza los que le dan a alguien un
  palo que se sabe que no tiene, hasta 32 intentos), `rollout_round` (juega hasta el final de la ronda
  con la red), `action_values` (valor de cada jugada legal, como máximo 8) y `pimc_action`. Todo en
  JAX, vectorizado sobre repartos y jugadas.
- `tests/test_search_pimc.py`: los repartos respetan lo que sabe el jugador, la simulación coincide
  con un loop común y los valores cubren exactamente las jugadas legales (6 tests, pasan en la -14).
- `scripts/search/analyze_decisions.py`:
  - `--mode analyze`: el agente juega contra sí mismo y en cada decisión PIMC estima cuánto valía
    cada jugada; reporta cuánto deja el agente contra la mejor según PIMC, por fase (canto o juego) y
    tamaño de mano, y cuánto coinciden.
  - `--mode play`: PIMC en un asiento y la red en los demás, contra la red en ese asiento con los
    mismos repartos: ¿PIMC le gana a la red?

## Resultados

Pendiente: primera corrida en la -14 (16/11 a la noche).
