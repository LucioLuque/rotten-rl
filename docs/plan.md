# Plan del proyecto

## Camino concreto

### Fase 1 — Construir y validar el juego (sin RL todavía)

1. Programar la lógica pura de la podrida (reglas, fases de apuesta, reparto de cartas, cálculo de
   puntaje) como funciones de Python separadas de cualquier interfaz.
2. Envolver esa lógica en `pyspiel.Game`/`State`, manejando `num_players()` como parámetro variable
   (2 a 7).
3. Validar jugando partidas con agentes random usando el loop estándar de OpenSpiel
   (`state.legal_actions()`, `state.apply_action()`, etc.): chequear que los puntajes, las fases de
   apuesta y el fin de partida se comporten como se espera.

### Fase 2 — Sanity check con NFSP

4. Conectar el `Game` al ejemplo de `examples/nfsp.py`, ajustando lo mínimo indispensable.
5. Correr primero con 2-3 jugadores fijos (el caso más simple), para reducir variables y confirmar
   que el agente aprende algo mejor que random.
6. Medir algo objetivo: puntaje promedio contra agentes random o heurísticos simples (por ejemplo,
   "apostá según cuántas cartas altas tenés").

### Fase 3 — Escalar a PSRO

7. Una vez que el pipeline de evaluación funcione (cómo se mide "qué tan bien juega" el agente),
   migrar a `psro_v2`, usando como oráculo interno algo simple al principio (DQN) para no sumar
   complejidad de golpe.
8. Empezar también con pocos jugadores (2-3) antes de escalar a 7, porque el tamaño del meta-juego
   crece con la cantidad de jugadores.
9. Comparar los resultados de PSRO contra los de NFSP con la misma métrica de la Fase 2.

### Fase 4 — Generalizar a n variable

10. Una vez que el pipeline ande bien con un número fijo de jugadores, extender el entrenamiento a
    que varíe entre 2 y 7 (se puede entrenar un modelo por cada n, o intentar arquitecturas que
    generalicen; esto último es más ambicioso y puede quedar para el final).

Es un camino incremental: primero se valida que el entorno está bien hecho, después se consigue un
resultado rápido con NFSP para tener algo funcionando, y por último se sube a PSRO, que es la
herramienta correcta para el problema real (n jugadores, suma general), sin haber invertido tiempo
en la parte más pesada antes de saber que el resto del pipeline funciona.

## Observación (V1)

Todo se codifica en posiciones relativas al agente y con padding hasta 7 jugadores, así el tamaño
no cambia con n.

| Componente | Tamaño | Explicación |
|---|---|---|
| Hand size | 7 | Cuántas cartas se reparten (h). One-hot. |
| Round index | 13 | En qué ronda está. |
| Cards remaining | 7 | Cuántas cartas le quedan a cada jugador, normalizado por h. Es un poco redundante (se podría calcular), pero son solo 7. |
| Fase | 2 | One-hot: bidding / playing. |
| Triunfo | 5 | Qué palo es el triunfo, más "no hay triunfo". |
| Palo de salida | 5 | One-hot: 4 palos + "todavía no se jugó ninguna carta en esta baza". |
| Active players | 7 | Máscara binaria: 1 por cada posición relativa que corresponde a un jugador real, 0 para padding. |
| Dealer relativo | 7 | One-hot de la posición del dealer. |
| Is last bidder | 1 | Si el agente es el último en cantar. |
| Bids | 7 × 9 = 63 | Para cada jugador, one-hot entre NOT_BID, 0, …, 7. |
| Tricks won | 7 | Bazas ganadas por cada jugador, normalizadas por h. |
| Void suits | 7 × 4 = 28 | Para cada jugador y palo: 1 si se sabe que ese jugador no tiene cartas de ese palo. |
| Scores | 7 | Puntaje acumulado de cada jugador, normalizado por el máximo teórico de la partida. |
| Bid minus tricks | 7 | (bid − tricks) / h para cada jugador: cuánto le falta, si llegó o si se pasó. |
| Información de las cartas | 52 × 10 = 520 | Para cada carta, one-hot de dónde está: no la vi · en mi mano · es la carta de triunfo · la jugó el jugador 0 … 6. |
| Cartas de la baza en curso | 52 | Flag de las cartas jugadas que pertenecen a esta baza. |
| Quién va ganando la baza | 7 | One-hot de posición relativa. |
| Cartas jugadas de cada palo | 4 | Normalizadas por 13. |
| Cartas ya vistas de cada palo | 4 × 13 = 52 | One-hot. |
| Jugadores que faltan en esta baza | 7 | One-hot de cuántos juegan después del agente. Se deduce fácil, pero es barato. |
| **Total** | **808** | |

## Recompensa

- Episodio = partida completa.
- Recompensa en decisiones intermedias = 0.
- Recompensa al terminar cada ronda:

  ```python
  if tricks_won == bid:
      score = 10 + 2 * tricks_won
  else:
      score = tricks_won

  reward = score / 24
  ```

- Variante a probar (`reward_scheme=relative`):
  `reward = (score - mean(score_oponentes)) / 24`

## Métricas

| Métrica | Definición | Qué indica |
|---|---|---|
| Puntaje medio por ronda | E[S] / (10 + 2h) | Puntaje normalizado por el máximo de la ronda. |
| Diferencia media de puntaje | E[S_i − S̄_opp] | Ventaja sobre el promedio de los rivales. |
| Win rate | # partidas ganadas / # partidas | Con n jugadores, empatar con los rivales da 1/n. |
| Exact bid rate | # rondas con tricks = bid / # rondas | Debería subir. |
| Error absoluto del canto | MAE_bid = E[\|bid − tricks\|] | Debería bajar. Probablemente la métrica más natural del proyecto; no reemplaza puntaje ni win rate, pero ayuda a interpretar qué aprendió. |
| Sesgo del canto | E[bid − tricks], con signo | El MAE no distingue al que sobrecanta del que subcanta, y son patologías distintas con arreglos distintos. |
| Calibración | E[tricks \| bid = b] para cada b | Distingue si el problema es que canta mal o que juega mal las cartas después de cantar bien. |
