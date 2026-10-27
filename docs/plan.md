# Plan del proyecto

## Plan revisado (26/10/2026)

Después de la primera noche de experimentos y de la revisión bibliográfica
(`docs/bibliografia.md`), el camino cambia en tres cosas:

- **NFSP deja de ser el algoritmo principal.** Con más de 2 jugadores no tiene garantías de
  convergencia, y en juegos de cartas de varios jugadores lo que anduvo es self-play más simple: DQN
  en Wizard (de la misma familia que la Podrida), PPO en Big 2, Deep Monte Carlo en Dou Dizhu. Se
  comparan NFSP, DQN compartido y PPO compartido con la misma evaluación (`scripts/train.py --algo`).
- **Diseño tomado de la literatura**: cada ronda como episodio, una red compartida por todos los
  asientos, y (próximo) canto y juego separados, crítico que vea todas las manos durante el
  entrenamiento y evaluación con repartos espejo.
- **PSRO pasa a una etapa posterior**: no para entrenar, sino para medir y mejorar la robustez de un
  agente que ya sea fuerte (buscar rivales que lo exploten). En Big 2, el self-play contra la
  política actual le ganó a entrenar contra un conjunto de versiones anteriores.

Las fases quedan así:

1. ~~Juego y validación~~ (hecho).
2. Elegir algoritmo y diseño con 2 jugadores, comparando con varias semillas (en curso: ver
   `docs/estado.md`).
3. Escalar a 3 y 4 jugadores con lo que mejor ande.
4. Robustez: liga de versiones anteriores o PSRO, rival PIMC más fuerte que la heurística.
5. Un modelo para n variable (2 a 7).

## Objetivo final y pendientes de diseño (27/10/2026)

**El objetivo final es ganar la partida**, no maximizar el puntaje de cada ronda. Hoy el reward es
por ronda (`relative`: puntaje de la ronda contra el promedio de los rivales), así que el agente no
tiene incentivo para arriesgar cuando va perdiendo al final ni para cuidarse cuando va ganando, y el
puntaje acumulado que ve en la observación casi no le sirve. Cuando haya una base sólida con el
reward por ronda:

- **Reward por ganar la partida**: al final, según el puesto (o mezclado con el de cada ronda para
  no perder la señal densa). La métrica que importa pasa a ser el win rate.
- **Empezar partidas en situaciones al azar**: en una ronda cualquiera de la escalera y con
  puntajes iniciales al azar, para que el agente vea muchas veces situaciones como "voy 40 abajo y
  quedan 2 rondas", que jugando desde el principio pueden ser raras. Es un parámetro del juego
  (ronda inicial y puntajes de arranque).

**Variantes de reglas que juega la gente** (para más adelante; las dos se deciden y se ven antes de
cantar, así que van en la observación):

- **No se puede cantar la misma cantidad 3 veces seguidas** (cualquier número, no solo 0): quien
  cantó 0 y 0 en las dos rondas anteriores no puede cantar 0 en esta. Impide cantar siempre 0, que
  parece lo más fácil. Si choca con la regla de que no todos pueden cumplir y no queda ningún canto
  posible (pasa poco, en general al final: no puede repetir 0 y cantar 1 haría que todos cumplan),
  **gana la de que no todos pueden cumplir**: ahí sí puede repetir. La observación tendría que
  incluir los cantos de las dos rondas anteriores de cada jugador.
- **Rondas sin triunfo**, decididas antes. La observación ya tiene un lugar para "sin triunfo" en
  el palo de triunfo; falta que el motor y el reparto lo permitan como opción.

**Identificar el estilo de cada rival durante la partida** (conservador o de cantar alto, si
cumple, si juega bien). Hoy el agente no puede: la observación solo tiene la ronda en curso y los
puntajes, y en self-play todos los rivales son él mismo, así que no hay nada que distinguir.
Haría falta: (1) estadísticas de cada rival en la observación (sesgo de sus cantos hasta ahora,
cuántas veces cumplió, cuánto canta respecto de sus cartas), o una red con memoria de la partida;
y (2) entrenar contra rivales con estilos distintos (versiones anteriores, heurísticas
conservadoras y agresivas), para que adaptarse sirva. Cuidado: adaptarse a un rival también
abre la puerta a que lo exploten a uno. "Tener suerte" no sirve para predecir.

**Resolver las rondas chicas de forma exacta** (idea, como control del RL): la ronda de 1 carta es
solo un canto (la jugada es forzada) con muy pocas situaciones distintas, así que se puede calcular
el equilibrio con CFR (OpenSpiel) y comparar con lo que aprendió el agente. Con 2 jugadores ni
hace falta: el segundo está obligado a cantar lo mismo que el primero (si el primero canta 1, el
segundo no puede cantar 0 porque sumaría 1; y al revés), y al primero le conviene cantar 1 con
cualquier carta, porque saliendo primero gana la baza con probabilidad ≥ 0.52 (un 2 que no es
triunfo) y cantar 1 conviene desde 0.5. Con 2 cartas
probablemente se pueda con MCCFR; de 3 en adelante el árbol es demasiado grande y hace falta RL.

## Plan original

### Camino concreto

#### Fase 1 — Construir y validar el juego (sin RL todavía)

1. Programar la lógica pura de la podrida (reglas, fases de apuesta, reparto de cartas, cálculo de
   puntaje) como funciones de Python separadas de cualquier interfaz.
2. Envolver esa lógica en `pyspiel.Game`/`State`, manejando `num_players()` como parámetro variable
   (2 a 7).
3. Validar jugando partidas con agentes random usando el loop estándar de OpenSpiel
   (`state.legal_actions()`, `state.apply_action()`, etc.): chequear que los puntajes, las fases de
   apuesta y el fin de partida se comporten como se espera.

#### Fase 2 — Sanity check con NFSP

4. Conectar el `Game` al ejemplo de `examples/nfsp.py`, ajustando lo mínimo indispensable.
5. Correr primero con 2-3 jugadores fijos (el caso más simple), para reducir variables y confirmar
   que el agente aprende algo mejor que random.
6. Medir algo objetivo: puntaje promedio contra agentes random o heurísticos simples (por ejemplo,
   "apostá según cuántas cartas altas tenés").

#### Fase 3 — Escalar a PSRO

7. Una vez que el pipeline de evaluación funcione (cómo se mide "qué tan bien juega" el agente),
   migrar a `psro_v2`, usando como oráculo interno algo simple al principio (DQN) para no sumar
   complejidad de golpe.
8. Empezar también con pocos jugadores (2-3) antes de escalar a 7, porque el tamaño del meta-juego
   crece con la cantidad de jugadores.
9. Comparar los resultados de PSRO contra los de NFSP con la misma métrica de la Fase 2.

#### Fase 4 — Generalizar a n variable

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
