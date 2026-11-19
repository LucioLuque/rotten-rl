# Plan del proyecto

## Plan desde el 16/11/2026: medir mejor, mejorar la receta, escalar

Al 16/11, `s20`, `s30` y `u72` (2, 3 y 7 jugadores) están en meseta con PPO y self-play; no la
pasaron ni el premio por ganar la partida ni la red de 1024 (`docs/match_value/`, `docs/net_size/`).
La heurística ya no distingue entre agentes buenos (todos le ganan igual). La liga sobre un agente
entrenado cierra un agujero de ~1 punto con 2 jugadores; con 3 y 7 está en estudio
(`docs/robustness/`). La idea del plan: **primero medir mejor, después mejorar, después escalar**.
Duraciones estimadas, según las noches de lab que haya.

| Etapa | Qué | Por qué en ese lugar | Duración |
|---|---|---|---|
| **A. Medir** | Explotadores de 3 y 7 a 48M (`queue24`). Comparar `s20`, `s30` y `u72` contra lo exacto (ronda 1 con los tres; ronda 2 con 3 jugadores; correr el solver de la ronda 2 con 2 jugadores, que nunca se corrió). PIMC como rival y como analizador de errores (en las partidas perdidas, ¿había una jugada claramente mejor?). Panel de rivales: nuestros mejores agentes, explotadores, PIMC y mi perfil cuando tenga partidas | Sin mejor medición no se sabe si lo que sigue mejora algo | 1–2 semanas |
| **B. Mejorar la receta** (con 2, 3 y 7) | Pérdida auxiliar que prediga las manos ajenas; canto y juego en cabezas separadas; liga con el que aprende en varios asientos; quizás MMD | Cada mejora habría que reentrenarla en todas las n: conviene encontrarlas con pocas | 2–3 semanas |
| **C. 4, 5 y 6 jugadores** | Con la mejor receta de B; cada agente nuevo, contra lo exacto donde exista | Para no entrenarlos dos veces | ~1 semana |
| **D. Variantes de reglas** | No cantar lo mismo 3 veces seguidas; rondas sin triunfo | Cambian el juego y obligan a reentrenar: con la receta fija | 1–2 semanas |
| **E. Modelo único de 2 a 7** | Contra los específicos de cada n; vale si pierde poco (por ejemplo, menos de 0,3 por partida) | Tiene que incluir las variantes | 1–2 semanas |
| **F. Eficiencia** | Red más chica (256): cuánta velocidad gana sin perder calidad | Cuando lo principal esté resuelto | días |
| **G. Búsqueda al jugar** | PIMC como jugador, sobre el modelo final | Se apoya en la mejor red | abierto |
| **H. Más adelante** | Identificar el estilo de cada rival; jugar contra personas | Necesita todo lo anterior | — |

Lo exacto de la ronda 2 con 5 a 7 jugadores (agrupación C, `docs/segunda-ronda/`) entra en C solo si
hace falta como control. El plan del 9/11 (abajo) queda como historia.

### Estado al 17/11

Regla: el plan se sigue en orden; si algo se saltea, se reordena o se descarta, lo decido yo y queda
anotado acá con fecha y motivo.

**A. Medir**, casi terminada:
- Hecho: explotadores de 3 y 7 a 48M (no les faltaba tiempo; `docs/robustness/`); ronda 1 contra el
  equilibrio con 2, 3 y 7 y ronda 2 con 2 y 3 (`docs/primera-ronda/`, `docs/segunda-ronda/`); PIMC
  como rival y analizador (juega peor que la red: descartado como rival, `docs/search/`).
- Pendiente: panel fijo de rivales (va mientras corre la B, no ocupa el lab); mi perfil (necesita
  partidas mías).

**B. Mejorar la receta**, en curso:
- Pérdida auxiliar de las manos ajenas: hecha, no mejora (`docs/aux_hands/`).
- Liga con el que aprende en varios asientos: hecha, no mejora ni con 3 ni con 7 (empata con seguir
  sin liga; con 3 no lo hace más robusto y con 7 no hay agujero que tapar; `docs/robustness/`).
- Canto y juego en redes separadas: hecho, no mejora (`h20` contra `s20full` −0,28, 4/16, 18/11;
  `docs/split_policy/`).
  Prueba extra con 3 jugadores (`h30` contra `s30full`, decidida el 18/11): empate (−0,03, 8/16).
- ~~MMD~~: descartado por ahora (17/11): más caro y sin un resultado en juegos parecidos que lo
  justifique. Se puede probar más adelante.
- También el 17/11: `s20full`, `s30full` y `u72full` (2, 3 y 7 de cero a 24M de una vez), porque los
  agentes de 2 y 7 que tenía estaban hechos en tramos retomados. `s30full` empató con `s30`: retomar
  con `--resume` equivale a entrenar de una vez (`docs/resume/`), así que no hacía falta rehacerlos;
  `s20full` y `u72full` quedan igual como base de cada prueba de la B (misma receta, de cero, 24M).

**B terminada el 18/11 a las 02:20:** ninguna de sus tres ideas mejoró (MMD descartado), así que la
receta queda como estaba.

**C. 4, 5 y 6 jugadores**: empieza el 18/11 con la receta actual: `f40`, `f50`, `f60` en la cola de la
-14 después de `u72full` (`docs/players_456/`). El 17/11 la había empezado antes de tiempo y la saqué.

## Plan desde el 9/11/2026: de ganar rondas a ganar partidas

Con la receta nueva de PPO en JAX, 2, 3 y 7 jugadores se acercan a la meseta con el reward por
ronda. Ese reward (puntaje de la ronda contra el promedio de los rivales) es una aproximación de
lo que importa, ganar la partida: no mira al rival que va primero ni la situación de la partida.
Además, como cada ronda es un episodio, el puntaje acumulado que el agente ve en la observación no
le sirve de nada (le da lo mismo ir 40 arriba o 40 abajo). Decidido el 9/11:

1. **Terminar las extensiones a 24M** de `s20`, `s30` y `u72` (2, 3 y 7 jugadores) con sus torneos
   contra 12M. No seguir a 48M con el reward por ronda.
2. **Objetivo de partida en el motor JAX:** la partida entera como episodio (el GAE no corta al
   cerrar cada ronda) y reward = el de cada ronda + λ × premio final (ganar como premio principal
   y un poco por puesto; solo premio final sería demasiado ruidoso). Partidas que arrancan en una
   ronda y con puntajes al azar, para ver seguido finales ajustados. La observación no cambia (ya
   tiene puntajes y ronda). Los torneos ya miden win rate, que pasa a ser la métrica principal.
3. **Experimento con 2 jugadores, currículo contra directo**, a igual cantidad de partidas (36M):
   - A: `s20` (24M con reward por ronda) + 12M con el objetivo nuevo, λ = 0, 0,5 y 1 (λ = 0, con la
     partida como episodio, es el control);
   - B: de cero con el objetivo nuevo (λ = 0,5 y 1), 36M seguidas.
   A contra B dice si conviene aprender primero a ganar rondas y después partidas o ir directo;
   la curva de B, si lo alcanza antes; B contra `s20`, cuánto rinde el premio de partida desde el
   principio. Después, 3 y 7 jugadores.
4. **4, 5 y 6 jugadores directamente con el objetivo nuevo** (cambia el orden del 30/10, para no
   entrenarlos dos veces).
5. **Robustez:** medir cuánto se le puede sacar al agente (entrenar uno que solo busque ganarle a
   una versión congelada); si tiene agujeros grandes, entrenar contra versiones anteriores.

Ahora no: PSRO completo (sin garantías con más de 2 jugadores, caro, en Big 2 no le ganó al
self-play; el punto 5 da lo útil), 48M con el reward por ronda, solo premio por ganar, búsqueda al
jugar (mucho código; va al final, sobre una red ya buena), ajustes chicos de la receta (menores
que el ruido de ~0,3; red más grande y entropía decreciente quedan para después del objetivo
nuevo) y cambios en la observación.

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
