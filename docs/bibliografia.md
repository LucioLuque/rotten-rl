# Bibliografía

Trabajos sobre juegos parecidos a La Podrida (juegos de bazas con canto e información oculta) y qué
podemos tomar de cada uno. Búsqueda inicial del 25/10/2026; lo marcado como "a verificar" no lo
leímos completo.

## Lo más relevante

### Wizard con DQN — Schumacher y Pleines, IEEE CoG 2022

[arXiv 2205.13834](https://arxiv.org/abs/2205.13834). Código:
<https://github.com/jonas-schumacher/drl-pomdp>. **Leído completo.**

Wizard es de la misma familia que la Podrida: 4 jugadores, 15 rondas de 1 a 15 cartas, cada jugador
canta cuántas bazas va a hacer; si acierta suma 20 + 10·bazas, si no resta 10 por cada baza de
diferencia.

- **DQN puro en self-play** (no NFSP), y funciona: 66–87% de acierto del canto en self-play (más
  alto en las rondas con pocas o muchas cartas, más bajo en las del medio), contra ~40–50% de una
  heurística y mucho menos de un agente al azar. Generaliza a rivales nunca vistos (la heurística).
- **Cada ronda es un episodio independiente**: el objetivo es maximizar los puntos de la ronda, no
  ganar la partida. Verifican que eso alcanza para ganar partidas: un DQN contra 3 heurísticas gana
  el 80% de las partidas completas.
- **Una red compartida por los 4 jugadores**, con un solo replay buffer: todos los asientos aportan
  experiencia a la misma red.
- **Redes separadas para canto y juego** (el canto es un MDP de un paso; el juego, de varios), y una
  por cada tamaño de ronda (30 redes en total). Entradas one-hot; acciones ilegales enmascaradas.
- Hiperparámetros: γ = 1, ε de 1.0 a 0.01 a lo largo del 90% del entrenamiento, replay buffer de
  300k (canto) y 600k (juego), batch de 1024, lr 0.0005, Adam, 3 capas ocultas (100-100-50 para el
  canto, 200-200-100 para el juego). Unas 400k rondas de entrenamiento por tamaño de ronda.
- **Lo que no sirvió**: agregar la historia de cartas con un LSTM, y una búsqueda en árbol sobre
  estados muestreados (PIMC). Ninguna superó al DQN básico.
- Observaciones: cantar más tarde da ventaja (se conocen los cantos previos), y el juego no es de
  suma cero: a veces conviene ayudar a otros a cumplir.

### DouZero — Zha et al., ICML 2021

[arXiv 2106.06135](https://arxiv.org/abs/2106.06135). Dou Dizhu (juego chino de 3 jugadores).
**Leído el abstract y el resumen del método.**

- **Deep Monte Carlo (DMC)**: aprende Q(estado, acción) con regresión al retorno Monte Carlo de la
  partida (sin bootstrapping como en DQN), con muchos actores en paralelo generando partidas.
- Primer lugar entre 344 agentes en el ranking de Botzone, entrenado desde cero en un servidor con 4
  GPUs. Codifica las acciones como entrada de la red (útil cuando hay muchas acciones).
- Seguimientos: [DouZero+](https://arxiv.org/abs/2204.02558) (modelado de rivales) y
  [AlphaDou](https://arxiv.org/abs/2407.10279) (agrega la fase de canto, que DouZero no aprendía).

### PIMC y búsqueda en juegos de bazas (Skat) — Buro, Long, Sturtevant y otros

- Buro et al., IJCAI 2009: primer jugador de Skat a nivel experto, con evaluación aprendida de
  partidas humanas e inferencia de las manos rivales
  ([paper](https://www.ijcai.org/Abstract/09/236)).
- Long et al., AAAI 2010: por qué funciona PIMC (muestrear manos rivales y resolver cada una como
  información perfecta) y cuándo falla (*strategy fusion*, *non-locality*). A verificar.
- [Rebstock, Solinas, Buro y Sturtevant, 2019](https://arxiv.org/abs/1905.10911): inferencia de
  manos con políticas aprendidas. PIMC e ISMCTS son el estado del arte de búsqueda en varios juegos
  de bazas. A verificar.
- [Solinas et al., 2019](https://arxiv.org/abs/1903.09604): una red predice dónde está cada carta
  para muestrear mejor en PIMC; mejora mucho el juego de cartas en Skat. A verificar.

En Wizard, la búsqueda sobre estados muestreados no ayudó; puede valer más como rival fuerte de
evaluación que como parte del agente.

## Otros

- [Bidding in Spades](https://arxiv.org/abs/1912.11323) (Cohensius et al.): canto con simulación
  Monte Carlo y heurísticas, más aprendizaje supervisado sobre millones de partidas reales; le gana
  al 56% de los humanos. No es RL.
- Bridge: [Joint Policy Search](https://arxiv.org/abs/2008.06495) (Tian, Gong y Jiang, NeurIPS
  2020): self-play para la subasta, le gana a WBridge5. Juego cooperativo por equipos; menos
  aplicable.
- [RLCard](https://arxiv.org/abs/1910.04376): toolkit de RL para juegos de cartas (no incluye Oh
  Hell).
- Batak (Baykal y Alpaslan, UBMK 2019): aprendizaje supervisado para el canto y Monte Carlo RL para
  el juego. A verificar.
- [Oh, Hell! Card Playing AI Bot](https://cis.temple.edu/~pwang/5603-AI/Project/2021S/Amend_Zach/Amend_Zach_FinalReport.pdf)
  (proyecto de estudiantes, Temple 2021). A verificar.
- No encontramos ningún trabajo de RL publicado específicamente sobre Oh Hell / La Podrida.

## Qué podemos tomar

Ordenado por lo que esperamos que rinda y lo poco que cuesta:

1. **Cada ronda como episodio.** Hoy el episodio es la partida entera con γ = 1: el valor de cantar
   en la ronda 1 incluye los puntos de las 12 rondas siguientes, que no dependen de esa decisión y
   solo agregan ruido. Cortar el episodio en cada ronda baja mucho la varianza. Es un cambio chico en
   `EpisodeRunner`.
2. **Una red compartida por todos los asientos** (con el asiento como parte de la observación, que
   ya está en posiciones relativas). Con n jugadores, cada red recibe n veces más datos. También
   simplifica jugar muchas partidas a la vez (ver `docs/optimizacion.md`).
3. **DQN puro en self-play** como alternativa a NFSP: en Wizard funciona, y en nuestro diagnóstico
   (`diag_dqn2p`) un DQN contra rivales al azar ya llegaba a 0.44 vs heurística en 20k episodios.
4. **Separar canto y juego** en dos redes (o dos cabezas): son problemas distintos, y puede ayudar
   con el problema de cantar de más.
5. **Buffers y batches más grandes** (300k–600k, batch 1024), que en Wizard fueron "vitales".
6. **Deep Monte Carlo** (DouZero) si DQN se estanca: más simple y muy paralelizable.
7. Evaluar también contra un rival de búsqueda (PIMC) cuando la heurística quede chica.
