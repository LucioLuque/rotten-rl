# Bibliografía

Trabajos sobre juegos parecidos a La Podrida (juegos de bazas con canto, información oculta, varios
jugadores) y sobre los métodos que nos importan. Búsqueda del 25/10/2026, a partir de una lista
inicial mía más búsquedas propias.

Para cada trabajo: **relevancia** (alta / media / baja) y **qué leímos** (completo, o solo
resumen/abstract). Lo que solo leímos por el abstract puede tener matices que no vimos.

La síntesis (qué nos dice la literatura y qué probar) está al final: [Qué nos dice](#qué-nos-dice-para-este-proyecto).

## 1. Oh Hell y juegos de la misma familia (canto de bazas)

### Wizard con DQN — Schumacher y Pleines, IEEE CoG 2022 · relevancia alta · leído completo

[arXiv 2205.13834](https://arxiv.org/abs/2205.13834) · código:
[jonas-schumacher/drl-pomdp](https://github.com/jonas-schumacher/drl-pomdp)

Wizard es de la misma familia: 4 jugadores, 15 rondas de 1 a 15 cartas, se canta cuántas bazas se
van a hacer; acertar da 20 + 10·bazas y errar resta 10 por baza de diferencia.

- **DQN puro en self-play** (no NFSP) funciona: 66–87% de acierto del canto en self-play (alto en
  rondas con pocas o muchas cartas, más bajo en las del medio), contra ~40–50% de una heurística.
  Generaliza a rivales que nunca vio.
- **Cada ronda es un episodio independiente**: maximizan los puntos de la ronda, no la victoria.
  Verifican que alcanza: un DQN contra 3 heurísticas gana el 80% de las partidas completas.
- **Una red compartida por los 4 jugadores** y un solo replay buffer.
- **Redes separadas para canto y juego** (el canto es un MDP de un paso) y una por tamaño de ronda
  (30 redes). Entradas one-hot, acciones ilegales enmascaradas.
- Hiperparámetros: γ = 1; ε de 1.0 a 0.01 en el 90% del entrenamiento; replay de 300k (canto) y
  600k (juego); batch 1024; lr 0.0005; Adam; 3 capas (100-100-50 canto, 200-200-100 juego). Dicen
  que un buffer y un batch grandes fueron "vitales". ~400k rondas por tamaño de ronda.
- **No sirvió**: agregar la historia de cartas con un LSTM, ni búsqueda sobre estados muestreados.
- Cantar más tarde da ventaja (se conocen los cantos previos). El juego no es de suma cero.

### History Filtering in Imperfect Information Games — Solinas, Rebstock, Sturtevant y Buro, 2023 · relevancia media · leído el HTML (experimentos)

[arXiv 2311.14651](https://arxiv.org/abs/2311.14651)

- **Usa Oh Hell** (3 jugadores, mazos reducidos) para experimentos de teoría de juegos: cómo
  muestrear historias (manos posibles de los rivales) consistentes con lo observado, con un sampler
  de Gibbs (MCMC). Enumerar es intratable: con 52 cartas hay más de 10^61 historias posibles.
- Con ~400 muestras aproximan bien el valor de un estado público que tiene 544k historias.
- No entrena agentes fuertes ni mide fuerza de juego; las políticas son de Q-learning
  independiente. Sirve si alguna vez hacemos búsqueda o inferencia de manos.

### OhHellProject — anyarko (GitHub) · relevancia baja · leído el README

[github.com/anyarko/OhHellProject](https://github.com/anyarko/OhHellProject)

Agente de RL para Oh Hell sobre la estructura de RLCard, con PPO (implementación propia en base a
Stable-Baselines3) y la idea de self-play de NFSP. Versión simplificada: una sola ronda de 10
cartas. Sin resultados publicados.

### Judgement (Kachufool) — Josh Holder · relevancia media · leída la página

[josh-holder.github.io/projects/featured/judgement](https://josh-holder.github.io/projects/featured/judgement/)

Misma mecánica (cantar y cumplir exacto). Proyecto personal con poca compute:

- Modelos separados: canto (supervisado con ~miles de cantos humanos), y juego (imitación de una
  heurística y después RL en self-play).
- Lo que no anduvo: DQN necesitaba un replay más grande que su memoria; retornos Monte Carlo dieron
  explosión de gradientes; A3C estabilizó pero dejó de mejorar. Le ganó 1 de 3 partidas al autor.
- Lección: con poca compute y sin infraestructura, el RL se estanca sin dar señales de por qué.

### Wizard con RL — Backhus et al., IEEE GCCE 2013 · relevancia baja · solo metadatos

[DOI 10.1109/GCCE.2013.6664846](https://doi.org/10.1109/GCCE.2013.6664846) (acceso cerrado).
Canto con un agente de reglas o una red supervisada (aproxima las bazas hechas) y juego con RL
basado en modelo. Antecedente del paper de 2022.

### Otros de la familia

- **Bidding in Spades** — Cohensius et al.
  ([arXiv 1912.11323](https://arxiv.org/abs/1912.11323)) · media · abstract. Canto con simulación
  Monte Carlo + heurísticas + aprendizaje supervisado sobre millones de partidas reales; le gana al
  56% de los humanos. Muestra que el canto se puede tratar como **predicción** de bazas.
- **Batak** — Baykal y Alpaslan, UBMK 2019 · baja · citado en Wizard: supervisado para el canto y
  Monte Carlo RL para el juego.
- **Thousand Schnapsen** — [Applied Sciences 15(4):2121, 2025](https://www.mdpi.com/2076-3417/15/4/2121)
  · baja · abstract (la página bloquea la lectura). Solo la fase de bazas, con un REINFORCE
  modificado que pesa más las partidas de mayor puntaje, más experience replay con importance
  weighting. Gana >65% a random y ~55–60% a alpha-beta.

## 2. Otros juegos de cartas con RL a gran escala

### DouZero — Zha et al., ICML 2021 · relevancia alta · abstract y resumen del método

[arXiv 2106.06135](https://arxiv.org/abs/2106.06135) (Dou Dizhu, 3 jugadores)

- **Deep Monte Carlo (DMC)**: regresión de Q(estado, acción) al retorno Monte Carlo, sin
  bootstrapping, con muchos actores en paralelo. Le ganó a DQN, A3C y otros; primer lugar entre 344
  agentes en Botzone, entrenado desde cero en un servidor con 4 GPUs.
- Seguimientos: [DouZero+](https://arxiv.org/abs/2204.02558) (modelado de rivales) y
  [AlphaDou](https://arxiv.org/abs/2407.10279) (agrega la fase de canto, que DouZero resolvía con
  una red supervisada).

### Big 2 con PPO — Charlesworth 2018, y Patwa 2026 · relevancia alta · abstracts

- [Charlesworth, arXiv 1808.10442](https://arxiv.org/abs/1808.10442): PPO en self-play puro, 4
  jugadores; le gana a humanos amateurs con poco entrenamiento y sin búsqueda.
- [Patwa, arXiv 2605.28863 (2026)](https://arxiv.org/abs/2605.28863): comparación controlada
  (mismo entorno, representación y presupuesto). **PPO le gana a Monte Carlo Q, SARSA y
  Q-learning**; una regularización de entropía moderada ayuda; y **self-play contra la política
  actual** es mejor currículum que contra checkpoints o rivales fijos.

### MAPPO e IPPO — Yu, Velu, Vinitsky et al., NeurIPS 2022 · relevancia media · abstract y resúmenes

[arXiv 2103.01955](https://arxiv.org/abs/2103.01955) ·
[post de BAIR](https://bair.berkeley.edu/blog/2021/07/14/mappo)

- **IPPO**: PPO donde la política y el crítico usan solo la observación propia. **MAPPO**: el
  **crítico es centralizado** y ve información global (el estado completo), mientras que la política
  usa solo la observación propia (*centralized training, decentralized execution*). El crítico solo
  sirve para bajar la varianza durante el entrenamiento, así que puede ver lo que el jugador no ve.
- Pensado para juegos **cooperativos** con reward compartido (Hanabi, StarCraft). Con poco ajuste
  rinde como los mejores métodos. Recomiendan pocas épocas por lote (5–15) y un clipping chico.
- Para nosotros (competitivo, suma general) no se usa tal cual, pero **la idea del crítico con
  información privilegiada sí aplica**: un crítico que vea todas las manos durante el entrenamiento
  baja la varianza sin que la política haga trampa al jugar. Es la misma idea que el crítico
  centralizado de VRPO (abajo), el oracle de Suphx y el entrenamiento con información perfecta de
  ScrofaZero.
- **Lo probamos** (`--ppo_critic central`, 29/10 a 4/11): empata con el crítico local. Con 3
  jugadores, +0.4 puntos por partida a ~2 errores estándar (torneo del 30/10); con 7, 0.337 contra
  ~0.33 contra la heurística. Ver abajo por qué era esperable.

### Cuándo ayuda un crítico centralizado — varios trabajos · relevancia media · abstracts y resúmenes

- **Lyu, Xiao, Daley y Amato, AAMAS 2021** (*Contrasting Centralized and Decentralized Critics in
  Multi-Agent RL*, [arXiv 2102.04402](https://arxiv.org/abs/2102.04402)): el crítico centralizado da
  el **mismo gradiente esperado** que el local y **no garantiza menos varianza**; puede tener más. En
  un trabajo posterior (AAAI 2022, *A Deeper Understanding of State-Based Critics*) muestran que un
  crítico que ve el estado completo **puede sesgar** el gradiente con información parcial: la
  política no puede usar lo que ve el crítico.
- **de Witt et al. 2020** (*Is Independent Learning All You Need in the StarCraft Multi-Agent
  Challenge?*, [arXiv 2011.09533](https://arxiv.org/abs/2011.09533)): IPPO, sin nada centralizado,
  rinde como los métodos centralizados incluso en un juego cooperativo. El paper de MAPPO también
  encuentra a IPPO competitivo en muchos escenarios.
- **PerfectDou** — Yang et al., NeurIPS 2022 ([arXiv 2203.16406](https://arxiv.org/abs/2203.16406)):
  el contraejemplo en cartas competitivas. PPO con un crítico que ve todas las manos
  (*perfect-training-imperfect-execution*) le gana a DouZero en Dou Dizhu, con mucho trabajo de
  features. Las partidas de Dou Dizhu son largas y la información oculta pesa todo el episodio.
- **Para nosotros**: lo que más resuelve MAPPO (repartir el mérito de un reward compartido) no
  existe en un juego competitivo con reward propio. Solo queda la baja de varianza, que es chica
  con episodios de una ronda y reward relativo. El cuello de botella está en otro lado (alargar el
  horizonte dio +2.2 contra +0.4). Una forma más directa de usar la información oculta es una
  **pérdida auxiliar que prediga las manos ajenas** (Suphx, PerfectDou y los trabajos de Skat de la
  sección 3).

### GAE en self-play con información imperfecta — Fan y Farina, 2026 · relevancia media · abstract

[arXiv 2605.19235](https://arxiv.org/abs/2605.19235). En PPO con self-play, el estimador de ventaja
GAE tiene varianza extra por muestrear las acciones futuras (aunque el crítico sea exacto).
Proponen VRPO, con un crítico de acción-valor centralizado y un Expected SARSA(λ). Probado en Dou
Dizhu y póker. A tener en cuenta si usamos PPO.

### ScrofaZero (Gongzhu) — Shi, Li y Sun, 2021 · relevancia media · leído el HTML

[arXiv 2102.07495](https://arxiv.org/abs/2102.07495)

- Estilo AlphaZero: self-play con MCTS y red de política y valor. **Entrena con información perfecta**
  (todas las manos visibles), que es más estable y rápido, y en el juego real **muestrea las manos
  ocultas** (muestreo estratificado + importance weighting + agrupar cartas irrelevantes).
- Nivel experto humano según sus evaluaciones; dos GPUs 2080Ti.

### Suphx (Mahjong) — Li et al., Microsoft 2020 · relevancia media · resumen

[arXiv 2003.13590](https://arxiv.org/abs/2003.13590)

- El mismo problema que nuestro "ronda vs. partida": el resultado final depende de varias rondas.
  Usan **predicción del reward global**: una red estima el resultado final de la partida desde la
  ronda actual, y eso reparte el crédito entre rondas.
- **Oracle guiding**: primero entrena viendo la información oculta y se la va quitando de a poco.
- Mejor que el 99.99% de los humanos rankeados en Tenhou.

### Otros

- [RLCard](https://arxiv.org/abs/1910.04376) · media · abstract y resúmenes: toolkit de RL para
  juegos de cartas (sin Oh Hell). En sus pruebas, **NFSP y DQN fueron muy inestables en juegos
  grandes** (UNO, Mahjong, Dou Dizhu).
- **Decision Transformers para Hearts, Whist y Spades** — Demirdöver, tesis de doctorado, METU 2026
  ([página](https://ceng.metu.edu.tr/en/unified-sequence-modeling-framework-imperfect-information-trick-taking-card-games))
  · baja: aprendizaje offline sobre partidas de agentes random y de reglas, y después iteración de
  experto. Interesante a futuro, no ahora.
- **Schnapsen con RL "shallow"** — Klačan y Zhang, 2026
  ([arXiv 2605.17162](https://arxiv.org/abs/2605.17162)) · baja: imitar partidas no alcanza para
  ganarle a un rival de búsqueda; RL con actualizaciones Monte Carlo sí, y combinar el valor
  aprendido con búsqueda mejora más.
- **SkyNet (Skyjo)** — Haile, ICANN 2026 ([arXiv 2603.27751](https://arxiv.org/abs/2603.27751)) ·
  baja: MuZero con cabezas auxiliares (predecir ganador y posición) y un currículum de rivales
  heurísticos + self-play.

## 3. Búsqueda en juegos de bazas (PIMC y afines)

- **GIB (bridge)** — Ginsberg, JAIR 2001 ([arXiv 1106.0669](https://arxiv.org/abs/1106.0669)) ·
  media · abstract. Muestreo Monte Carlo de las manos ocultas + resolver cada una con información
  perfecta ("double dummy"). Nivel experto en 2001. Sugiere que sirve para juegos de bazas, y que el
  póker necesita otra cosa.
- **Skat** — [Buro et al., IJCAI 2009](https://www.ijcai.org/Abstract/09/236) (primer jugador a
  nivel experto, con inferencia de manos rivales); Long et al., AAAI 2010 (cuándo funciona PIMC:
  errores de *strategy fusion* y *non-locality*);
  [Rebstock et al. 2019](https://arxiv.org/abs/1905.10911) y
  [Solinas et al. 2019](https://arxiv.org/abs/1903.09604) (redes que predicen dónde está cada carta
  para muestrear mejor). Media · abstracts. PIMC e ISMCTS son el estado del arte de búsqueda en
  varios juegos de bazas.
- **Outer-learning en Skat** — Edelkamp, 2025 ([arXiv 2512.15435](https://arxiv.org/abs/2512.15435))
  · media · abstract: las **decisiones tempranas (canto, elección de juego) pesan más en el
  resultado que el juego fino del medio y del final**. Amplía bases de partidas humanas con
  millones de partidas de self-play.

En Wizard, la búsqueda sobre estados muestreados no ayudó al agente. Puede valer más como **rival
fuerte de evaluación** cuando la heurística quede chica.

## 4. Teoría: varios jugadores y suma general

- NFSP con más de 2 jugadores: **no tiene garantías de convergencia**; empíricamente da
  estrategias mucho menos explotables que random en póker simple de 3+ jugadores
  ([capítulo de Springer](https://link.springer.com/10.1007/978-3-319-75931-9_5)) · abstract.
- [Ge et al. 2024](https://arxiv.org/abs/2406.04201): en juegos de más de 2 jugadores, los
  equilibrios de Nash no se pueden calcular eficientemente, puede haber varios con valores
  distintos, y **no son inexplotables** como en dos jugadores de suma cero · abstract.
- [Sychrovský et al. 2025](https://arxiv.org/abs/2504.18868): meta-learning para aproximar Nash en
  juegos de suma general · baja · abstract.

### Alternativas al self-play simple (agregado el 6/11/2026, de memoria, sin releer los papers)

- **Self-play regularizado**: *Magnetic Mirror Descent* — Sokota et al., ICLR 2023
  ([arXiv 2206.05825](https://arxiv.org/abs/2206.05825)): a PPO se le suma un KL hacia una política
  "imán" que se va actualizando; converge en dos jugadores de suma cero y funciona bien como
  regularizador en general. **R-NaD** (DeepNash, Perolat et al., Science 2022, Stratego): idea
  parecida a gran escala. Media: dan estabilidad, pero las garantías son para dos jugadores.
- **Poblaciones**: PSRO (Lanctot et al. 2017), NeuPL (Liu et al. 2022) y la **liga de AlphaStar**
  (Vinyals et al. 2019: agentes principales, *exploiters* que buscan sus debilidades y versiones
  viejas). Media: sirven para ser robustos contra estilos distintos (perfiles, humanos), menos para
  subir el promedio contra la heurística. NFSP, que ya probamos, es de esta familia.
- **Estrategia base + búsqueda al jugar**: **Pluribus** — Brown y Sandholm, Science 2019 (póker de
  6 jugadores; CFR para la base y búsqueda limitada en cada decisión), **ReBeL** (Brown et al.,
  NeurIPS 2020) y **Student of Games** (Schmid et al., Science Advances 2023). Alta como idea: es el
  antecedente más cercano para muchos jugadores. Con el motor en JAX, simular sale barato; la
  versión simple para nosotros es PIMC guiado por la política y el valor de PPO (sección 3).
- **Diplomacy** (7 jugadores, suma general): DORA (Bakhtin et al. 2021), **piKL** (Jacob et al.,
  ICML 2022) y Cicero (Meta, Science 2022). El self-play puro converge a convenciones raras que no
  andan con humanos; regularizar hacia una política humana (piKL) arregla eso. Baja por ahora: nos
  sirve si algún día juntamos partidas reales.

## 5. Evaluación y herramientas

- **GTO Wizard Benchmark** (póker, 2026, [arXiv 2603.23660](https://arxiv.org/abs/2603.23660)) ·
  media · abstract: usan **AIVAT**, una técnica de reducción de varianza que llega a la misma
  significancia con ~10 veces menos manos. La idea general aplica: **reducir la varianza de la
  evaluación** en vez de jugar más partidas.
- **Valet** (2026, [arXiv 2603.03252](https://arxiv.org/abs/2603.03252)) · baja: testbed de 21
  juegos de cartas tradicionales en el lenguaje RECYCLE. No confirmamos si incluye Oh Hell.
- **POPGym Arcade** (ICML 2026, [arXiv 2503.01450](https://arxiv.org/abs/2503.01450)) · baja:
  sobre memoria en RL; los críticos pueden asignar crédito a historia irrelevante. Coincide con que
  el LSTM no ayudó en Wizard.

No pudimos abrir: la pregunta de [AI StackExchange](https://ai.stackexchange.com/questions/25439)
(bloquea la lectura automática), [IEEE 6633646](https://ieeexplore.ieee.org/document/6633646) (no
identificamos el documento) y el artículo de [ProQuest](https://www.proquest.com/docview/2635404992).

## Qué nos dice para este proyecto

Ningún trabajo publicado entrena agentes fuertes para Oh Hell / La Podrida completa. Lo más cercano
es Wizard (2022). Lo que la literatura sugiere, ordenado por evidencia y costo:

1. **Cortar el episodio en cada ronda** (Wizard; Suphx resuelve el mismo problema con predicción
   del reward global). Hoy el valor de cantar en la ronda 1 arrastra los puntos de las 12 rondas
   siguientes, que no dependen de esa decisión: es varianza pura. Cambio chico en `EpisodeRunner`.
2. **Una red compartida por todos los asientos** (Wizard; es lo estándar en self-play de cartas).
   Con n jugadores, n veces más datos por red; además facilita el entrenamiento en lote.
3. **Reconsiderar NFSP.** La evidencia en juegos de cartas de varios jugadores favorece self-play
   más simple: DQN en Wizard, PPO en Big 2 (que en 2026 le ganó a Q-learning, SARSA y Monte Carlo Q
   en una comparación controlada), DMC en Dou Dizhu. NFSP no tiene garantías con más de 2 jugadores,
   y en RLCard fue inestable en juegos grandes. Probar **DQN compartido** (en nuestro diagnóstico ya
   llegaba a 0.44 vs heurística en 20k episodios) y **PPO con self-play contra la política actual y
   regularización de entropía**. El plan original de PSRO conviene revisarlo con esto en mente.
4. **Separar canto y juego** (Wizard, Judgement, AlphaDou). El canto es una predicción de bazas, y
   según Edelkamp es la decisión que más pesa en el resultado. Puede ayudar con el problema de cantar
   de más.
5. **Buffers y batches más grandes** si seguimos con DQN (Wizard: 300k–600k, batch 1024).
6. **Evaluación con menos varianza**: repartos espejo (las mismas cartas jugadas rotando los
   asientos, como el bridge duplicado). Es la versión simple de lo que hace AIVAT en póker.
7. **Lo que no priorizar**: historia con LSTM (no ayudó en Wizard) y búsqueda dentro del agente (no
   ayudó en Wizard). Más adelante, un jugador PIMC como **rival de evaluación** más fuerte que la
   heurística.

**Actualización del 6/11/2026**, con PPO compartido ya andando: el crítico central (MAPPO) empató
con el local, como era esperable en un juego competitivo con episodios cortos. Lo que queda por
probar, ordenado por costo:

1. **Pérdida auxiliar que prediga las manos ajenas**: casi gratis, y usa la información oculta de
   una forma más directa que el crítico.
2. **Búsqueda al jugar** (PIMC guiado por la política y el valor de PPO, al estilo Pluribus): es lo
   de mayor techo y el motor en JAX la abarata. Ojo: en Wizard la búsqueda dentro del agente no
   ayudó (punto 7), así que conviene medirla primero en las últimas bazas, donde las manos
   muestreadas están más restringidas.
3. **Regularizar el self-play** (MMD) o entrenar con una **población o liga** si los perfiles
   muestran estilos que lo explotan.
