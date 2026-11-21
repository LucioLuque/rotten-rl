# Panel fijo de rivales

Pendiente de la etapa A del plan del 16/11 (`docs/plan.md`). La heurística ya no distingue entre
agentes buenos (todos le ganan más o menos igual), y cada torneo hasta ahora comparaba contra un par de
agentes elegidos para ese experimento. El panel es un conjunto **fijo** de rivales por cantidad de
jugadores, contra el que mido cualquier agente nuevo siempre de la misma manera.

## Qué tiene

`panel.json`: por cantidad de jugadores, cada rival con su corrida (semilla 0) y un episodio fijo, para
que el panel no cambie aunque la corrida siga:

- la heurística;
- una versión anterior de nuestro agente (a 12M, o a 6M con 4 a 6 jugadores), para ver progreso;
- el agente de referencia del juego normal (`s20full`, `s30full`, `f40`, `f50`, `f60`, `u72full`);
- con 2, 3 y 7: uno entrenado con liga y un explotador entrenado contra el de referencia (para ver si
  un agente nuevo abre agujeros), y los de las variantes (`d`, regla de no repetir; `m`, formatos).

Faltan (no están en el motor JAX): los perfiles de canto escritos a mano (`rotten_rl/profiles/rules.py`,
solo en pyspiel) y mi perfil cuando tenga partidas. Con 4 a 6 jugadores todavía no hay liga, explotador
ni variantes.

## Cómo se usa

```bash
python scripts/eval/panel.py --players 3 --agents ~/scratch/runs/m30_*,~/scratch/runs/s30full_* \
    --formats --out ~/scratch/panel_3p.md
```

Cada agente (un patrón con `*` promedia sus semillas) juega una evaluación duplicada contra cada rival:
el agente en un asiento y el rival en todos los demás, cada reparto una vez por asiento (1000 repartos).
La tabla da por rival la diferencia de puntaje por partida y el win rate (empate 1/n), el promedio sin
la heurística (que dominaría el promedio) y el peor rival. `--formats` sortea el formato de cada reparto
(`docs/rule_variants/`); sin él, la escalera 1..7..1. En la -14, 2 agentes contra 7 rivales con 4
semillas: 21 s.

## Primera prueba (20/11, 2 jugadores, formatos sorteados)

| Agente | heurística | s20@12M | s20full | l20 | e20zl | d20 | m20 | Promedio (sin heurística) |
|---|---|---|---|---|---|---|---|---|
| `m20` | +39,06 | +4,03 | +3,53 | +3,07 | +2,05 | +2,95 | +0,32 | +2,66 |
| `s20full` | +35,13 | +0,67 | +0,05 | −0,17 | −0,29 | +0,10 | −2,99 | −0,44 |

(`m20` contra sí mismo da +0,32 y no 0 porque cada semilla juega contra la semilla 0 del panel.)

## Especialistas con formatos (21/11, formatos sorteados)

3 jugadores (empate 33,3 %; diferencia por partida, win rate):

| Agente | heurística | s30@12M | s30full | l30m | e30z | d30 | m30 | Promedio (sin heurística) |
|---|---|---|---|---|---|---|---|---|
| `m30` | +23,89 (66,6 %) | +2,38 | +3,20 | +2,57 | +3,87 | +3,19 | +0,31 | **+2,59** |
| `s30full` | +20,01 (61,4 %) | −0,78 | +0,19 | −0,53 | +0,42 | −0,27 | −3,53 | −0,75 |
| `d30` | +20,33 (61,4 %) | −0,89 | −0,03 | −0,52 | +0,29 | −0,40 | −3,80 | −0,89 |

4 jugadores (empate 25 %):

| Agente | heurística | f40@6M | f40 | m40 | Promedio (sin heurística) |
|---|---|---|---|---|---|
| `m40` | +21,61 (58,3 %) | +4,68 | +3,20 | +0,09 | **+2,66** |
| `f40` | +17,13 (50,9 %) | +1,00 | −0,06 | −3,84 | −0,97 |

Con formatos, el especialista de formatos le gana a todo el panel por 2 a 4 puntos, incluido el
explotador de `s30` (+3,87): no tiene un agujero que ese explotador encuentre. Su peor rival es él mismo
(la semilla 0). Los agentes de la escalera vieja quedan alrededor de 0 entre ellos y pierden ~3,5 contra
el de formatos. 7 jugadores, cuando termine `m72`; 5 y 6, con `m50` y `m60`.
