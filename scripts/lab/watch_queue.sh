#!/bin/bash
# Vigila una noche de corridas desde la laptop y termina (para que Claude lo atienda) cuando pasa
# algo: terminó otra corrida de la cola, apareció un archivo esperado (un torneo), llegó la hora
# del respaldo, o NO SE PUEDE MIRAR el lab. Nunca se queda mirando a ciegas.
#
#   scripts/lab/watch_queue.sh <log de la cola> <terminadas ya vistas> [archivo a esperar]
#   p.ej.  scripts/lab/watch_queue.sh queue7.log 3 torneo_3p.md     (rutas relativas a ~/scratch)
#
# Salida (una línea): TERMINO <n> | ARCHIVO | HORA | FALLA <detalle>.
# - Cada consulta usa su propia conexión (ControlPath=none): no depende de la conexión compartida,
#   que se rompe del lado del lab ("unknown userid") y dejaba al vigilante sin ver nada (noche del
#   29/10: 3 horas a ciegas, 7 lugares sin reemplazo).
# - Si falla 2 veces seguidas (~5 minutos), cierra la conexión compartida y sale con FALLA.
# - De lunes a viernes sale con HORA a las 06:35 (respaldo y reporte) y no consulta el lab hasta
#   las 22:00, porque cierra a las 07:00. Sábados y domingos sigue mirando todo el día.
log=$1; known=$2; file=${3:-}
fails=0
while true; do
  now=$((10#$(date +%H%M))); dow=$(date +%u)          # 1 = lunes ... 7 = domingo
  if [ "$dow" -le 5 ] && [ $now -ge 635 ] && [ $now -lt 2200 ]; then
    echo "HORA $(date +%H:%M)"; exit 0
  fi
  out=$(timeout 60 ssh -o ControlPath=none -o ConnectTimeout=20 labs004 \
    "grep -c terminó ~/scratch/logs/$log; if [ -n '$file' ] && [ -f ~/scratch/$file ]; then echo 1; else echo 0; fi" 2>&1)
  n=$(echo "$out" | sed -n 1p); f=$(echo "$out" | sed -n 2p)
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    fails=$((fails + 1))
    if [ $fails -ge 2 ]; then
      ssh -O exit labs004 >/dev/null 2>&1
      echo "FALLA $(date +%H:%M) $(echo "$out" | tail -1)"; exit 1
    fi
    sleep 150; continue
  fi
  fails=0
  if [ "$n" -gt "$known" ]; then echo "TERMINO $n"; exit 0; fi
  if [ "$f" = "1" ]; then echo "ARCHIVO"; exit 0; fi
  sleep 300
done
