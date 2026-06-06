#!/usr/bin/env bash
# Read-only client-balance метрики для подтверждения post-nav-fix плюса.
# Метрики (Фрай): net/sec, eat-rate, glucose-level. НИЧЕГО не меняет.
LOG="/c/Users/Mr. Krabs/AppData/Roaming/utopia-client/client.log"
N="${1:-8}"

echo "=== BALANCE METRICS (last $N windows, ticks=300 each) ==="
echo "-- net/sec + ratio (ENERGY_CALIB) --"
grep "ENERGY_CALIB" "$LOG" | tail -"$N" | while read -r line; do
  ts=$(echo "$line" | awk '{print $2}' | cut -d',' -f1)
  net=$(echo "$line" | grep -oE 'net=-?[0-9.]+' | cut -d= -f2)
  ratio=$(echo "$line" | grep -oE 'ratio=[0-9.]+' | cut -d= -f2)
  inc=$(echo "$line" | grep -oE 'income=[0-9.]+' | cut -d= -f2)
  cost=$(echo "$line" | grep -oE 'cost=[0-9.]+' | cut -d= -f2)
  echo "  $ts net=$net ratio=$ratio (inc=$inc cost=$cost)"
done

echo "-- eat-rate (NAV_DIAG: eat+p40_ate per 300t) + gather_onf + onf_rate --"
grep "NAV_DIAG ticks=" "$LOG" | tail -"$N" | while read -r line; do
  ts=$(echo "$line" | awk '{print $2}' | cut -d',' -f1)
  eat=$(echo "$line" | grep -oE ' eat=[0-9]+' | cut -d= -f2)
  p40=$(echo "$line" | grep -oE 'p40_ate=[0-9]+' | cut -d= -f2)
  go=$(echo "$line" | grep -oE 'gather_onf=[0-9]+' | cut -d= -f2)
  onf=$(echo "$line" | grep -oE 'onf_rate=[0-9.]+' | cut -d= -f2)
  echo "  $ts eat-rate=$((eat + p40)) (eat=$eat p40=$p40) gather_onf=$go onf_rate=$onf"
done

echo "-- glucose-level + viability (BIOCHEM_DEBUG, last 4) --"
grep "BIOCHEM_DEBUG" "$LOG" | tail -4 | grep -oE 'c103927:e=[0-9.]+,cort=[0-9.]+,ser=[0-9.]+,g=[0-9.]+' | sed 's/^/  /'

echo "-- VERDICT helper: avg net over last $N --"
grep "ENERGY_CALIB" "$LOG" | tail -"$N" | grep -oE 'net=-?[0-9.]+' | cut -d= -f2 | \
  awk '{s+=$1; n++} END {if(n>0) printf "  avg net=%.1f over %d windows  → %s\n", s/n, n, (s/n>=0?"PLUS (баланс встал)":"MINUS (ещё дефицит)")}'
