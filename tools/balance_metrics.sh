#!/usr/bin/env bash
# Клиент-АВТОРИТЕТНЫЕ метрики (Фрай 06.06): net/sec, eats/sec, on-flora.
# Кросс-чек к серверной дельте Хьюберта на том же окне. НИЧЕГО не меняет.
# net/sec = net окна / wall-секунды окна (норм. по timestamp-дельте).
LOG="/c/Users/Mr. Krabs/AppData/Roaming/utopia-client/client.log"
N="${1:-8}"

_secs() { date -d "2026-06-06 $1" +%s 2>/dev/null; }

echo "=== CLIENT-AUTHORITATIVE BALANCE (last $N windows) ==="
echo "-- net/sec (ENERGY_CALIB, норм. на wall-сек окна) --"
prev_ts=""
grep "ENERGY_CALIB" "$LOG" | tail -"$N" | while read -r line; do
  ts=$(echo "$line" | awk '{print $2}' | cut -d',' -f1)
  net=$(echo "$line" | grep -oE 'net=-?[0-9.]+' | cut -d= -f2)
  inc=$(echo "$line" | grep -oE 'income=[0-9.]+' | cut -d= -f2)
  cost=$(echo "$line" | grep -oE 'cost=[0-9.]+' | cut -d= -f2)
  cur=$(_secs "$ts")
  if [ -n "$prev_ts" ] && [ -n "$cur" ]; then
    dt=$((cur - prev_ts))
    [ "$dt" -le 0 ] && dt=1
    nps=$(awk "BEGIN{printf \"%.2f\", $net/$dt}")
    echo "  $ts net/sec=$nps (net=$net за ${dt}s, inc=$inc cost=$cost)"
  else
    echo "  $ts net=$net (inc=$inc cost=$cost) [первое окно — dt n/a]"
  fi
  prev_ts=$cur
done

echo "-- eats/sec + on-flora (NAV_DIAG, норм. на wall-сек) --"
prev_ts=""
grep "NAV_DIAG ticks=" "$LOG" | tail -"$N" | while read -r line; do
  ts=$(echo "$line" | awk '{print $2}' | cut -d',' -f1)
  eat=$(echo "$line" | grep -oE ' eat=[0-9]+' | cut -d= -f2)
  p40=$(echo "$line" | grep -oE 'p40_ate=[0-9]+' | cut -d= -f2)
  onf=$(echo "$line" | grep -oE 'onf_rate=[0-9.]+' | cut -d= -f2)
  go=$(echo "$line" | grep -oE 'gather_onf=[0-9]+' | cut -d= -f2)
  cur=$(_secs "$ts")
  tot=$((eat + p40))
  if [ -n "$prev_ts" ] && [ -n "$cur" ]; then
    dt=$((cur - prev_ts)); [ "$dt" -le 0 ] && dt=1
    eps=$(awk "BEGIN{printf \"%.3f\", $tot/$dt}")
    echo "  $ts eats/sec=$eps (eat+p40=$tot за ${dt}s) on_flora=$onf gather_onf=$go"
  else
    echo "  $ts eats=$tot on_flora=$onf gather_onf=$go [первое окно]"
  fi
  prev_ts=$cur
done

echo "-- glucose/energy/mb (BIOCHEM_DEBUG, last 4) --"
grep "BIOCHEM_DEBUG" "$LOG" | tail -4 | grep -oE 'c103927:e=[0-9.]+,cort=[0-9.]+,ser=[0-9.]+,g=[0-9.]+,mb=[a-z]*' | sed 's/^/  /'
