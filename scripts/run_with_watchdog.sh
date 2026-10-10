#!/bin/bash
# Usage: run_with_watchdog.sh <stall_seconds> <log_file> -- <command...>
# Restarts <command> when it fails or <log_file> stalls; ibench run resumes finished items.
STALL=$1; LOG=$2; shift 3
WATCH="${WATCH_FILE:-$LOG}"
mkdir -p "$(dirname "$LOG")"
for attempt in $(seq 1 30); do
  echo "[watchdog] attempt $attempt: $*" | tee -a "$LOG"
  touch "$WATCH"
  setsid "$@" >> "$LOG" 2>&1 &
  PID=$!
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    AGE=$(( $(date +%s) - $(stat -c %Y "$WATCH") ))
    if (( AGE > STALL )); then
      echo "[watchdog] no output for ${AGE}s (limit ${STALL}s): stopping process group $PID" | tee -a "$LOG"
      kill -9 -- "-$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
      sleep 10
      break
    fi
  done
  wait "$PID"; RC=$?
  if [ "$RC" -eq 0 ]; then echo "[watchdog] clean exit after attempt $attempt" | tee -a "$LOG"; exit 0; fi
  echo "[watchdog] exit code $RC, restarting in 30s" | tee -a "$LOG"
  sleep 30
done
echo "[watchdog] 30 attempts exhausted" | tee -a "$LOG"; exit 1
