#!/bin/bash
# run_with_watchdog.sh <stall_seconds> <log_file> -- <command...>
#
# Runs <command> with its output appended to <log_file>. Restarts it when it exits with
# an error, and kills and restarts it when <log_file> has not been written for
# <stall_seconds>. `ibench run` resumes from the items already in preds.jsonl, so a
# restart loses at most the item in progress. Choose <stall_seconds> longer than the
# longest single item. Exits 0 when <command> exits 0; gives up after 30 attempts.
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
