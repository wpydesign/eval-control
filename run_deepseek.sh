#!/bin/bash
# Run DeepSeek lattice with resume + nohup (survives context timeout)
cd /home/z/my-project
export PYTHONPATH=/home/z/my-project/download:$PYTHONPATH

LOG=/home/z/my-project/download/lcge_engine/lattice/output/deepseek_run.log

nohup python3 download/lcge_engine/lattice/run_deepseek_lattice.py \
    -o download/lcge_engine/lattice/output/runs_deepseek.jsonl \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Log: $LOG"
echo "Started at: $(date)"
