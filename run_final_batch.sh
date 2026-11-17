#!/bin/bash
# Final validation batch: MOSI 4 budgets x {zeros, detector} x 10 seeds,
# base encoder, MOSEI 40ep, frozen-template control.
# Idempotent: skips any run whose eval JSON exists. (run_final_batch.sh, as executed 2026-11-02)
set -u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate oldpy
OUT=/workspace/runs_final
mkdir -p $OUT/ckpt $OUT/logs $OUT/eval
cd /workspace/upload/XlNet_test

train_eval() {
  local DS=$1 EP=$2 SRC=$3 S=$4; shift 4
  local CS=$SRC; [ "$SRC" = "frozentpl" ] && CS=detector
  local TAG="${DS}_${EP}ep_${SRC}_seed${S}"
  [ -f "$OUT/eval/eval_${TAG}.json" ] && { echo "SKIP $TAG"; return; }
  echo "=== RUN $TAG $(date) ==="
  env "$@" python3 run_stage3.py --dataset $DS --seed $S --n_epochs $EP       --use_asynchrony true --use_corruption true --corruption_prob 0.3       --chaos_source $CS 2>&1 | tee $OUT/logs/train_${TAG}.txt
  local CK=$(ls -t /tmp/${DS}_seed${S}_*_final.pt 2>/dev/null | head -1)
  if [ -n "$CK" ]; then cp "$CK" $OUT/ckpt/final_${TAG}.pt && rm -f "$CK"; fi
  local EV="$OUT/ckpt/final_${TAG}.pt"
  [ -f "$EV" ] || { echo "WARN no ckpt $TAG"; return; }
  python3 text_corrupt_eval.py --dataset $DS --seed $S --chaos_source $CS     --model_path $EV --out $OUT/eval/eval_${TAG}.json 2>&1 | tail -2
  if [ "$DS" = "mosi" ] && [ "$CS" = "detector" ]; then
    for MF in 0.25 0.50 0.75; do
      python3 text_corrupt_eval.py --dataset $DS --seed $S --chaos_source $CS         --model_path $EV --mask_fraction $MF         --out $OUT/eval/eval_${TAG}_mask${MF}.json 2>&1 | tail -1
    done
    python3 text_corrupt_eval.py --dataset $DS --seed $S --chaos_source $CS       --model_path $EV --shuffle_chaos       --out $OUT/eval/eval_${TAG}_shuffled.json 2>&1 | tail -1
  fi
  echo "=== DONE $TAG $(date) ==="
}

SEEDS="42 123 2024 1 7 100 555 777 3141 9001"
for EP in 40 80 100 120; do
  for SRC in zeros detector; do
    for S in $SEEDS; do train_eval mosi $EP $SRC $S; done
  done
done
for S in $SEEDS; do
  TAG="mosi_40ep_base_seed${S}"
  [ -f "$OUT/eval/eval_${TAG}.json" ] && { echo "SKIP $TAG"; continue; }
  echo "=== RUN $TAG $(date) ==="
  python3 run_stage3.py --dataset mosi --seed $S --n_epochs 40     --use_asynchrony false --use_corruption false --chaos_source zeros     2>&1 | tee $OUT/logs/train_${TAG}.txt
  CK=$(ls -t /tmp/mosi_seed${S}_asyncFalse_corrFalse_srczeros_final.pt 2>/dev/null | head -1)
  [ -n "$CK" ] && cp "$CK" $OUT/ckpt/final_${TAG}.pt && rm -f "$CK"
  python3 text_corrupt_eval.py --dataset mosi --seed $S --chaos_source zeros     --model_path $OUT/ckpt/final_${TAG}.pt --out $OUT/eval/eval_${TAG}.json 2>&1 | tail -2
  echo "=== DONE $TAG $(date) ==="
done
for SRC in zeros detector; do
  for S in 42 123 2024; do train_eval mosei 40 $SRC $S; done
done
for S in 42 123 2024; do
  train_eval mosi 40 frozentpl $S TRUSTGATE_FREEZE_TEMPLATE=1
done
echo "ALL RUNS COMPLETE"
