#!/bin/bash
# Modality ablation (paper Sect. 4.3, Table 4 / Fig. 3).
# Variant A: base encoder trained normally; zero AV at eval.
# Variant B: text-only (AV zeroed in train AND test via datasets/mosi_zeroav.pkl).
set -u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate oldpy
OUT=/workspace/runs_ablation
mkdir -p $OUT/ckpt $OUT/logs $OUT/eval
cd /workspace/upload/XlNet_test
SEEDS="42 123 2024 1 7 100 555 777 3141 9001"

for S in $SEEDS; do
  TAG="mosi_40ep_base_seed${S}"
  [ -f "$OUT/eval/eval_A_${TAG}.json" ] && { echo "SKIP A-$TAG"; continue; }
  echo "=== RUN A $TAG $(date) ==="
  python3 run_stage3.py --dataset mosi --seed $S --n_epochs 40     --use_asynchrony false --use_corruption false --chaos_source zeros     2>&1 | tee $OUT/logs/train_A_${TAG}.txt
  CK=$(ls -t /tmp/mosi_seed${S}_asyncFalse_corrFalse_srczeros_final.pt 2>/dev/null | head -1)
  [ -n "$CK" ] && cp "$CK" $OUT/ckpt/final_A_${TAG}.pt && rm -f "$CK"
  EV=$OUT/ckpt/final_A_${TAG}.pt
  [ -f "$EV" ] || { echo "WARN no ckpt A-$TAG"; continue; }
  for FLAGS in "" "--zero_audio" "--zero_video" "--zero_audio --zero_video"; do
    SUF=$(echo "clean${FLAGS}" | tr -d ' -' | sed 's/cleanzero/clean_/')
    python3 text_corrupt_eval.py --dataset mosi --seed $S --chaos_source zeros       --model_path $EV $FLAGS --out $OUT/eval/eval_A_${TAG}${SUF}.json 2>&1 | tail -1
  done
  echo "=== DONE A $TAG $(date) ==="
done

for S in $SEEDS; do
  TAG="mosi_40ep_textonly_seed${S}"
  [ -f "$OUT/eval/eval_B_${TAG}.json" ] && { echo "SKIP B-$TAG"; continue; }
  echo "=== RUN B $TAG $(date) ==="
  python3 run_stage3.py --dataset mosi_zeroav --seed $S --n_epochs 40     --use_asynchrony false --use_corruption false --chaos_source zeros     2>&1 | tee $OUT/logs/train_B_${TAG}.txt
  CK=$(ls -t /tmp/mosi_zeroav_seed${S}_asyncFalse_corrFalse_srczeros_final.pt 2>/dev/null | head -1)
  [ -n "$CK" ] && cp "$CK" $OUT/ckpt/final_B_${TAG}.pt && rm -f "$CK"
  EV=$OUT/ckpt/final_B_${TAG}.pt
  [ -f "$EV" ] || { echo "WARN no ckpt B-$TAG"; continue; }
  python3 text_corrupt_eval.py --dataset mosi_zeroav --seed $S --chaos_source zeros     --model_path $EV --out $OUT/eval/eval_B_${TAG}.json 2>&1 | tail -1
  echo "=== DONE B $TAG $(date) ==="
done
echo "ABLATION COMPLETE"
