#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for year in 2014 2016 2019; do
  if [[ ! -f "data/crohme/${year}/test.jsonl" ]]; then
    "${PYTHON_BIN}" -m dec_unimumer.crohme --download
    break
  fi
done

"${PYTHON_BIN}" -m dec_unimumer.experiment \
  --dataset-path data/raw/unimumer_mathwriting_hf \
  --crohme-root data/crohme \
  --max-cycles 5 \
  --patience 1 \
  --epochs-per-cycle 1 \
  --recognition-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --inference-batch-size 4 \
  --generation-eval-batch-size 1 \
  --max-length 0 \
  --max-pixels 160000 \
  --max-new-tokens 384 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --vision-lora-target-modules qkv \
  --seed 42 \
  --report-to wandb \
  --wandb-project dynamic-error-corpus-unimumer \
  --cdm-evaluator external/UniMERNet/cdm/evaluation.py \
  --cdm-docker-image unimernet-cdm:latest \
  --cdm-pools 4 \
  "$@"
