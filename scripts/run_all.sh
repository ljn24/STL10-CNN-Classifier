#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
    configs/baseline.yaml
    configs/advanced.yaml
    configs/ablation/Z1_no_residual.yaml
    configs/ablation/Z2_no_se.yaml
    configs/ablation/Z3_no_augment.yaml
    configs/ablation/Z4_no_cutmix.yaml
    configs/ablation/Z5_no_all_aug.yaml
    configs/ablation/Z6_no_recipe.yaml
)

for cfg in "${CONFIGS[@]}"; do
    echo "====== Train: $cfg ======"
    python -m src.train --config "$cfg"
    echo "====== Eval:  $cfg ======"
    python -m src.eval  --config "$cfg"
    echo ""
done
