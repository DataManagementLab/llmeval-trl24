#!/bin/bash

set -e

limit_instances=2000
models=("gpt-4o-2024-08-06" "gpt-3.5-turbo-1106" "gpt-4o-mini-2024-07-18")
schema_modes=("opaque" "multi-table")  # "descriptive" (not in the paper)
perturbation_modes=("single" "multi")

for model in "${models[@]}"; do
  for schema_mode in "${schema_modes[@]}"; do
    for perturbation_mode in "${perturbation_modes[@]}"; do
      if [[ "$perturbation_mode" = "single" && ! ( "$model" = "gpt-4o-2024-08-06" && "$schema_mode" = "opaque" ) ]]; then
        continue  # we need perturbation_mode = "single" only for gpt-4o and schema_mode = "opaque"
      fi
      bash scripts/entity_matching/run.sh \
        exp_name="exp-pay-to-inv_${model}_${schema_mode}_${perturbation_mode}" \
        dataset="pay_to_inv" \
        api_name="openai" \
        model="$model" \
        limit_instances="$limit_instances" \
        dataset.schema_mode="$schema_mode" \
        dataset.perturbation_mode="$perturbation_mode"
    done
  done
done
