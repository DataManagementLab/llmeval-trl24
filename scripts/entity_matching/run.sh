#!/bin/bash

set -e

# extract dataset name from command line input
for arg in "$@"; do
  if [[ $arg =~ dataset=([^[:space:]]+) ]]; then
    dataset="${BASH_REMATCH[1]}"
    break
  fi
done

python scripts/entity_matching/"$dataset"/preprocess.py "$@"
python scripts/entity_matching/prepare_requests.py "$@"
python scripts/execute_requests.py -cp "../config/entity_matching" "$@"
python scripts/entity_matching/evaluate.py "$@"
