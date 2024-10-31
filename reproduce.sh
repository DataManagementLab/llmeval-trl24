#!/bin/bash

set -e

if ! command -v conda >/dev/null; then
    echo "You must have Conda installed (see README.md)!"
    exit
fi

if [ ! -d "data/openai_cache" ]; then
    echo "You must manually obtain \`data/openai_cache\` (see README.md)!"
    exit
fi

bash scripts/entity_matching/create_dataset.sh
bash scripts/entity_matching/experiments.sh
bash scripts/entity_matching/gather.sh
