#!/bin/bash

set -e

python scripts/entity_matching/gather_results.py dataset=pay_to_inv
python scripts/entity_matching/plot.py dataset=pay_to_inv