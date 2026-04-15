#!/bin/bash
#SBATCH --job-name=wg-eval
#SBATCH --account=AIFAC_5C0_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --mem=368G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=0
#SBATCH --time=23:59:59
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --switches=1

set -euo pipefail

module load gcc/12.2.0
module load cuda/12.2

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file.yml> [run_id] [label]"
    exit 1
fi

CONFIG_FILE="$1"
RUN_ID="${2:-}"
LABEL="${3:-}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Starting Evaluation Job at $(date)"
echo "Config : $CONFIG_FILE"
echo "Run ID : $RUN_ID"
echo "Label  : $LABEL"

TMP_CONFIG=$(mktemp)
trap 'rm -f "$TMP_CONFIG"' EXIT

if [ -n "$RUN_ID" ]; then
    # config patch: add/update one run_id entry while keeping other settings intact
    export WG_CONFIG_FILE="$CONFIG_FILE"
    export WG_RUN_ID="$RUN_ID"
    export WG_LABEL="$LABEL"
    uv run --offline python - <<'EOF' > "$TMP_CONFIG"
import os

import yaml

with open(os.environ["WG_CONFIG_FILE"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

run_id = os.environ["WG_RUN_ID"].strip()
label = os.environ.get("WG_LABEL", "").strip()

if "run_ids" not in cfg or not isinstance(cfg["run_ids"], dict):
    raise ValueError("Config must contain 'run_ids' as a dictionary")

existing = cfg["run_ids"].get(run_id, {})
if not isinstance(existing, dict):
    existing = {}

if not label:
    label = existing.get("label", run_id)

cfg["run_ids"][run_id] = {
    **existing,
    "label": label,
}

print(yaml.safe_dump(cfg, sort_keys=False))
EOF
else
    cp "$CONFIG_FILE" "$TMP_CONFIG"
fi

if [ ! -s "$TMP_CONFIG" ]; then
    echo "Error: generated config is empty"
    exit 1
fi

export WG_TMP_CONFIG="$TMP_CONFIG"
uv run --offline python - <<'EOF'
import os

import yaml

with open(os.environ["WG_TMP_CONFIG"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

for key in ["evaluation", "run_ids"]:
    if key not in cfg:
        raise ValueError(f"Generated config is missing required key: {key}")
EOF

echo "Modified config:"
cat "$TMP_CONFIG"


# Run Evaluation
uv run --offline evaluate --config "$TMP_CONFIG"

echo "Finished Evaluation Job at $(date)"