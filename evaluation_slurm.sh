#!/bin/bash
#SBATCH --job-name=wg-eval
#SBATCH --account=AIFAC_5C0_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --mem=368G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=23:59:59
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --switches=1

module load gcc/12.2.0
module load cuda/12.2

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file.yml>"
    exit 1
fi

CONFIG_FILE="$1"

echo "Starting Evaluation Job at $(date)"
echo "Using config: $CONFIG_FILE"

# Run Evaluation
uv run --offline evaluate --config "$CONFIG_FILE"

echo "Finished Evaluation Job at $(date)"