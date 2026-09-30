#!/bin/bash
#SBATCH --partition=H100,audible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

BENCHMARK="${1:?benchmark name required}"
ROOT="/home/ids/hmalard/TACO2"
cd "$ROOT"

source /home/ids/hmalard/miniconda3/etc/profile.d/conda.sh
conda activate fcclip2

export PYTHONPATH="${ROOT}/TACO:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$ROOT/TACO/results" "$ROOT/TACO/logs"
python -m taco.eval --benchmark "$BENCHMARK" --runs 3 --output "$ROOT/TACO/results/${BENCHMARK}.json"
python -m taco.check_results "$ROOT/TACO/results/${BENCHMARK}.json"
