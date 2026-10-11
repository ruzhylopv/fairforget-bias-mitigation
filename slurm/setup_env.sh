#!/bin/bash
# Create (or update) your personal Python env. Run once per person, on the login node:
#
#   make env
#
# torch comes from the PyTorch module (built for GH200); the env only adds this repo and its
# other dependencies on top, via --system-site-packages.
set -eo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RMU_SETUP=1 source "$REPO_DIR/slurm/env.sh"

if [[ ! -d $ENV_DIR ]]; then
    echo "Creating env at $ENV_DIR"
    python3 -m venv --system-site-packages "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

echo "Installing $REPO_DIR (editable) and its dependencies"
pip install --quiet -e "$REPO_DIR"

mkdir -p "$TEAM_SCRATCH/runs" "$TEAM_SCRATCH/logs" "$TEAM_ROOT/results"

python - "$ENV_DIR" <<'PY'
import sys

import rmu
import torch
import transformers

env_dir = sys.argv[1]
print(f"rmu from:      {rmu.__file__}")
print(f"torch:         {torch.__version__} from {torch.__file__}")
print(f"transformers:  {transformers.__version__}")
if torch.__file__.startswith(env_dir):
    sys.exit("ERROR: pip installed its own torch into the env. It must come from the module.")
print("Env OK.")
PY
