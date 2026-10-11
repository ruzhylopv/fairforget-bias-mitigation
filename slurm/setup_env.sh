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

# The module stack already has some of transformers' dependencies, in versions pip accepts but
# transformers' own import-time check rejects (e.g. regex). Install exactly what that check
# wants; pip only touches packages that don't satisfy it. Read without importing transformers.
echo "Upgrading packages that are too old for transformers' import check"
mapfile -t TRANSFORMERS_REQS < <(python - <<'PY'
import ast
import importlib.util
import pathlib
import re

pkg_dir = pathlib.Path(importlib.util.find_spec("transformers").submodule_search_locations[0])
table = {}
exec((pkg_dir / "dependency_versions_table.py").read_text(), table)
check = (pkg_dir / "dependency_versions_check.py").read_text()
names = ast.literal_eval(re.search(r"pkgs_to_check_at_runtime = (\[.*?\])", check, re.S).group(1))
for name in names:
    if name != "python" and name in table["deps"]:
        print(table["deps"][name])
PY
)
pip install --quiet "${TRANSFORMERS_REQS[@]}"

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
