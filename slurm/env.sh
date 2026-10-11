# Set up modules, your Python env and caches on JUPITER. Source it, don't run it:
#
#   source slurm/env.sh
#
# Every job script sources this file, so this is the only place that knows about modules.

PROJECT_ACCOUNT=e-dev-2026d09-075
TEAM_ROOT=/e/project1/$PROJECT_ACCOUNT/fairforget-bias-mitigation
TEAM_SCRATCH=/e/scratch/$PROJECT_ACCOUNT/fairforget-bias-mitigation
# One env per person: `pip install -e` points it at that person's clone.
ENV_DIR=$TEAM_ROOT/envs/$USER

# Without GCCcore the PyTorch module is hidden and there is no `python`.
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1

# Keep caches out of $HOME (20 GB / 80k files quota) and on scratch.
CACHE_DIR=$TEAM_SCRATCH/cache/$USER
export XDG_CACHE_HOME=$CACHE_DIR
export PIP_CACHE_DIR=$CACHE_DIR/pip
export HF_HOME=$CACHE_DIR/huggingface
export TRITON_CACHE_DIR=$CACHE_DIR/triton

# Compute nodes have no internet. Models load from the shared folder; fail fast instead of
# hanging if something tries to download from the Hugging Face Hub.
export HF_HUB_OFFLINE=1

# Write prints to the job log immediately (not in buffered chunks), so `tail -f` shows progress.
export PYTHONUNBUFFERED=1

# New files are readable and writable by the team (project group), not by everyone.
umask 0007

if [[ -f $ENV_DIR/bin/activate ]]; then
    source "$ENV_DIR/bin/activate"
    # Modules put their packages on PYTHONPATH, which Python searches before the venv. Put the
    # venv first so packages upgraded in it (e.g. regex for transformers) win over module copies.
    ENV_SITE=$(echo "$ENV_DIR"/lib/python3.*/site-packages)
    export PYTHONPATH=$ENV_SITE${PYTHONPATH:+:$PYTHONPATH}
elif [[ -z $RMU_SETUP ]]; then
    echo "slurm/env.sh: no Python env at $ENV_DIR yet. Create it once with: make env" >&2
fi
