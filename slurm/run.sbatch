#!/bin/bash
# Run one command on a GPU node. Don't sbatch this directly; use slurm/submit.sh (or make submit),
# which creates the log folder and records the purpose.
#
# A job is billed for the whole node (288 cores, 4 GPUs) even if it uses one GPU.
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/logs/%x-%j.out

set -eo pipefail

# sbatch runs a copy of this script, so locate the repo from where it was submitted.
REPO_DIR=$SLURM_SUBMIT_DIR
cd "$REPO_DIR"
source slurm/env.sh
source slurm/usage_log.sh

RUN_DIR=$(python -m rmu.config paths.runs_root)/$SLURM_JOB_ID
export RUN_DIR
mkdir -p "$RUN_DIR"
usage_log_start "$*"

# What ran, from which code: enough to reproduce the run later.
{
    echo "job_id:   $SLURM_JOB_ID"
    echo "purpose:  ${RUN_PURPOSE:-}"
    echo "command:  $*"
    echo "user:     $USER"
    echo "nodes:    $SLURM_JOB_NODELIST"
    echo "started:  $(date "+%F %T")"
    echo "repo:     $REPO_DIR"
    echo "commit:   $(git rev-parse HEAD)"
    echo "changes:  $(git status --porcelain | wc -l | tr -d " ") uncommitted file(s)"
} | tee "$RUN_DIR/job.txt"
git diff > "$RUN_DIR/uncommitted.diff"

srun "$@"
