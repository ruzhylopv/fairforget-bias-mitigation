#!/bin/bash
# Submit a command as a GPU job. It is logged to the team usage log automatically.
#
#   slurm/submit.sh "<purpose>" <command...>
#
# Examples:
#   slurm/submit.sh "pilot: RMU on Lapa" python -m rmu.unlearn --model_name_or_path lapa-12b-instruct
#   JOB_TIME=00:20:00 slurm/submit.sh "smoke test" python scripts/smoke_test.py
#
# Optional settings: JOB_TIME (default 01:00:00), JOB_NAME (default rmu).
set -eo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_DIR/slurm/env.sh"
mkdir -p "$TEAM_SCRATCH/logs" "$TEAM_SCRATCH/runs"

export RUN_PURPOSE=$1
shift

cd "$REPO_DIR"
job_id=$(sbatch --parsable --time="${JOB_TIME:-01:00:00}" --job-name="${JOB_NAME:-rmu}" \
    slurm/run.sbatch "$@")
run_dir=$(python -m rmu.config paths.runs_root)/$job_id
echo "Submitted job $job_id"
echo "  log:     $TEAM_SCRATCH/logs/${JOB_NAME:-rmu}-$job_id.out"
echo "  outputs: $run_dir/"
echo "  status:  squeue -j $job_id     cancel: scancel $job_id"
