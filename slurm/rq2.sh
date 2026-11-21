#!/bin/bash
# RQ2 probing pipeline (python -m probing.pipeline) for one model and bias type, as Slurm jobs.
#
#   slurm/rq2.sh <model> <bias> [all|quick|null]         or: make rq2 MODEL=... BIAS=... [STAGE=...]
#
#   quick  job A (GPU node): meta, extraction, gate 1, probes, 5-seed null + C grid, stats (gate 2)
#   null   job B (long): the 100-seed null, then stats; refuses to start unless gate 2 passed
#   all    (default) A, then B with --dependency=afterok:A, so B starts only if A succeeded
#          (if A fails, B stays pending forever: scancel it)
#
# A run that was cut off continues where it stopped: submit the same line again.
# Before submitting, the pipeline is dry-run here on the login node, so a run that cannot start
# (model not in pairs.csv, missing config) fails now instead of after the queue.
#
# Optional settings:
#   QUICK_TIME (default 02:00:00), NULL_TIME (default 12:00:00): time limit per job
#   NULL_PIECES (default 1): the null as that many jobs in a row, each continuing the last
#                (afterany, so a piece cut off by its time limit is picked up by the next)
#   JOB_DEPENDENCY: dependency of the first job submitted (e.g. afterok:1234567)
#   DRY_RUN=1: only the dry run, nothing is submitted
set -eo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
MODEL=$1
BIAS=$2
RQ2_STAGE=${3:-all}  # not STAGE: `module load Stages/...` (env.sh) exports STAGE=2026
case $RQ2_STAGE in all | quick | null) ;; *) echo "stage must be all, quick or null" >&2; exit 1 ;; esac

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source slurm/env.sh

# The stage that runs first decides what can be checked now (gate 2 does not exist before A).
first=$([[ $RQ2_STAGE == null ]] && echo null || echo quick)
python -m probing.pipeline --model "$MODEL" --bias "$BIAS" --stage "$first" --dry_run
[[ -n ${DRY_RUN:-} ]] && exit 0

pipeline="python -m probing.pipeline --model $MODEL --bias $BIAS"
submit() {  # submit <purpose> <time> <name> <dependency> <stage>; prints the job id
    local out
    out=$(JOB_TIME=$2 JOB_NAME=$3 JOB_CPUS=288 JOB_DEPENDENCY=$4 \
        slurm/submit.sh "$1" $pipeline --stage "$5")
    echo "$out" >&2
    sed -n 's/^Submitted job //p' <<<"$out"
}

dep=${JOB_DEPENDENCY:-}
if [[ $RQ2_STAGE != null ]]; then
    a=$(submit "rq2 quick: $MODEL / $BIAS" "${QUICK_TIME:-02:00:00}" rq2-quick "$dep" quick)
    dep=afterok:$a
fi
if [[ $RQ2_STAGE != quick ]]; then
    for ((i = 1; i <= ${NULL_PIECES:-1}; i++)); do
        b=$(submit "rq2 null ($i/${NULL_PIECES:-1}): $MODEL / $BIAS" "${NULL_TIME:-12:00:00}" \
            rq2-null "$dep" null)
        dep=afterany:$b
    done
fi
