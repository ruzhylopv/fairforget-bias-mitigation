#!/bin/bash
# RQ2 probing pipeline (python -m probing.pipeline) for one model and bias type, as Slurm jobs.
#
#   slurm/rq2.sh <model> <bias> [all|quick|null|extract]   or: make rq2 MODEL=... BIAS=... [STAGE=...]
#
#   quick  job A (GPU node): meta, extraction, gate 1, probes, 5-seed null + C grid, stats (gate 2)
#   null   job B (long): the 100-seed null, then stats; refuses to start unless gate 2 passed
#   all    (default) A, then B with --dependency=afterok:A, so B starts only if A succeeded
#          (if A fails, Slurm drops all of B's pieces)
#   extract  one short GPU job: meta, extraction, gate 1 only (EXTRACT_TIME, default 00:45:00).
#          Short jobs start sooner in a busy queue; quick later skips these steps.
#
# A run that was cut off continues where it stopped: submit the same line again.
# Before submitting, the pipeline is dry-run here on the login node, so a run that cannot start
# (model not in pairs.csv, missing config) fails now instead of after the queue.
#
# Optional settings:
#   QUICK_TIME (default 02:00:00), NULL_TIME (default 12:00:00), EXTRACT_TIME (00:45:00)
#   NULL_PIECES (default 1): the null as that many jobs in a row, each continuing the last
#                (afterany, so a piece cut off by its time limit is picked up by the next;
#                every piece also needs job A to have succeeded)
#   JOB_DEPENDENCY: dependency of the first job submitted (e.g. afterok:1234567)
#   DRY_RUN=1: only the dry run, nothing is submitted
set -eo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
MODEL=$1
BIAS=$2
RQ2_STAGE=${3:-all}  # not STAGE: `module load Stages/...` (env.sh) exports STAGE=2026
case $RQ2_STAGE in
all | quick | null | extract) ;;
*) echo "stage must be all, quick, null or extract" >&2; exit 1 ;;
esac

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source slurm/env.sh

# The stage that runs first decides what can be checked now (gate 2 does not exist before A).
case $RQ2_STAGE in null | extract) first=$RQ2_STAGE ;; *) first=quick ;; esac
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
if [[ $RQ2_STAGE == extract ]]; then
    submit "rq2 extract: $MODEL / $BIAS" "${EXTRACT_TIME:-00:45:00}" rq2-extract "$dep" extract \
        >/dev/null
    exit 0
fi
if [[ $RQ2_STAGE != null ]]; then
    a=$(submit "rq2 quick: $MODEL / $BIAS" "${QUICK_TIME:-02:00:00}" rq2-quick "$dep" quick)
    dep=afterok:$a
    quick_ok=afterok:$a,
fi
if [[ $RQ2_STAGE != quick ]]; then
    for ((i = 1; i <= ${NULL_PIECES:-1}; i++)); do
        b=$(submit "rq2 null ($i/${NULL_PIECES:-1}): $MODEL / $BIAS" "${NULL_TIME:-12:00:00}" \
            rq2-null "$dep" null)
        # Every later piece also needs job A to have succeeded (comma = and): if A fails, all
        # pieces are dropped, instead of the 2nd starting once Slurm has dropped the 1st.
        dep=${quick_ok:-}afterany:$b
    done
fi
