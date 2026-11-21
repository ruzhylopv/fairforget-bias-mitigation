#!/bin/bash
# Your jobs since a date (default: today), as `sacct` lists them, plus two columns:
#
#   EstimatedStart  pending jobs: the start Slurm estimates (squeue %S), or why it has none yet,
#                   e.g. "(Dependency)" for a null job waiting on its quick job, "(Priority)"
#   Description     what the job is: its purpose (PURPOSE of make submit / make rq2), taken from
#                   the job comment (set by slurm/submit.sh), else the team usage log (finished
#                   jobs), else the job's command ("rq2 quick: qwen3-8b / gender" for the pipeline)
#
# Pending jobs that sacct leaves out (those still waiting on a dependency) are added at the end.
#
#   slurm/jobs.sh [YYYY-MM-DD]        or: make jobs [SINCE=YYYY-MM-DD]
set -eo pipefail

since=${1:-$(date +%F)}
log=$(python -m rmu.config paths.usage_log 2>/dev/null ||
    echo /e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/usage_log.tsv)

# Passed to awk through the environment: some awks refuse newlines in -v values.
# QUEUE: "<id>|<name>|<state>|<estimated start>|<reason>|<comment>|<command>" per queued job.
export QUEUE USAGE
QUEUE=$(squeue -h -u "$USER" -o "%i|%j|%T|%S|%r|%k|%o" 2>/dev/null || true)
# USAGE: "<job id>\t<purpose>" of the jobs in the usage log (written when a job ends).
USAGE=$(awk -F'\t' 'NR > 1 { print $2 "\t" $5 }' "$log" 2>/dev/null | tail -n 5000 || true)

sacct -u "$USER" -S "$since" -X \
    --format=JobID,JobName%12,Start,Elapsed,NNodes,NodeList%14,State,ExitCode |
    awk '
        function describe(cmd,    w, n, i, model, bias, stage) {
            # The pipeline command -> "rq2 <stage>: <model> / <bias>"; else the command itself.
            n = split(cmd, w, " ")
            for (i = 1; i < n; i++) {
                if (w[i] == "--model") model = w[i + 1]
                if (w[i] == "--bias") bias = w[i + 1]
                if (w[i] == "--stage") stage = w[i + 1]
            }
            if (cmd ~ /probing\.pipeline/ && model != "")
                return "rq2 " stage ": " model " / " bias
            sub(/^.*run\.sbatch /, "", cmd)
            return substr(cmd, 1, 70)
        }
        BEGIN {
            n = split(ENVIRON["USAGE"], u, "\n")
            for (i = 1; i <= n; i++) {
                split(u[i], f, "\t")
                if (f[1] != "" && f[2] != "") desc[f[1]] = f[2]
            }
            n = split(ENVIRON["QUEUE"], q, "\n")
            for (i = 1; i <= n; i++) {
                split(q[i], f, "|")
                id = f[1]
                if (id == "") continue
                if (f[3] == "PENDING") {
                    est[id] = (f[4] == "N/A" || f[4] == "") ? "(" f[5] ")" : f[4]
                    order[++m] = id
                    name[id] = f[2]
                }
                if (f[6] != "" && f[6] != "(null)") desc[id] = f[6]
                else if (!(id in desc)) desc[id] = describe(f[7])
            }
        }
        NR == 1 { printf "%s  %-21s %s\n", $0, "EstimatedStart", "Description"; next }
        NR == 2 { printf "%s  %-21s %s\n", $0, "---------------------", "-----------"; next }
        { seen[$1] = 1; printf "%s  %-21s %s\n", $0, est[$1], desc[$1] }
        END {
            for (i = 1; i <= m; i++) {
                id = order[i]
                if (!(id in seen))
                    printf "%-12s %12s %19s %10s %8s %14s %10s %8s   %-21s %s\n", id, name[id], \
                        "Unknown", "00:00:00", "1", "None assigned", "PENDING", "0:0", est[id], \
                        desc[id]
            }
        }
    '
