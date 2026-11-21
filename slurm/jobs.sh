#!/bin/bash
# Your jobs since a date (default: today), as `sacct` lists them, plus one last column: for
# pending jobs the start time Slurm estimates (squeue %S), or why it has none yet, e.g.
# "(Dependency)" for a null job waiting on its quick job, "(Priority)" while queued behind others.
# Pending jobs that sacct leaves out (those still waiting on a dependency) are added at the end.
#
#   slurm/jobs.sh [YYYY-MM-DD]        or: make jobs [SINCE=YYYY-MM-DD]
set -eo pipefail

since=${1:-$(date +%F)}
# "<id> <name> <estimated start> <reason>" for every pending job; the estimate is N/A until the
# scheduler has planned the job, and it moves as other jobs end early or arrive.
# Passed to awk through the environment: some awks refuse newlines in -v values.
export PENDING_JOBS
PENDING_JOBS=$(squeue -h -u "$USER" -t PENDING -o "%i %j %S %r" 2>/dev/null || true)

sacct -u "$USER" -S "$since" -X \
    --format=JobID,JobName%12,Start,Elapsed,NNodes,NodeList%14,State,ExitCode |
    awk '
        BEGIN {
            n = split(ENVIRON["PENDING_JOBS"], lines, "\n")
            for (i = 1; i <= n; i++) {
                split(lines[i], f, " ")
                if (f[1] == "") continue
                est[f[1]] = (f[3] == "N/A" || f[3] == "") ? "(" f[4] ")" : f[3]
                name[f[1]] = f[2]
                order[++m] = f[1]
            }
        }
        NR == 1 { print $0 "  EstimatedStart"; next }
        NR == 2 { print $0 "  -------------------"; next }
        { seen[$1] = 1; print $0 (($1 in est) ? "  " est[$1] : "") }
        END {
            for (i = 1; i <= m; i++) {
                id = order[i]
                if (!(id in seen))
                    printf "%-12s %12s %19s %10s %8s %14s %10s %8s   %s\n", id, name[id], \
                        "Unknown", "00:00:00", "1", "None assigned", "PENDING", "0:0", est[id]
            }
        }
    '
