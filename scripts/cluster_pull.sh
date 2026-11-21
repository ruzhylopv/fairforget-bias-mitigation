#!/bin/bash
# Copy files from your clone on JUPITER into this repo (laptop side; rsync over ssh).
#
#   scripts/cluster_pull.sh [path ...]        or: make cluster-pull [WHAT="path ..."]
#
# Paths are relative to the repo root (default: results). Files that exist only here are left
# alone; files on both sides are replaced by the cluster's copy. Refuses if any of the local
# files have uncommitted changes (FORCE=1 overwrites them), and shows git status afterwards.
#
# Settings (environment):
#   JUPITER_HOST   ssh host or ~/.ssh/config alias (default: jupiter)
#   JUPITER_USER   cluster user (default: the User of JUPITER_HOST in ~/.ssh/config)
#   CLUSTER_CLONE  the clone on the cluster (default: the team folder's bias-mitigation-<user>)
#   CLUSTER_SSH    the ssh command (default: ssh -4)
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
host=${JUPITER_HOST:-jupiter}
user=${JUPITER_USER:-$(ssh -G "$host" | awk '/^user /{print $2}')}
clone=${CLUSTER_CLONE:-/e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/bias-mitigation-$user}
paths=("$@")
[[ ${#paths[@]} -gt 0 ]] || paths=(results)

dirty=$(git status --porcelain -- "${paths[@]}")
if [[ -n $dirty && -z ${FORCE:-} ]]; then
    echo "Local uncommitted changes would be overwritten (commit them first, or FORCE=1):"
    echo "$dirty"
    exit 1
fi

echo "== $host:$clone -> ${paths[*]}"
for p in "${paths[@]}"; do
    p=${p%/}
    # "host:clone/p" (no trailing slash) into p's parent folder updates p itself, whether it is
    # a file or a folder. (No -R: macOS's openrsync ignores the /./ marker and would recreate
    # the whole absolute path here.)
    mkdir -p "$(dirname "$p")"
    rsync -rtvc -e "${CLUSTER_SSH:-ssh -4}" --exclude __pycache__ --exclude .ipynb_checkpoints \
        "$host:$clone/$p" "$(dirname "$p")/" | grep -v -e '/$' -e '^$' -e '^Transfer starting' || true
done
echo "== git status"
git status --short -- "${paths[@]}"
