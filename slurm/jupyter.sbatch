#!/bin/bash
# Start JupyterLab on a GPU node, to run notebook cells from VS Code on your laptop.
# Submit with `make jupyter`, then follow the instructions printed in the job log.
#
# The node is billed for as long as this job runs, even while you're not running cells.
# Cancel it when you're done: scancel <JOB_ID>
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=jupyter
#SBATCH --output=/e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/logs/%x-%j.out

set -eo pipefail

REPO_DIR=$SLURM_SUBMIT_DIR
cd "$REPO_DIR"
source slurm/env.sh
module load JupyterLab/4.5.5
source slurm/usage_log.sh
export RUN_PURPOSE=${RUN_PURPOSE:-interactive Jupyter}
usage_log_start "jupyter lab"

KERNEL=rmu-$USER
python -m ipykernel install --user --name "$KERNEL" --display-name "rmu ($USER)" > /dev/null

PORT=8888
cat <<EOF

=====================================================================
JupyterLab is starting on $(hostname), port $PORT.

1. On your LAPTOP, in a new terminal, open the tunnel (keep it open):

     make tunnel HOST=$(hostname) JUPITER_USER=$USER

   (or: ssh -N -L $PORT:$(hostname):$PORT $USER@login.jupiter.fz-juelich.de)

2. Below, find the line with  http://127.0.0.1:$PORT/lab?token=...
   In VS Code: kernel picker -> Existing Jupyter Server -> paste
     http://localhost:$PORT/?token=<the token>
   then choose the kernel "rmu ($USER)".

3. When done: scancel $SLURM_JOB_ID   (the node costs budget until then)
=====================================================================

EOF

srun python -m jupyterlab --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.port_retries=0
