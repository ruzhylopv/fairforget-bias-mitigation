# Slurm Cheatsheet

Slurm is the workload manager used to request and manage compute resources on JUPITER.

---

## 1. Check Available Resources

### Show partitions and nodes

```bash
sinfo
```

More detailed:

```bash
sinfo -o "%P %a %l %D %G"
```

Useful for checking available partitions, time limits, node counts, and GPUs.

---

## 2. Interactive GPU Allocation

For interactive work and Jupyter notebooks, request a GPU allocation:

```bash
salloc --partition=booster --gpus=1 --time=00:30:00
```

After Slurm grants the allocation, you are placed on a compute node.

Check the allocation:

```bash
echo $SLURM_JOB_ID
hostname
nvidia-smi
```

Check the job status:

```bash
squeue -j $SLURM_JOB_ID
```

### Important

The `--time` argument is the **maximum wall-clock duration of the allocation**.

For example:

```bash
--time=00:30:00
```

means the allocation can exist for up to 30 minutes. It does not mean 30 minutes of actual GPU computation.

---

## 3. Check Your Jobs

Show your running and pending jobs:

```bash
squeue -u $USER
```

Show information about the current job:

```bash
scontrol show job $SLURM_JOB_ID
```

Compact view:

```bash
squeue -j $SLURM_JOB_ID \
  -o "%.18i %.9P %.20j %.8T %.10M %.10L %.6D %R"
```

Important fields include:

| Field        | Meaning              |
| ------------ | -------------------- |
| `JOBID`      | Job identifier       |
| `PARTITION`  | Partition being used |
| `NAME`       | Job name             |
| `ST`         | Job state            |
| `TIME`       | Time elapsed         |
| `TIME_LIMIT` | Maximum allowed time |
| `NODELIST`   | Allocated node       |

---

## 5. Cancel a Job

Cancel a specific job:

```bash
scancel <JOB_ID>
```

Cancel all of your jobs:

```bash
scancel -u $USER
```

Use the latter carefully.

---

## 6. End an Interactive Allocation

When you are finished with an interactive `salloc` session:

```bash
exit
```

This releases the allocation.


Verify that the job is gone:

```bash
squeue -u $USER
```

If an allocation reaches its time limit, Slurm will terminate it automatically.

---

## 7. Interactive vs Batch Jobs

### `salloc`

Request resources for an **interactive session**.

```bash
salloc --partition=booster --gpus=1 --time=00:30:00
```

Useful for:

* Jupyter notebooks
* debugging
* experimenting interactively
* testing GPU code

---

### `srun`

Run a command using Slurm resources.

```bash
srun python script.py
```

Useful for:

* launching individual tasks
* running commands inside an allocation
* testing programs

---

### `sbatch`

Submit a script as a non-interactive job.

```bash
sbatch job.sh
```

Example `job.sh`:

```bash
#!/bin/bash

#SBATCH --partition=booster
#SBATCH --gpus=1
#SBATCH --time=01:00:00

module load Python/3.13.5
module load PyTorch/2.9.1

uv run python train.py
```

Submit it:

```bash
sbatch job.sh
```

Check its status:

```bash
squeue -u $USER
```

---

## 8. Job States

Common Slurm states:

| State | Meaning                         |
| ----- | ------------------------------- |
| `PD`  | Pending — waiting for resources |
| `R`   | Running                         |
| `CG`  | Completing                      |
| `CD`  | Completed                       |
| `F`   | Failed                          |
| `CA`  | Cancelled                       |
| `TO`  | Timed out                       |

For a pending job, check why:

```bash
squeue -j <JOB_ID> -o "%.18i %.8T %R"
```

---

## 9. Check a Completed Job

After a job finishes:

```bash
sacct -j <JOB_ID>
```

More detailed:

```bash
sacct -j <JOB_ID> \
  --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS
```

This is useful for checking how much time and memory a job actually used.

---

## 10. GPU Check

Inside a GPU allocation:

```bash
nvidia-smi
```

For PyTorch:

```bash
uv run python -c \
"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output should indicate that CUDA is available and show the allocated GPU.

---

## 11. Useful Environment Variables

Slurm automatically provides information about your job through environment variables.

```bash
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_JOB_PARTITION
echo $SLURM_JOB_NODELIST
echo $SLURM_JOB_GPUS
```

The most commonly useful one is:

```bash
echo $SLURM_JOB_ID
```

---

### Mental Model

```text
salloc  → "Give me resources"
srun    → "Run this command using Slurm"
sbatch  → "Run this script as a job"
squeue  → "What are my jobs doing?"
scontrol → "Give me detailed job information"
sacct   → "What happened to my finished job?"
scancel → "Stop my job"
```
