.PHONY: setup test lint format env smoke submit jupyter tunnel jobs

# ---- Laptop -------------------------------------------------------------

setup:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# Open the SSH tunnel to a Jupyter job (run on the laptop; HOST is printed in the job log).
JUPITER_LOGIN ?= login.jupiter.fz-juelich.de
tunnel:
	@test -n "$(HOST)" -a -n "$(JUPITER_USER)" || \
		{ echo "usage: make tunnel HOST=<compute node> JUPITER_USER=<cluster username>"; exit 1; }
	ssh -4 -N -L 8888:$(HOST):8888 $(JUPITER_USER)@$(JUPITER_LOGIN)

# ---- JUPITER login node -------------------------------------------------

# One-time per person: create your Python env.
env:
	bash slurm/setup_env.sh

# Short GPU job that checks GPU, model loading, layer hook and scratch.
smoke:
	JOB_TIME=00:20:00 JOB_NAME=smoke slurm/submit.sh "smoke test" python scripts/smoke_test.py

# Any command as a GPU job, e.g.
#   make submit PURPOSE="pilot: RMU on Lapa" CMD="python -m rmu.unlearn --model_name_or_path lapa-12b-instruct"
# Optional: JOB_TIME=02:00:00 (default 01:00:00)
submit:
	@test -n "$(PURPOSE)" -a -n "$(CMD)" || \
		{ echo 'usage: make submit PURPOSE="..." CMD="python -m ..." [JOB_TIME=hh:mm:ss]'; exit 1; }
	JOB_TIME=$(or $(JOB_TIME),01:00:00) slurm/submit.sh "$(PURPOSE)" $(CMD)

# JupyterLab on a GPU node; the job log tells you how to connect.
jupyter:
	@mkdir -p /e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/logs
	@sbatch --time=$(or $(JOB_TIME),01:00:00) slurm/jupyter.sbatch

# Your jobs since this morning (or since SINCE=YYYY-MM-DD).
jobs:
	@sacct -u $$USER -S $(or $(SINCE),$$(date +%F)) -X \
		--format=JobID,JobName%12,Start,Elapsed,NNodes,NodeList%14,State,ExitCode
