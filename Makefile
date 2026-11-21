.PHONY: setup test lint format cluster-pull env smoke submit rq2evaluate unlearn-eval jupyter tunnel jobs watch

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

# Copy results from your clone on JUPITER into this repo (scripts/cluster_pull.sh), e.g.
#   make cluster-pull                                            # all of results/
#   make cluster-pull WHAT=results/rq2/gender/lapa-12b-instruct
# Refuses to overwrite uncommitted local files unless FORCE=1. Uses the ssh alias `jupiter`
# (JUPITER_HOST=... for another one).
cluster-pull:
	@FORCE=$(FORCE) bash scripts/cluster_pull.sh $(WHAT)

# ---- JUPITER login node -------------------------------------------------

# One-time per person: create your Python env.
env:
	bash slurm/setup_env.sh

# Short GPU job that checks GPU, model loading, layer hook and scratch.
smoke:
	JOB_TIME=00:20:00 JOB_NAME=smoke slurm/submit.sh "smoke test" python scripts/smoke_test.py

# Any command as a GPU job, e.g.
#   make submit PURPOSE="pilot: RMU on Lapa" CMD="python -m rmu.unlearn --model_name_or_path lapa-12b-instruct"
# Optional: JOB_TIME=02:00:00 (default 01:00:00), JOB_CPUS=72 (CPU cores for the task;
# default: Slurm's, possibly 1; the node is billed whole either way), JOB_DEPENDENCY=afterok:<id>
# (start only after that job succeeded).
submit:
	@test -n "$(PURPOSE)" -a -n "$(CMD)" || \
		{ echo 'usage: make submit PURPOSE="..." CMD="python -m ..." [JOB_TIME=hh:mm:ss] [JOB_CPUS=n]'; exit 1; }
	JOB_TIME=$(or $(JOB_TIME),01:00:00) JOB_CPUS=$(JOB_CPUS) JOB_DEPENDENCY=$(JOB_DEPENDENCY) \
		slurm/submit.sh "$(PURPOSE)" $(CMD)

# RQ2 probing pipeline for one model and bias type (docs/rq2_probing.md): a GPU job for
# everything up to gate 2, then the long null in a second job that starts only if the first
# succeeded. Submitting the same line again continues an interrupted run.
#   make rq2 MODEL=qwen3-8b BIAS=race                 # both jobs
#   make rq2 MODEL=qwen3-8b BIAS=race STAGE=quick     # or STAGE=null
#   make rq2 MODEL=qwen3-8b BIAS=race STAGE=extract   # meta, extraction, gate 1: a short job
#   make rq2 MODEL=qwen3-8b BIAS=race DRY_RUN=1       # only print the steps and what is done
# STAGE is read from the command line only: the JUPITER Stages module exports STAGE=2026.
# Optional: QUICK_TIME=02:00:00, NULL_TIME=12:00:00, EXTRACT_TIME=00:45:00,
# NULL_PIECES=2 (null as 2 jobs in a row)
rq2:
	@test -n "$(BIAS)" || \
		{ echo 'usage: make rq2 MODEL=<model> BIAS=<bias> [STAGE=all|quick|null] [DRY_RUN=1]'; exit 1; }
	DRY_RUN=$(DRY_RUN) QUICK_TIME=$(QUICK_TIME) NULL_TIME=$(NULL_TIME) NULL_PIECES=$(NULL_PIECES) \
		EXTRACT_TIME=$(EXTRACT_TIME) \
		slurm/rq2.sh $(MODEL) $(BIAS) $(if $(filter command line,$(origin STAGE)),$(STAGE),all)

# Evaluate a model on the BBQ benchmark (defaults to MODEL=lapa-12b-instruct).
# Usage:
#   make evaluate
#   make evaluate MODEL=lapa-12b-instruct
# Optional: GENDER_TYPES="F M", JOB_TIME=01:00:00, EXTRA_ARGS="..."
MODEL ?= lapa-12b-instruct
evaluate:
	$(MAKE) submit PURPOSE="eval: BBQ on $(MODEL)" \
		CMD="python -m evaluation.bbq.evaluate --model_name_or_path $(MODEL)$(if $(GENDER_TYPES), --gender_types $(GENDER_TYPES))$(if $(EXTRA_ARGS), $(EXTRA_ARGS))" \
		JOB_TIME=$(or $(JOB_TIME),01:00:00)

# Adaptive RMU (split, words, beta 1, lr 5e-4, alpha 5, layer 32) + BBQ eval of the result,
# in one job (see docs/unlearn_eval.md). MODEL defaults to lapa-12b-instruct (set above).
#   make unlearn-eval MODEL=qwen3-8b
#   make unlearn-eval MODEL=gemma3-12b-it EXTRA_ARGS="--no-save_model"
# Optional: JOB_TIME=02:00:00 (default), EXTRA_ARGS="..." (any rmu.unlearn / eval option)
unlearn-eval:
	$(MAKE) submit PURPOSE="unlearn+eval: adaptive RMU + BBQ on $(MODEL)" \
		CMD="python -m evaluation.unlearn_eval --model_name_or_path $(MODEL)$(if $(EXTRA_ARGS), $(EXTRA_ARGS))" \
		JOB_TIME=$(or $(JOB_TIME),02:00:00)

# JupyterLab on a GPU node; the job log tells you how to connect.
jupyter:
	@mkdir -p /e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/logs
	@sbatch --time=$(or $(JOB_TIME),01:00:00) slurm/jupyter.sbatch

# Your jobs since this morning (or since SINCE=YYYY-MM-DD). The last column is Slurm's estimated
# start of pending jobs, or why there is none yet, e.g. (Dependency) or (Priority).
jobs:
	@bash slurm/jobs.sh $(SINCE)

# Follow a job's log live (Ctrl+C stops watching, not the job), e.g. make watch JOB=2274146
# LOSSES=1 shows only the loss / cosine lines.
watch:
	@test -n "$(JOB)" || { echo "usage: make watch JOB=<job id> [LOSSES=1]"; exit 1; }
	@# Wait while the job is queued or starting up; give up only if it left the queue without a log.
	@# While pending, show Slurm's reason and its estimated start (%S; N/A until the scheduler
	@# has planned the job, and it can move as other jobs finish early or arrive).
	@while ! log=$$(ls /e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/logs/*-$(JOB).out 2>/dev/null); do \
		info=$$(squeue -h -j $(JOB) -o '%T %S %r' 2>/dev/null); \
		[ -n "$$info" ] || { echo; echo "Job $(JOB) is not in the queue and has no log. Check: make jobs"; exit 1; }; \
		set -- $$info; state=$$1; start=$$2; reason=$$3; \
		if [ "$$state" != PENDING ]; then when="started $$(echo $$start | tr T ' ')"; \
		elif [ -z "$$start" ] || [ "$$start" = N/A ]; then when="no start estimate yet"; \
		else when="estimated start $$(echo $$start | tr T ' ')"; fi; \
		printf '\r%-100s' "Waiting for job $(JOB): $$state ($$reason), $$when, now $$(date +%H:%M:%S)"; \
		sleep 10; \
	done; \
	echo; echo "== $$log"; \
	if [ -n "$(LOSSES)" ]; then tail -n +1 -f "$$log" | grep --line-buffered -E "loss|cosine|nan|Error|Saved"; \
	else tail -n +1 -f "$$log"; fi
