.PHONY: setup lint format gpu-check

setup:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

gpu-check:
	sbatch scripts/test_gpu.sh
