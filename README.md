# FairForget Unlearning

Research code for studying fairness-aware machine unlearning. This repository is being set up for reproducible experiments on a Slurm HPC cluster.

## Status

The repository currently contains the project scaffold and a Slurm GPU smoke check. Experiment pipelines, datasets, and reported results have not yet been added.

## Quick start

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
make setup
make lint
```

For a local GPU check, activate the environment and run `python -c 'import torch; print(torch.cuda.is_available())'` after installing a PyTorch build compatible with your CUDA setup. On the cluster, see [HPC setup](docs/hpc.md).

## Repository layout

```text
config/       Experiment configuration
data/         Dataset documentation; generated data is not committed
docs/         Project and HPC documentation
notebooks/    Exploratory analysis
results/      Experiment outputs and reports
scripts/      Reproducible commands and scheduler jobs
src/rmu/      Python package
tests/        Automated checks
```

## Reproducibility

Python requirements and development tools are declared in `pyproject.toml`; `uv.lock` records the resolved environment. Record the configuration, random seed, code revision, and software/hardware environment for each experiment. Keep large datasets, checkpoints, and transient outputs out of Git; document their sources and checksums.

## License

See [LICENSE](LICENSE).
