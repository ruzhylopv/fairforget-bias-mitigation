#!/bin/bash

#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1

module load PyTorch/2.9.1

python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

    x = torch.rand(15, 15, device="cuda")
    y = x @ x

    print("GPU computation: OK")
    print("Result shape:", y.shape)
PY