"""Check that a GPU job can do everything the pilot needs. Submit with `make smoke`.

Checks, in order: GPU visible, model folder found, model and tokenizer load, the layers and
parameters RMU trains exist, a forward pass with the layer hook works, and the run folder on
scratch is writable. Each step prints OK or fails with a message.
"""

import os
import socket
import sys
import time

import torch

from rmu.config import resolve_model, run_dir
from rmu.utils import (
    forward_with_cache,
    get_decoder_layers,
    get_hidden_size,
    get_params,
    load_model,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "lapa-12b-instruct"
LAYER_ID = 7
LAYER_IDS = [5, 6, 7]
PARAM_NAMES = ["mlp.down_proj.weight"]


def step(name):
    print(f"\n== {name}", flush=True)


step("GPU")
print("host:", socket.gethostname())
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FAIL: no GPU visible. Is this running inside a Slurm job with --gres=gpu?")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 1024**3:.0f} GB")

step(f"Model folder for {MODEL!r}")
path = resolve_model(MODEL)
print("path:", path)
if not os.path.isfile(os.path.join(path, "config.json")):
    sys.exit(f"FAIL: no config.json in {path}. Check the models section of config/default.yaml.")

step("Load model and tokenizer")
start = time.time()
model, tokenizer = load_model(MODEL)
print(f"loaded {type(model).__name__} in {time.time() - start:.0f}s, dtype {model.dtype}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

step("Layers and trained parameters")
layers = get_decoder_layers(model)
print(f"{len(layers)} decoder layers, hidden size {get_hidden_size(model)}")
params = get_params(model, LAYER_IDS, PARAM_NAMES)
print(
    f"trainable for RMU: {PARAM_NAMES} in layers {LAYER_IDS} -> shapes",
    [tuple(p.shape) for p in params],
)

step(f"Forward pass with hook on layer {LAYER_ID}")
inputs = tokenizer(["The nurse said she was tired."], return_tensors="pt").to(model.device)
activations = forward_with_cache(model, inputs, module=layers[LAYER_ID], no_grad=True)
print("activations:", tuple(activations.shape), activations.dtype)
if not torch.isfinite(activations).all():
    sys.exit("FAIL: activations contain NaN or inf")

step("Write to the run folder")
out = run_dir()
out.mkdir(parents=True, exist_ok=True)
(out / "smoke_test.ok").write_text(f"{MODEL} {path}\n")
print("wrote", out / "smoke_test.ok")

print("\nAll checks passed.")
