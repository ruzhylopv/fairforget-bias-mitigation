from huggingface_hub import snapshot_download

MODEL_ID = "lapa-llm/lapa-12b-pt"

MODEL_DIR = "../models/lapa-12b-pt"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
)

print(f"Model downloaded to: {MODEL_DIR}")