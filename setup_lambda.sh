#!/usr/bin/env bash
# setup_lambda.sh — Run on Lambda instance after SCP.
# Installs deps, downloads model + dataset.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== CLR POC — Lambda Setup ==="

# JN: avoid errors
export USE_TF=0

echo "[1/4] Installing Python dependencies..."
pip install --upgrade pip
pip install --upgrade Pillow
pip install \
    torch \
    "transformers>=4.44,<4.47" \
    "datasets>=2.19" \
    "peft>=0.12,<0.14" \
    "accelerate>=0.33" \
    "bitsandbytes>=0.43" \
    "trl>=0.11,<0.14" \
    spacy pandas matplotlib seaborn

echo "[2/4] Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "[3/4] Downloading Qwen 2.5-3B-Instruct..."
mkdir -p ./models
python - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen2.5-3B-Instruct"
save_path = "./models/qwen25-3b-instruct"

if os.path.isfile(os.path.join(save_path, "config.json")):
    print(f"  Model already exists at {save_path}, skipping.")
else:
    print(f"  Downloading {model_id} ...")
    snapshot_download(repo_id=model_id, local_dir=save_path)
    print(f"  Saved to {save_path}")
PYEOF

echo "[4/4] Downloading Anthropic HH-RLHF..."
mkdir -p ./data
python - <<'PYEOF'
from datasets import load_dataset
import os

save_path = "./data/hh-rlhf"

if os.path.isdir(save_path) and os.path.isfile(os.path.join(save_path, "dataset_dict.json")):
    print(f"  Dataset already exists at {save_path}, skipping.")
else:
    print("  Downloading Anthropic/hh-rlhf ...")
    ds = load_dataset("Anthropic/hh-rlhf")
    ds.save_to_disk(save_path)
    print(f"  Saved to {save_path}")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "  Verify GPU: python -c \"import torch; print(torch.cuda.get_device_name(0))\""
echo "  Run:        bash run_experiment.sh"
