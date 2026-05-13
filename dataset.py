from datasets import load_dataset
import os
dataset = load_dataset("CrisTaf/sementationAi")

print(dataset)

os.makedirs("/disk/SN-NVS-2026-raw/hsi_dataset/train/images", exist_ok=True)
for i, example in enumerate(dataset['train']):
    example['image'].save(f"/disk/SN-NVS-2026-raw/hsi_dataset/train/images/img_{i}.jpg")
    
dataset.cleanup_cache_files()

from huggingface_hub import snapshot_download

# This downloads the entire dataset to a local folder
dataset_path = snapshot_download(repo_id="mick2332/inferenciIA", repo_type="dataset",local_dir="/disk/SN-NVS-2026-raw/hsi_dataset/test")
print(f"Dataset downloaded to: {dataset_path}")
