import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sam3.model_builder import build_sam3_image_model

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

IMAGE_PATH = "/home/hensemberk/dev/ia_project/image_part_001.jpg"

CHECKPOINT_PATH = (
    "/home/hensemberk/dev/ia_project/logs/my_experiment/checkpoints/checkpoint_3.pt"
)

BPE_PATH = (
    "/home/hensemberk/dev/ia_project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
)

# ---------------------------------------------------
# Build model
# ---------------------------------------------------

model = build_sam3_image_model(
    bpe_path=BPE_PATH,
    device="cuda",
    eval_mode=True,
    enable_segmentation=True,
    checkpoint_path=CHECKPOINT_PATH,
)

model.eval()

print("Model loaded")

# ---------------------------------------------------
# Load image
# ---------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

image_np = np.array(image)

# ---------------------------------------------------
# Basic preprocessing
# ---------------------------------------------------

img = torch.from_numpy(image_np).float() / 255.0
img = img.permute(2, 0, 1).unsqueeze(0).cuda()

# Normalize (same as training)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

img = (img - mean) / std

# ---------------------------------------------------
# Forward pass
# ---------------------------------------------------

with torch.no_grad():

    outputs = model({
        "image": img
    })

print("\nOUTPUT KEYS:\n")

if isinstance(outputs, dict):
    for k in outputs.keys():
        print(k)
else:
    print(type(outputs))

print("\nDONE")