"""
SAM3 LoRA Fine-tuned Inference - Single Image
==============================================
Uses Sam3Processor with the correct API:
  1. set_image()          — encode backbone features once
  2. set_text_prompt()    — query per category, returns state with boxes/masks/scores
  3. reset_all_prompts()  — clear between categories

Usage:
    python infer_single.py
    python infer_single.py --image /path/to/image.jpg --checkpoint /path/to/checkpoint.pt
"""

import argparse
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_IMAGE = "/home/hensemberk/dev/ia_project/image_part_001.jpg"
DEFAULT_CKPT  = "/home/hensemberk/dev/ia_project/logs/my_experiment/checkpoints/checkpoint.pt"
DEFAULT_BPE   = "/home/hensemberk/dev/ia_project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_OUT   = "/home/hensemberk/dev/ia_project/logs/inference_result.png"
RESOLUTION    = 1008
CONFIDENCE    = 0.3

# Your fine-tuning categories — must match training annotations exactly
CATEGORIES = [
    "bareland",
    "rangeland",
    "developed_space",
    "road",
    "tree",
    "water",
    "agriculture_land",
    "building",
]

# Colors per category (RGB 0-1)
COLORS = [
    (0.80, 0.60, 0.40),  # bareland        - tan
    (0.60, 0.80, 0.40),  # rangeland       - light green
    (0.55, 0.55, 0.55),  # developed_space - grey
    (0.25, 0.25, 0.25),  # road            - dark grey
    (0.10, 0.65, 0.10),  # tree            - green
    (0.10, 0.40, 0.90),  # water           - blue
    (0.90, 0.80, 0.10),  # agriculture     - yellow
    (0.90, 0.20, 0.20),  # building        - red
]


# ─────────────────────────────────────────────
# LoRA — must match training exactly
# ─────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.scale = alpha / r
        self.lora_A = nn.Linear(original.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = self.lora_A.to(original.weight.device)
        self.lora_B = self.lora_B.to(original.weight.device)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def apply_lora_to_model(model, r=16, alpha=32, dropout=0.1, target="attn.proj"):
    for p in model.parameters():
        p.requires_grad = False
    replaced = 0
    for name, module in list(model.named_modules()):
        if name.endswith(target) and isinstance(module, nn.Linear):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1],
                    LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
    print(f"  LoRA: replaced {replaced} layers (target='{target}', r={r}, alpha={alpha})")
    return model


# ─────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────

def load_finetuned_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    # strip DDP 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
    return model


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def visualize(image_pil, results, output_path):
    img_np = np.array(image_pil)
    h, w   = img_np.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle("SAM3 Fine-tuned — Urban Remote Sensing Inference", fontsize=14, y=1.01)

    axes[0].imshow(img_np)
    axes[0].set_title("Detections (boxes)")
    axes[0].axis("off")
    axes[1].imshow(img_np)
    axes[1].set_title("Segmentation masks")
    axes[1].axis("off")

    legend_patches = []
    seen = set()

    for cat_name, color, boxes, scores, masks in results:
        if not boxes:
            continue
        for box, score, mask in zip(boxes, scores, masks):
            x1, y1, x2, y2 = box
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x1, max(y1 - 4, 0),
                f"{cat_name} {score:.2f}",
                color='white', fontsize=7,
                bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor='none')
            )
            if mask is not None:
                colored = np.zeros((h, w, 4), dtype=float)
                colored[mask] = [*color, 0.45]
                axes[1].imshow(colored)

        if cat_name not in seen:
            legend_patches.append(mpatches.Patch(color=color, label=cat_name))
            seen.add(cat_name)

    if legend_patches:
        for ax in axes:
            ax.legend(handles=legend_patches, loc='upper right',
                      fontsize=8, framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Build base model (loads HF pretrained weights)
    print("\nBuilding SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=args.bpe,
        device=device,
        eval_mode=False,          # False so LoRA state dict loads correctly
        enable_segmentation=True,
        load_from_HF=True,
    )

    # 2. Apply LoRA structure (same as training)
    print("Applying LoRA adapters...")
    model = apply_lora_to_model(model, r=16, alpha=32, dropout=0.1, target="attn.proj")

    # 3. Load fine-tuned weights into LoRA model
    model = load_finetuned_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # 4. Build processor
    processor = Sam3Processor(
        model=model,
        resolution=RESOLUTION,
        device=device,
        confidence_threshold=args.confidence,
    )

    # 5. Load image
    print(f"\nLoading image: {args.image}")
    image_pil = Image.open(args.image).convert("RGB")
    print(f"  Size: {image_pil.size}")

    # 6. Encode image backbone features ONCE (expensive — only done once)
    print("Encoding image features...")
    state = processor.set_image(image_pil)

    # 7. Query each category with set_text_prompt()
    #    This runs the full grounding forward per category text
    print(f"\nRunning inference for {len(CATEGORIES)} categories...")
    results = []

    for idx, (cat_name, color) in enumerate(zip(CATEGORIES, COLORS)):
        print(f"  [{idx+1}/{len(CATEGORIES)}] '{cat_name}'...", end=" ", flush=True)

        try:
            # set_text_prompt encodes text + runs transformer + applies confidence filter
            # returns updated state with 'boxes', 'masks', 'scores'
            out_state = processor.set_text_prompt(cat_name, state)

            boxes  = []
            scores = []
            masks  = []

            if "boxes" in out_state and out_state["boxes"] is not None:
                b = out_state["boxes"].cpu().numpy()        # (N, 4) x1y1x2y2 pixels
                s = out_state["scores"].cpu().numpy().flatten()
                m = out_state["masks"].cpu().numpy()        # (N, 1, H, W) bool

                for i in range(len(b)):
                    boxes.append(b[i].tolist())
                    scores.append(float(s[i]))
                    masks.append(m[i, 0])                   # (H, W) bool

            print(f"{len(boxes)} detections")
            results.append((cat_name, color, boxes, scores, masks))

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append((cat_name, color, [], [], []))

        # Clear text prompt + geometric prompts before next category
        processor.reset_all_prompts(state)

    # 8. Summary
    print("\n── DETECTION SUMMARY ──────────────────────")
    total = 0
    for cat_name, _, boxes, scores, _ in results:
        n = len(boxes)
        total += n
        avg  = f"avg score {sum(scores)/n:.3f}" if n > 0 else ""
        flag = "✓" if n > 0 else " "
        print(f"  {flag} {cat_name:<22} {n:>3} detections  {avg}")
    print(f"  {'TOTAL':<24} {total:>3} detections")
    print("────────────────────────────────────────────")

    # 9. Visualise and save
    visualize(image_pil, results, args.output)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM3 LoRA Single-Image Inference")
    parser.add_argument("--image",      default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--bpe",        default=DEFAULT_BPE)
    parser.add_argument("--output",     default=DEFAULT_OUT)
    parser.add_argument("--confidence", default=CONFIDENCE, type=float,
                        help="Confidence threshold 0-1 (lower = more detections)")
    args = parser.parse_args()
    main(args)