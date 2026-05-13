"""
SAM3 Ablation Inference — Base vs Fine-tuned
=============================================
Runs inference with TWO model variants on the same image:
  1. Base SAM3 weights (pretrained only)
  2. Fine-tuned SAM3 + LoRA weights

Produces a single 3-panel figure:
  [Original Image] | [Base SAM3 Segmentation] | [Fine-tuned SAM3 Segmentation]

Each panel shows a semantic segmentation map where each pixel is colored
by its highest-scoring category. Background / no detection = black.

Usage:
    python infer_ablation.py --image /path/to/image.jpg
    python infer_ablation.py --image /path/to/image.jpg --confidence 0.2
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
# PATHS
# ─────────────────────────────────────────────

BASE_WEIGHTS  = "/home/hensemberk/dev/ia_project/weights/sam3.pt"
LORA_WEIGHTS  = "/home/hensemberk/dev/ia_project/logs/my_experiment/checkpoints/checkpoint_11.pt"
BPE_PATH      = "/home/hensemberk/dev/ia_project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_IMAGE = "/home/hensemberk/dev/ia_project/image_part_001.jpg"
DEFAULT_OUT   = "/home/hensemberk/dev/ia_project/logs/ablation_result.png"
RESOLUTION    = 1008
CONFIDENCE    = 0.25   # same for both models

# ─────────────────────────────────────────────
# CATEGORIES & COLORS
# Fine-tuning categories (id 1-8) + road variants
# Black (0,0,0) = background / no detection
# ─────────────────────────────────────────────

CATEGORIES = [
    "bareland",        # id 1
    "rangeland",       # id 2
    "developed_space", # id 3
    "road",            # id 4
    "tree",            # id 5
    "water",           # id 6
    "agriculture_land",# id 7
    "building",        # id 8
]

# Vivid, clearly distinguishable colors per category (RGB 0-1)
# Background pixels stay black
CAT_COLORS = {
    "bareland":         (0.82, 0.62, 0.35),  # warm tan
    "rangeland":        (0.45, 0.78, 0.30),  # grass green
    "developed_space":  (0.65, 0.65, 0.65),  # medium grey
    "road":             (0.20, 0.20, 0.20),  # near-black grey
    "tree":             (0.05, 0.50, 0.05),  # dark green
    "water":            (0.10, 0.45, 0.95),  # vivid blue
    "agriculture_land": (0.95, 0.85, 0.10),  # golden yellow
    "building":         (0.90, 0.15, 0.15),  # vivid red
}


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


def apply_lora(model, r=16, alpha=32, dropout=0.1, target="attn.proj"):
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
    print(f"    LoRA: replaced {replaced} layers")
    return model


# ─────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────

def build_base_model(device):
    """Load SAM3 with original pretrained weights only."""
    print("  Building base SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        load_from_HF=True,          # loads pretrained weights from HuggingFace
        checkpoint_path=None,
    )
    print(f"  Base model ready (no fine-tuning)")
    return model


def build_lora_model(device):
    """Load SAM3 with LoRA adapters + fine-tuned weights."""
    print("  Building LoRA fine-tuned model...")
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        device=device,
        eval_mode=False,            # keep False so state_dict loads cleanly
        enable_segmentation=True,
        load_from_HF=True,          # start from pretrained, then override with fine-tuned
    )
    model = apply_lora(model, r=16, alpha=32, dropout=0.1, target="attn.proj")

    print(f"  Loading fine-tuned checkpoint: {LORA_WEIGHTS}")
    ckpt = torch.load(LORA_WEIGHTS, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    Missing keys ({len(missing)}): {missing[:2]}...")
    if unexpected:
        print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:2]}...")
    print(f"    Loaded from epoch {ckpt.get('epoch', '?')}")

    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference(model, image_pil, device, confidence):
    """
    Run all categories through the processor.
    Returns a dict: {cat_name: (masks_list, scores_list)}
    """
    processor = Sam3Processor(
        model=model,
        resolution=RESOLUTION,
        device=device,
        confidence_threshold=confidence,
    )

    state = processor.set_image(image_pil)
    results = {}

    for cat_name in CATEGORIES:
        try:
            out = processor.set_text_prompt(cat_name, state)

            masks_list  = []
            scores_list = []

            if "masks" in out and out["masks"] is not None and len(out["masks"]) > 0:
                m = out["masks"].cpu().numpy()   # (N, 1, H, W) bool
                s = out["scores"].cpu().numpy().flatten()
                for i in range(len(m)):
                    masks_list.append(m[i, 0])   # (H, W)
                    scores_list.append(float(s[i]))

            results[cat_name] = (masks_list, scores_list)

        except Exception as e:
            print(f"    WARNING: error on '{cat_name}': {e}")
            results[cat_name] = ([], [])

        processor.reset_all_prompts(state)

    return results


# ─────────────────────────────────────────────
# BUILD SEGMENTATION MAP
# Each pixel gets the color of the highest-scoring category mask that covers it.
# Background = black.
# ─────────────────────────────────────────────

def build_seg_map(results, orig_h, orig_w):
    """
    Returns:
        seg_rgb  : (H, W, 3) float32  0-1  — colored segmentation map
        score_map: (H, W)    float32        — max score at each pixel
    """
    seg_rgb   = np.zeros((orig_h, orig_w, 3), dtype=np.float32)  # black background
    score_map = np.zeros((orig_h, orig_w),    dtype=np.float32)

    for cat_name in CATEGORIES:
        color = CAT_COLORS[cat_name]
        masks_list, scores_list = results.get(cat_name, ([], []))

        for mask, score in zip(masks_list, scores_list):
            # mask may be at model resolution — resize to original if needed
            if mask.shape != (orig_h, orig_w):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                    (orig_w, orig_h), PILImage.NEAREST
                )
                mask = np.array(mask_pil) > 127

            # Only update pixels where this score beats the current best
            better = mask & (score > score_map)
            seg_rgb[better]   = color
            score_map[better] = score

    return seg_rgb, score_map


# ─────────────────────────────────────────────
# DETECTION SUMMARY PRINT
# ─────────────────────────────────────────────

def print_summary(label, results):
    print(f"\n  ── {label} ─────────────────────────")
    total = 0
    for cat_name in CATEGORIES:
        masks_list, scores_list = results.get(cat_name, ([], []))
        n = len(masks_list)
        total += n
        avg = f"avg {sum(scores_list)/n:.3f}" if n > 0 else ""
        flag = "✓" if n > 0 else " "
        print(f"    {flag} {cat_name:<22} {n:>3}  {avg}")
    print(f"    {'TOTAL':<24} {total:>3}")


# ─────────────────────────────────────────────
# VISUALISE — 3-panel figure
# ─────────────────────────────────────────────

def visualize(image_pil, results_base, results_lora, output_path, confidence):
    img_np = np.array(image_pil)
    orig_h, orig_w = img_np.shape[:2]

    seg_base, _ = build_seg_map(results_base, orig_h, orig_w)
    seg_lora, _ = build_seg_map(results_lora, orig_h, orig_w)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor('#1a1a1a')

    titles = [
        "Original Image",
        f"Base SAM3\n(pretrained only, conf={confidence})",
        f"Fine-tuned SAM3 + LoRA\n(epoch 11, conf={confidence})",
    ]
    panels = [img_np, seg_base, seg_lora]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color='black', label='background (no detection)')
    ]
    for cat_name in CATEGORIES:
        color = CAT_COLORS[cat_name]
        legend_patches.append(mpatches.Patch(color=color, label=cat_name))

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=5,
        fontsize=9,
        framealpha=0.85,
        facecolor='#2a2a2a',
        labelcolor='white',
        edgecolor='grey',
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "SAM3 Ablation Study — Base vs Fine-tuned (Urban Remote Sensing)",
        color='white', fontsize=13, y=1.01
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {output_path}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Image: {args.image}\n")

    # Load image
    image_pil = Image.open(args.image).convert("RGB")
    print(f"Image size: {image_pil.size}")

    # ── Model 1: Base SAM3 ───────────────────
    print("\n[1/2] BASE SAM3 (pretrained only)")
    base_model = build_base_model(device)
    print("  Running inference...")
    results_base = run_inference(base_model, image_pil, device, args.confidence)
    print_summary("Base SAM3", results_base)
    del base_model
    torch.cuda.empty_cache()

    # ── Model 2: Fine-tuned LoRA ─────────────
    print("\n[2/2] FINE-TUNED SAM3 + LoRA")
    lora_model = build_lora_model(device)
    print("  Running inference...")
    results_lora = run_inference(lora_model, image_pil, device, args.confidence)
    print_summary("Fine-tuned LoRA", results_lora)
    del lora_model
    torch.cuda.empty_cache()

    # ── Visualise ────────────────────────────
    print("\nGenerating ablation figure...")
    visualize(image_pil, results_base, results_lora, args.output, args.confidence)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM3 Ablation — Base vs Fine-tuned")
    parser.add_argument("--image",      default=DEFAULT_IMAGE,
                        help="Path to input image")
    parser.add_argument("--output",     default=DEFAULT_OUT,
                        help="Path to save output figure")
    parser.add_argument("--confidence", default=CONFIDENCE, type=float,
                        help="Confidence threshold (same for both models)")
    args = parser.parse_args()
    main(args)