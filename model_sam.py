import os
import torch
from PIL import Image
from dotenv import load_dotenv
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from huggingface_hub import login
from torchvision.transforms import ToTensor

load_dotenv()
access_token = os.getenv("access_token")

login(token=access_token)

import matplotlib.pyplot as plt
import numpy as np

def show_results(image, masks, boxes, scores, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for mask, box, score in zip(masks, boxes, scores):
        # Convert mask to numpy and show as semi-transparent overlay
        m = mask[0].cpu().numpy()
        mask_image = np.zeros((*m.shape, 4))
        mask_image[m > 0] = [1, 0, 0, 0.35] # Red mask with 35% opacity
        plt.gca().imshow(mask_image)
        
        # Draw Bounding Box
        x1, y1, x2, y2 = box.cpu().numpy()
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue', linewidth=2))
        plt.text(x1, y1, f"{score:.2f}", color='white', backgroundcolor='blue')

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Model on {device}...")
# Ensure you provide the checkpoint path or rely on default
model = build_sam3_image_model().to(device).to(torch.bfloat16)
model.eval() # Set to evaluation mode

print(f"Model dtype: {next(model.parameters()).dtype}")

print(f"Loading Processor...")
processor = Sam3Processor(model)

# Load and convert image to RGB (crucial step)
image_path = "/home/hensemberk/dev/ia_project/image_part_001.jpg"
image = Image.open(image_path).convert("RGB")

image_tensor = ToTensor()(image).to(device).to(torch.bfloat16)
# Prepare image features (runs the vision encoder once)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    inference_state = processor.set_image(image_tensor)

    # Run text prompt inference
    # Note: You can also use a list for better concept separation
    prompts = ["green zone", "buildings"]

    all_masks = []
    all_boxes = []
    all_scores = []
    all_labels = []

    # 1. Collect results from all prompts
    for p in prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=p)
        num_found = len(output['scores'])
        print(f"Prompt '{p}' found {num_found} instances.")
        
        if num_found > 0:
            all_masks.extend(output["masks"])
            all_boxes.extend(output["boxes"])
            all_scores.extend(output["scores"])
            all_labels.extend([p] * num_found) # Keep track of which text matched

# 2. Update your function to handle labels
def show_results_v2(image, masks, boxes, scores, labels, output_path):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    
    for mask, box, score, label in zip(masks, boxes, scores, labels):
        # Mask overlay
        m = mask[0].cpu().numpy()
        mask_rgba = np.zeros((*m.shape, 4))
        # Use different colors for different labels if you like
        color = [1, 0, 0, 0.35] if label == "buildings" else [0, 1, 0, 0.35]
        mask_rgba[m > 0] = color
        ax.imshow(mask_rgba)
        
        # Bounding Box
        x1, y1, x2, y2 = box.cpu().numpy()
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='yellow', linewidth=1.5))
        
        # Text and Score Label
        caption = f"{label}: {score:.2f}"
        ax.text(x1, y1 - 5, caption, color='white', fontsize=8, 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() # Close to free memory
    print(f"Final output saved to {output_path}")

# 3. Run it
if len(all_scores) > 0:
    show_results_v2(image, all_masks, all_boxes, all_scores, all_labels, "final_detection.png")
