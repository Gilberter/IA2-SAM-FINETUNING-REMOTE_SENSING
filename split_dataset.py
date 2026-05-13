import os
import json
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = "/disk/SN-NVS-2026-raw/hsi_dataset/dataset"

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "annotations_updated.json")

TRAIN_DIR = "/disk/SN-NVS-2026-raw/hsi_dataset/train"
VAL_DIR = "/disk/SN-NVS-2026-raw/hsi_dataset/val"

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")

TRAIN_ANNOTATIONS = os.path.join(TRAIN_DIR, "annotations.json")
VAL_ANNOTATIONS = os.path.join(VAL_DIR, "annotations.json")

# ============================================================
# SETTINGS
# ============================================================

VAL_SPLIT = 0.2
RANDOM_SEED = 42

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)

# ============================================================
# LOAD COCO ANNOTATIONS
# ============================================================

with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# ============================================================
# SPLIT IMAGE IDS
# ============================================================

image_ids = [img["id"] for img in images]

train_ids, val_ids = train_test_split(
    image_ids,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_ids = set(train_ids)
val_ids = set(val_ids)

# ============================================================
# SPLIT IMAGES
# ============================================================

train_images = []
val_images = []
# /disk/SN-NVS-2026-raw/hsi_dataset/dataset/images
# 
for img in images:
    if img["id"] in train_ids:
        train_images.append(img)
    else:
        val_images.append(img)

# ============================================================
# SPLIT ANNOTATIONS
# ============================================================

train_annotations = []
val_annotations = []

for ann in annotations:
    if ann["image_id"] in train_ids:
        train_annotations.append(ann)
    else:
        val_annotations.append(ann)

# ============================================================
# COPY IMAGE FILES
# ============================================================

def copy_images(images_list, dst_dir):

    for img in images_list:

        filename = img["file_name"]

        src = os.path.join(IMAGES_DIR, filename)
        dst = os.path.join(dst_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"[WARNING] Missing image: {src}")

print("Copying train images...")
copy_images(train_images, TRAIN_IMAGES_DIR)

print("Copying val images...")
copy_images(val_images, VAL_IMAGES_DIR)

# ============================================================
# CREATE TRAIN COCO
# ============================================================

train_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "categories": coco.get("categories", []),
    "images": train_images,
    "annotations": train_annotations
}

# ============================================================
# CREATE VAL COCO
# ============================================================

val_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "categories": coco.get("categories", []),
    "images": val_images,
    "annotations": val_annotations
}

# ============================================================
# SAVE JSON FILES
# ============================================================

with open(TRAIN_ANNOTATIONS, "w") as f:
    json.dump(train_coco, f, indent=2)

with open(VAL_ANNOTATIONS, "w") as f:
    json.dump(val_coco, f, indent=2)

# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("Dataset split completed")
print("===================================")

print(f"Train images: {len(train_images)}")
print(f"Val images:   {len(val_images)}")

print(f"Train annotations: {len(train_annotations)}")
print(f"Val annotations:   {len(val_annotations)}")

print("\nSaved:")
print(TRAIN_DIR)
print(VAL_DIR)