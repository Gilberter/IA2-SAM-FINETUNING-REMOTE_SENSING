import json
import os

# Paths
ANNOTATION_JSON = "/disk/SN-NVS-2026-raw/hsi_dataset/dataset/annotations.json"

# Load COCO annotation file
with open(ANNOTATION_JSON, "r") as f:
    coco = json.load(f)

# Sort images by COCO image id
sorted_images = sorted(coco["images"], key=lambda x: x["id"])

# Create mapping and update annotation.json
mapping = {}

for idx, image_info in enumerate(sorted_images):

    old_name = image_info["file_name"]

    # New png filename
    new_name = f"img_{idx}.jpg"

    # Store mapping
    mapping[new_name] = {
        "id": image_info["id"],
        "original_file": old_name
    }

    # IMPORTANT:
    # Change file_name inside annotation.json
    image_info["file_name"] = new_name

# Save updated annotation file
UPDATED_JSON = "/disk/SN-NVS-2026-raw/hsi_dataset/dataset/annotations_updated.json"

with open(UPDATED_JSON, "w") as f:
    json.dump(coco, f, indent=2)

# Save mapping file
MAPPING_JSON = "/disk/SN-NVS-2026-raw/hsi_dataset/dataset/image_id_mapping.json"

with open(MAPPING_JSON, "w") as f:
    json.dump(mapping, f, indent=2)

# Print mapping
for k, v in mapping.items():
    print(f"{v['original_file']}  -->  {k}")

print("\nUpdated annotation file saved to:")
print(UPDATED_JSON)

print("\nMapping file saved to:")
print(MAPPING_JSON)