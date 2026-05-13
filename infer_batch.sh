#!/bin/bash
#SBATCH --job-name=gs_hyperspectral_infer
#SBATCH --account=gs_hyperspectral
#SBATCH --output=./logs/infer_%j.out
#SBATCH --error=./logs/infer_%j.log

#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time=03:00:00

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHECKPOINT="./logs/my_experiment/checkpoints/checkpoint_11.pt"
BPE="/home/hensemberk/dev/ia_project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CONFIDENCE=0.3
OUTPUT_BASE="./logs/inference_results"

TEST_DIRS=(
    "/disk/SN-NVS-2026-raw/hsi_dataset/test/cabeceratitles"
    "/disk/SN-NVS-2026-raw/hsi_dataset/test/centrotitles"
    "/disk/SN-NVS-2026-raw/hsi_dataset/test/uistitles"
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate sam3ju

cd /home/hensemberk/dev/ia_project

mkdir -p ./logs
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start:      $(date)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Checkpoint: $CHECKPOINT"
echo "Confidence: $CONFIDENCE"
echo "========================================"

# ─────────────────────────────────────────────
# ITERATE OVER TEST FOLDERS AND IMAGES
# ─────────────────────────────────────────────

total_images=0
processed=0
failed=0

# Count total images first
for dir in "${TEST_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
        total_images=$((total_images + count))
    fi
done

echo "Total images to process: $total_images"
echo ""

for dir in "${TEST_DIRS[@]}"; do

    if [ ! -d "$dir" ]; then
        echo "WARNING: Directory not found, skipping: $dir"
        continue
    fi

    # Folder name for output subfolder (e.g. cabeceratitles)
    folder_name=$(basename "$dir")
    output_dir="$OUTPUT_BASE/$folder_name"
    mkdir -p "$output_dir"

    echo "──────────────────────────────────────────"
    echo "Processing folder: $folder_name"
    echo "  Input:  $dir"
    echo "  Output: $output_dir"
    echo "──────────────────────────────────────────"

    # Find all images in this folder (non-recursive, one level)
    while IFS= read -r image_path; do

        image_name=$(basename "$image_path")
        image_stem="${image_name%.*}"
        output_path="$output_dir/${image_stem}_result.png"

        # Skip if already processed (useful for resuming)
        if [ -f "$output_path" ]; then
            echo "  [SKIP] Already exists: $image_name"
            processed=$((processed + 1))
            continue
        fi

        echo "  [$(( processed + 1 ))/$total_images] $image_name"

        python inference.py \
            --image      "$image_path" \
            --checkpoint "$CHECKPOINT" \
            --bpe        "$BPE" \
            --output     "$output_path" \
            --confidence "$CONFIDENCE"

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            processed=$((processed + 1))
        else
            echo "  ERROR: Failed on $image_name (exit code $exit_code)"
            failed=$((failed + 1))
        fi

    done < <(find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | sort)

    echo ""
done

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

echo "========================================"
echo "INFERENCE COMPLETE"
echo "  Processed:  $processed"
echo "  Failed:     $failed"
echo "  Output dir: $OUTPUT_BASE"
echo "  End:        $(date)"
echo "========================================"

# List output files per folder
echo ""
echo "Output summary:"
for dir in "${TEST_DIRS[@]}"; do
    folder_name=$(basename "$dir")
    output_dir="$OUTPUT_BASE/$folder_name"
    if [ -d "$output_dir" ]; then
        n=$(find "$output_dir" -name "*.png" | wc -l)
        echo "  $folder_name: $n result images"
    fi
done