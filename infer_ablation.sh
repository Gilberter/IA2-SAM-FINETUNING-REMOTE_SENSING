#!/bin/bash
#SBATCH --job-name=gs_hyperspectral_ablation
#SBATCH --account=gs_hyperspectral
#SBATCH --output=./logs/ablation_%j.out
#SBATCH --error=./logs/ablation_%j.log

#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time=06:00:00

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CONFIDENCE=0.25
OUTPUT_BASE="./logs/ablation_results"

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
echo "Confidence: $CONFIDENCE"
echo "Output:     $OUTPUT_BASE"
echo "========================================"

# ─────────────────────────────────────────────
# COUNT TOTAL
# ─────────────────────────────────────────────

total_images=0
for dir in "${TEST_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -maxdepth 1 -type f \
            \( -iname "*.jpg" -o -iname "*.jpeg" \
               -o -iname "*.png" \
               -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
        total_images=$((total_images + count))
    fi
done
echo "Total images: $total_images"
echo ""

# ─────────────────────────────────────────────
# ITERATE
# ─────────────────────────────────────────────

processed=0
failed=0

for dir in "${TEST_DIRS[@]}"; do

    if [ ! -d "$dir" ]; then
        echo "WARNING: not found, skipping: $dir"
        continue
    fi

    folder_name=$(basename "$dir")
    output_dir="$OUTPUT_BASE/$folder_name"
    mkdir -p "$output_dir"

    echo "──────────────────────────────────────────"
    echo "Folder: $folder_name"
    echo "  Input:  $dir"
    echo "  Output: $output_dir"
    echo "──────────────────────────────────────────"

    while IFS= read -r image_path; do

        image_name=$(basename "$image_path")
        image_stem="${image_name%.*}"
        output_path="$output_dir/${image_stem}_ablation.png"

        # Resume: skip already done
        if [ -f "$output_path" ]; then
            echo "  [SKIP] $image_name — already processed"
            processed=$((processed + 1))
            continue
        fi

        echo "  [$(( processed + 1 ))/$total_images] $image_name"

        python infer_ablation.py \
            --image      "$image_path" \
            --output     "$output_path" \
            --confidence "$CONFIDENCE"

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            processed=$((processed + 1))
            echo "  → saved: $output_path"
        else
            echo "  ERROR on $image_name (exit $exit_code)"
            failed=$((failed + 1))
        fi

        echo ""

    done < <(find "$dir" -maxdepth 1 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" \
           -o -iname "*.png" \
           -o -iname "*.tif" -o -iname "*.tiff" \) | sort)

done

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

echo ""
echo "========================================"
echo "ABLATION COMPLETE"
echo "  Processed : $processed"
echo "  Failed    : $failed"
echo "  End       : $(date)"
echo "========================================"

echo ""
echo "Results per folder:"
for dir in "${TEST_DIRS[@]}"; do
    folder_name=$(basename "$dir")
    output_dir="$OUTPUT_BASE/$folder_name"
    if [ -d "$output_dir" ]; then
        n=$(find "$output_dir" -name "*_ablation.png" | wc -l)
        echo "  $folder_name : $n ablation images"
    fi
done
echo ""
echo "All results in: $OUTPUT_BASE"