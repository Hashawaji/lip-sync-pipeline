#!/bin/bash
# Process a single directory with MFA phoneme extraction and transformation
# Expects text.txt and audio.mp3 in the base directory

# Check if base directory is provided as argument
if [ $# -eq 0 ]; then
    echo "Error: No base directory provided"
    echo "Usage: ./process_single_directory.sh <base_directory>"
    echo "Example: ./process_single_directory.sh /home/ist/Desktop/synth_audios/my_audio"
    exit 1
fi

# Base directory from command line argument
BASE_DIR="$1"

# Fixed file names
TEXT_FILE="${BASE_DIR}/text.txt"
AUDIO_FILE="${BASE_DIR}/audio.mp3"
OUTPUT_DIR="${BASE_DIR}"

# Path to the MFA phoneme CLI script
MFA_SCRIPT="/home/ist/Desktop/lip-sync-pipeline/mfa_phoneme_cli_v3.py"
# MFA_SCRIPT="/home/ist/Desktop/lip-sync-pipeline/mfa_phoneme_fast_cli.py"

# Path to the phoneme JSON transformation script
TRANSFORM_SCRIPT="/home/ist/Desktop/lip-sync-pipeline/phoneme-json-transformation.py"

# Conda environment name
CONDA_ENV="mfa-env"
# CONDA_ENV="mfa-dev"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

echo "✓ Successfully activated $CONDA_ENV"
echo ""

# Validate base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    conda deactivate
    exit 1
fi

# Validate text file exists
if [ ! -f "$TEXT_FILE" ]; then
    echo "Error: Text file not found: $TEXT_FILE"
    echo "Expected: text.txt in $BASE_DIR"
    conda deactivate
    exit 1
fi

# Validate audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    echo "Expected: audio.mp3 in $BASE_DIR"
    conda deactivate
    exit 1
fi

# Validate MFA script exists
if [ ! -f "$MFA_SCRIPT" ]; then
    echo "Error: MFA script does not exist: $MFA_SCRIPT"
    conda deactivate
    exit 1
fi

# Validate transformation script exists
if [ ! -f "$TRANSFORM_SCRIPT" ]; then
    echo "Error: Transformation script does not exist: $TRANSFORM_SCRIPT"
    conda deactivate
    exit 1
fi

echo "=========================================="
echo "MFA Phoneme Extraction - Single Directory"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo "Audio file: $AUDIO_FILE"
echo "Text file: $TEXT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Run MFA phoneme extraction
echo "----------------------------------------"
echo "Step 1: Running MFA phoneme extraction"
echo "----------------------------------------"
python3 "$MFA_SCRIPT" \
    --single_file "$AUDIO_FILE" "$TEXT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_jobs 1

# Check if MFA succeeded
if [ $? -ne 0 ]; then
    echo "✗ Error: MFA phoneme extraction failed"
    conda deactivate
    exit 1
fi

echo "✓ Successfully completed MFA phoneme extraction"
echo ""

# Step 2: Run phoneme JSON transformation
PHONEME_JSON="${OUTPUT_DIR}/complete_phoneme_alignments.json"

if [ ! -f "$PHONEME_JSON" ]; then
    echo "✗ Error: complete_phoneme_alignments.json not found at $PHONEME_JSON"
    conda deactivate
    exit 1
fi

echo "----------------------------------------"
echo "Step 2: Running phoneme JSON transformation"
echo "----------------------------------------"
python3 "$TRANSFORM_SCRIPT" \
    --input_json "$PHONEME_JSON" \
    --transform-type fixed-length

# Check if transformation succeeded
if [ $? -ne 0 ]; then
    echo "✗ Error: Phoneme JSON transformation failed"
    conda deactivate
    exit 1
fi

echo "✓ Successfully completed phoneme JSON transformation"
echo ""

# Step 3: Clean corpus folder
CORPUS_DIR="/home/ist/Desktop/video-retalking/mfa_workspace/corpus"
if [ -d "$CORPUS_DIR" ]; then
    echo "----------------------------------------"
    echo "Step 3: Cleaning corpus folder"
    echo "----------------------------------------"
    rm -rf "$CORPUS_DIR"/*
    if [ $? -eq 0 ]; then
        echo "✓ Successfully cleaned corpus folder"
    else
        echo "⚠ Warning: Failed to clean corpus folder"
    fi
else
    echo "Note: Corpus folder not found at $CORPUS_DIR (skipping cleanup)"
fi

echo ""
echo "=========================================="
echo "Processing Complete!"
echo "=========================================="
echo ""
echo "Output files saved in: $OUTPUT_DIR"
echo "  - complete_phoneme_alignments.json"
echo "  - complete_phoneme_alignments_w_reps_fixed_len.json"
echo ""

# Deactivate conda environment
echo "Deactivating conda environment..."
conda deactivate
echo "✓ Environment deactivated"
echo ""