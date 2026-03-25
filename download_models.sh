#!/usr/bin/env bash
# Download HSMR model checkpoints and auxiliary regressors from HuggingFace.
# Run this from the 3rdParty/HSMR/ directory before using run_image_to_mesh.sh.
#
# NOTE: The SKEL body model requires free registration and must be downloaded
#       manually from https://skel.is.tue.mpg.de/download.php
#       Then run:  mv skel_models_v1.1 data_inputs/body_models/skel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "==> Downloading HSMR checkpoint..."
mkdir -p data_inputs/released_models
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/released_models/HSMR-ViTH-r1d1.tar.gz' \
    -O HSMR-ViTH-r1d1.tar.gz
tar -xzvf HSMR-ViTH-r1d1.tar.gz -C data_inputs/released_models/
rm HSMR-ViTH-r1d1.tar.gz
echo "==> Checkpoint saved to data_inputs/released_models/HSMR-ViTH-r1d1/"

echo "==> Downloading auxiliary regressors..."
mkdir -p data_inputs/body_models
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/SMPL_to_J19.pkl' \
    -O data_inputs/body_models/SMPL_to_J19.pkl
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/J_regressor_SKEL_mix_MALE.pkl' \
    -O data_inputs/body_models/J_regressor_SKEL_mix_MALE.pkl
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/J_regressor_SMPL_MALE.pkl' \
    -O data_inputs/body_models/J_regressor_SMPL_MALE.pkl
echo "==> Regressors saved to data_inputs/body_models/"

echo ""
echo "======================================================="
echo "  Manual step required: SKEL body model"
echo "======================================================="
echo "  1. Register (free) at https://skel.is.tue.mpg.de/download.php"
echo "  2. Download skel_models_v1.1.zip"
echo "  3. Run:"
echo "       unzip /path/to/skel_models_v1.1.zip"
echo "       mv skel_models_v1.1 data_inputs/body_models/skel"
echo "======================================================="
