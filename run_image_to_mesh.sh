#!/usr/bin/env bash
# Run image_to_mesh.py inside the HSMR Docker container.
# Outputs are written to data_outputs/demos/ on the host.
#
# Usage:
#   ./run_image_to_mesh.sh -i <input_path> [options]
#
# All arguments are forwarded directly to image_to_mesh.py.
# Examples:
#   ./run_image_to_mesh.sh -i data_inputs/demo/example_imgs/ --save_mesh --save_json
#   ./run_image_to_mesh.sh -i data_inputs/demo/example_imgs/ -o data_outputs/my_run --save_mesh
#   ./run_image_to_mesh.sh -i my_video.mp4 --save_json --save_viz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="hsmr:latest"

mkdir -p "${SCRIPT_DIR}/data_inputs/model_cache"

docker run --gpus all --rm \
    -v "${SCRIPT_DIR}/data_inputs:/workspace/HSMR/data_inputs" \
    -v "${SCRIPT_DIR}/data_outputs:/workspace/HSMR/data_outputs" \
    -v "${SCRIPT_DIR}/exp:/workspace/HSMR/exp" \
    -v "${SCRIPT_DIR}/lib:/workspace/HSMR/lib" \
    -v "${SCRIPT_DIR}/data_inputs/model_cache:/root/.torch/iopath_cache" \
    "${IMAGE}" \
    python exp/image_to_mesh.py "$@"
