# HSMR Docker

## Build

```bash
cd 3rdParty/HSMR
docker build -t hsmr:latest .
```

## Run (GPU)

```bash
docker run --gpus all --rm -it \
    -v "$(pwd)/data_inputs:/workspace/HSMR/data_inputs" \
    -v "$(pwd)/data_outputs:/workspace/HSMR/data_outputs" \
    hsmr:latest bash
```

- `data_inputs/` is mounted so model weights and body models persist without being baked into the image.
- `data_outputs/` is mounted so outputs written inside the container are immediately visible on the host.

## Demo inside container

```bash
python exp/run_demo.py --input_path "data_inputs/demo/example_imgs"
```

## image_to_mesh.py with shared output

Run the script non-interactively and write results directly to your host:

```bash
docker run --gpus all --rm \
    -v "$(pwd)/data_inputs:/workspace/HSMR/data_inputs" \
    -v "$(pwd)/data_outputs:/workspace/HSMR/data_outputs" \
    hsmr:latest \
    python exp/image_to_mesh.py \
        -i data_inputs/demo/example_imgs/ \
        -o data_outputs/demos \
        --save_mesh --save_json
```

Results appear under `3rdParty/HSMR/data_outputs/demos/` on your host as soon as they are written.

You can also bind-mount any arbitrary host directory as the output:

```bash
docker run --gpus all --rm \
    -v "$(pwd)/data_inputs:/workspace/HSMR/data_inputs" \
    -v "/path/on/host:/outputs" \
    hsmr:latest \
    python exp/image_to_mesh.py -i data_inputs/demo/example_imgs/ -o /outputs --save_mesh
```

## Notes

- Base image: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
- PyTorch: `2.3.1+cu121`
- Python: `3.10`
- `PYOPENGL_PLATFORM=egl` is set for headless rendering via PyRender.
- The SKEL submodule is cloned at build time from
  <https://github.com/MarilynKeller/SKEL> if `thirdparty/SKEL/` is empty.
- Body models (SKEL, SMPL) and checkpoints must be placed under `data_inputs/`
  manually (they require registration/license agreements – see
  [docs/SETUP.md](docs/SETUP.md)).
