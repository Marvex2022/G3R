# G3R: Gradient Guided Generalizable Reconstruction — engineering reproduction

This repository is a **paper-aligned, from-scratch reimplementation** of **G3R: Gradient Guided Generalizable Reconstruction (ECCV 2024)**. It contains both training and reconstruction/inference code.

Paper: https://www.datocms-assets.com/163962/1755337317-paper.pdf  
Project: https://waabi.ai/research/g3r

> This is **not the authors' official code**. The paper and supplementary material leave several implementation details unspecified. Those choices are isolated in YAML configs and documented in `docs/REPRO_NOTES.md` rather than silently presented as paper facts.

## What is implemented

The core iteration is

```text
source images -> differentiable render -> d L_source / d S_t
              -> normalize gradient channel-wise
              -> G3R-Net(S_t, grad_t, timestep)
              -> S_{t+1} = S_t + gamma(t) * delta_t
```

Training uses **source-view gradients as the optimizer input**, while the learned optimizer is supervised by rendering **both source and held-out target/novel views**. The reconstruction state has `C=46` channels per neural Gaussian: 14 explicit Gaussian attributes plus a 32-D latent feature.

Implemented components:

- `g3r/gaussian.py`: neural Gaussian state, 46->14 one-layer/tanh residual decoder, paper initialization rules.
- `g3r/networks.py`: separate static and dynamic **TorchSparse SparseResUNet42-style** learned optimizers; timestep embedding at the bottleneck; 2D two-residual-block distant/sky optimizer.
- `g3r/gradient_lift.py`: source-image photometric-gradient lifting and per-channel max-absolute normalization.
- `g3r/renderer.py`: differentiable Gaussian rasterization using `gsplat`.
- `g3r/engine.py`: 24-step training/reconstruction loop, step-wise network optimization, warmup, full/turbo inference.
- `train.py`: learned-optimizer training entry point.
- `infer.py`: scene reconstruction/inference entry point and optional novel-view rendering.
- Static background, dynamic object-local Gaussians + per-frame actor poses, and optional spherical distant/sky component.

## Paper-aligned defaults

The included PandaSet configuration uses the values reported in the paper/supplement where available:

- neural Gaussian state: **46 channels = 14 Gaussian + 32 latent**
- sparse optimizer: TorchSparse `SparseResUNet` family
- normal reconstruction: **24 iterations**
- PandaSet turbo: **12 iterations**
- training initialization budget: **800k static+dynamic points**
- normal PandaSet inference budget: **3M static+dynamic points**
- turbo PandaSet inference budget: **1.5M points**
- closest **10 source + 10 target frames** for driving-scene training
- Adam learning rate **1e-4**
- `lambda_lpips = 0.01`, `lambda_reg = 0.01`
- scale initialized from the third nearest scaffold point, identity quaternion, opacity 0.7; remaining attributes/features randomly initialized by default
- no spherical harmonics: direct RGB color per Gaussian

The BlendedMVS config switches to the reported 1.5M/3.5M point budgets, 25 source + 25 target images, and 24-step turbo reconstruction.

## Important fidelity boundaries

Several details are not numerically specified by the publication. This repo therefore exposes them as engineering choices:

1. **Exact cosine update schedule `gamma(t)`** — the paper says it follows a DDIM cosine schedule, but does not publish the exact scalar expression/amplitude. `cosine_ddim_gamma()` uses a normalized squared-cosine schedule.
2. **Sparse voxel size** — not reported. PandaSet defaults to 0.20 m and BlendedMVS to 0.10 m; these are meant to be tuned.
3. **LPIPS backbone** — the loss weight is reported but the backbone is not; the config defaults to VGG and exposes `lpips_net`.
4. **Internal Gaussian parameterization** — the paper defines the 14 attributes but not the renderer's unconstrained storage transforms. This implementation uses xyz, log-scale, raw `wxyz` quaternion, RGB logits, and opacity logit.
5. **Gradient differentiation order** — source gradients are treated as stop-gradient inputs (`create_graph=False`). This avoids an unsupported/huge second-order path through Gaussian rasterization.
6. **Unrolling** — the scene state is detached between reconstruction iterations. The network is optimized at each iteration, but gradients are not backpropagated through all previous iterations.
7. **2D CNN wording in the supplement** — the supplement first says static and dynamic components use SparseResUNet, then refers to a 2D CNN as a “background reconstruction network.” Because the method separately represents distant/sky content on a spherical image, this repo uses that 2D CNN for the distant/sky component.
8. `gsplat` is used instead of the paper's Taichi 3DGS rasterizer. The learned-optimizer algorithm is unchanged, but exact rasterization numerics may differ.

For a line-by-line mapping, see `docs/REPRO_NOTES.md`.

## Installation

A modern CUDA Linux environment is recommended.

```bash
conda create -n g3r-repro python=3.10 -y
conda activate g3r-repro

# Install the PyTorch build matching your CUDA first.
# Example only; choose the command from pytorch.org for your machine.
pip install torch torchvision

pip install -r requirements.txt
```

### TorchSparse

The paper uses TorchSparse's SparseResUNet. TorchSparse wheels are CUDA/PyTorch-version sensitive, so install a build matching your environment rather than pinning a universal wheel here:

```bash
git clone https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
pip install -v .
```

For a quick CPU-side architecture/unit smoke test you can use `configs/debug_small.yaml`; real Gaussian rendering/training still requires CUDA.

## Data format

Each scene is described by one JSON manifest. The component NPZ format is intentionally minimal:

```text
background.npz
  xyz:   float32 [N,3]       required
  scale: float32 [N]         optional; otherwise 3-NN is computed at load
  rgb:   float32 [N,3]       optional engineering shortcut

actor_car_17.npz
  xyz:   float32 [M,3]       in the actor's local coordinate system
  scale: float32 [M]         optional
```

For the most paper-faithful initialization, omit `rgb`; color and latent attributes are randomly initialized. If an NPZ contains RGB, this implementation optionally uses it as an initialization shortcut.

A manifest has the following structure:

```json
{
  "name": "scene_0001",
  "components": {
    "background": "background.npz",
    "actors": {
      "car_17": "actor_car_17.npz"
    },
    "sky": {
      "path": "sky.npz",
      "height": 512,
      "width": 2048
    }
  },
  "views": [
    {
      "image": "images/000000.jpg",
      "K": [[1000,0,960],[0,1000,540],[0,0,1]],
      "w2c": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "width": 1920,
      "height": 1080,
      "frame_id": 0,
      "split": "source",
      "actor_to_world": {
        "car_17": [[1,0,0,3.2],[0,1,0,1.0],[0,0,1,0],[0,0,0,1]]
      }
    }
  ]
}
```

`w2c` is a **world-to-camera** 4x4 matrix. Actor transforms are **actor-local to world**.

A training scene-list JSON is simply:

```json
{
  "scenes": [
    "scene_0001/scene.json",
    "scene_0002/scene.json"
  ]
}
```

See `examples/scene_dynamic.example.json` and `examples/train_scenes.example.json`.

## Preparing a scaffold

For the local PandaSet checkout configured in `configs/pandaset_paper.yaml`, convert
the raw sequences into manifests and static LiDAR scaffolds with:

```bash
python scripts/prepare_pandaset.py --config configs/pandaset_paper.yaml
```

For a quick plumbing check, prepare only one sequence with fewer points:

```bash
python scripts/prepare_pandaset.py --config configs/pandaset_paper.yaml \
  --sequences 003 --max-points 50000
```

By default the converter uses PandaSet's non-stationary cuboids to remove moving
points from the background, aggregate actor-local NPZ scaffolds by UUID, and add
per-view `actor_to_world` transforms. Use `--static-only` to build the earlier
static-background baseline instead.

If you already have a point scaffold from LiDAR, a mesh sample, SfM, or another reconstructor:

```bash
python scripts/prepare_points.py \
  --input raw_background.npz \
  --output background.npz
```

This precomputes the third-nearest-neighbor scale once instead of doing the KD-tree query at every run.

For a static scene, a basic camera manifest can be generated with:

```bash
python scripts/make_scene_manifest.py \
  --name scene_0001 \
  --points background.npz \
  --cameras cameras.json \
  --output scene.json
```

For driving scenes, build actor-local NPZs and fill `actor_to_world` for each view. If an actor is absent in a view, simply omit its transform in that view; the renderer skips it.

## Training

PandaSet-style training:

```bash
python train.py \
  --config configs/pandaset_paper.yaml \
  --device cuda
```

`--scenes` and `--output` may still be supplied to override `paths.scenes` and
`paths.output` from the YAML config.

Checkpoints are written as `checkpoint_XXXX.pt` and `latest.pt`.

The outer loop samples one scene, initializes its neural Gaussians from scratch, selects source/target observations, runs the iterative reconstruction optimizer, and updates the shared learned optimizer **after every reconstruction step**.

For initial plumbing/debugging, change the config to `configs/debug_small.yaml` and use a very small scaffold/image resolution. That backend is deliberately a point-wise MLP and is **not** a paper-quality substitute for SparseResUNet.

## Inference / reconstruction

Full reconstruction:

```bash
python infer.py \
  --config configs/pandaset_paper.yaml \
  --checkpoint runs/g3r_pandaset/latest.pt \
  --scene /path/to/scene.json \
  --output outputs/scene_0001 \
  --device cuda \
  --render-targets
```

PandaSet turbo mode:

```bash
python infer.py \
  --config configs/pandaset_paper.yaml \
  --checkpoint runs/g3r_pandaset/latest.pt \
  --scene /path/to/scene.json \
  --output outputs/scene_0001_turbo \
  --device cuda \
  --turbo \
  --render-targets
```

Outputs:

```text
outputs/scene_0001/
  gaussians/
    background.npz
    actor_<id>.npz
    sky.npz
  renders/
    000001.png
    ...
```

Each output Gaussian NPZ contains both `neural_features` and decoded explicit `xyz/scale/quat_wxyz/rgb/opacity`.

## Sky / distant region

The supplementary material describes a cropped spherical distant representation centered around the ego pose. `g3r.gaussian.make_sky_sphere()` implements the reported geometry (default radius 2048 m, latitude +30° to -15°). A helper is included:

```bash
python scripts/make_sky.py \
  --center 0 0 0 \
  --height 512 --width 2048 \
  --output sky.npz
```

The paper reports 512x2048 for training and 1024x4096 at normal inference for PandaSet. Be aware that these grids are large.

## Validation / smoke tests

```bash
python -m compileall g3r train.py infer.py scripts
python tests/test_core.py
python scripts/check_scene.py /path/to/scene.json
```

`tests/test_core.py` intentionally avoids requiring TorchSparse/gsplat so the core state math can be checked before setting up CUDA extensions.

## Recommended reproduction order

For practical debugging, do not start with 3M points. First make one static scene work with 5k–50k points and 256–512 px images; verify that source-image gradient lifting reduces held-out-view loss; then enable TorchSparse; then add actor-local dynamic Gaussians; only after the loop is stable should you scale to the paper point budgets and sky grid.
