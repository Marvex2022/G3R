# G3R reproduction notes: publication statement vs implementation choice

This file records the implementation decisions that matter for numerical reproduction.

| Item | Paper / supplementary material | This repository |
|---|---|---|
| Neural Gaussian feature width | C=46; 14 Gaussian parameters + 32 latent features | Exact |
| Gaussian attributes | position 3, scale 3, orientation 4, color 3, opacity 1 | Exact semantic layout |
| Initial position | downsampled LiDAR for driving / sampled mesh surface for BlendedMVS | `xyz` supplied by component NPZ |
| Initial scale | distance to third-nearest point | Exact |
| Initial rotation | identity | Exact |
| Initial opacity | 0.7 | Exact |
| Remaining initialization | random | Random RGB if no RGB supplied + random 32-D latent; optional RGB input is an engineering shortcut |
| SH | none | direct RGB, `sh_degree=None` |
| Learned Gaussian decoder | one linear layer, tanh, residual/skip to explicit Gaussian channels | `GaussianDecoder` |
| Gradient signal | image reconstruction gradient wrt neural Gaussians from source views | differentiable `gsplat` MSE gradient wrt all state features |
| Gradient normalization | per channel by max absolute value | Exact |
| Second-order gradient | not specified | stop-gradient (`create_graph=False`) |
| Static/dynamic optimizer | separate networks, same TorchSparse SparseResUNet architecture | separate `SparseG3RNet` instances |
| SparseResUNet topology | authors say TorchSparse SparseResUNet, no architecture tuning | follows public `SparseResUNet42` channel topology: stem 32; encoder 32/64/128/256; decoder 256/128/96/96 |
| Sparse voxel size | not reported | 0.20 m PandaSet, 0.10 m BlendedMVS defaults; configurable |
| Timestep | positional encoding concatenated at final encoder layer | sinusoidal embedding at bottleneck; width 32 (unreported) |
| Optimizer output | tanh | Exact |
| Scene update | S_(t+1)=S_t + gamma(t) G_theta(...) | Exact form |
| gamma(t) | cosine schedule from DDIM; exact scalar schedule unreported | normalized squared-cosine alpha-bar style schedule |
| Training supervision | source + novel/target views | Exact split semantics |
| Gradient input | source views only | Exact |
| Training network update | at each reconstruction step | Exact |
| Full driving steps | 24 | Exact |
| PandaSet turbo | 12 steps, 1.5M static/dynamic points | Exact reported values |
| Normal PandaSet inference | 3M static/dynamic points | Exact reported value |
| PandaSet training points | 800k static/dynamic total | Exact reported value |
| Driving train views | closest 10 source and 10 target | Exact count; “closest” defaults to frame distance |
| BlendedMVS train views | 25 source + 25 novel | Exact count; “closest” uses camera-center distance |
| Adam LR | 1e-4 | Exact |
| LPIPS weight | 0.01 | Exact |
| LPIPS backbone | not reported | `vgg` default, configurable |
| Gaussian flatness regularizer weight | 0.01 | Exact |
| Regularizer epsilon | not reported | 0.01 default, configurable |
| Rasterizer | Taichi 3D Gaussian Splatting implementation | `gsplat` for maintainability; not numerically identical |
| Long unroll backprop | not explicitly stated | truncated between iterations to keep memory practical |
| Dynamic actors | instance-wise foreground, transformed to scene coordinates per frame | actor NPZ in local coordinates + `actor_to_world` in each camera view |
| Distant/sky | sphere-image representation; 2D network described in supplement, wording ambiguous | spherical Gaussian grid + 2-res-block 2D CNN |
| PandaSet sphere | radius 2048 m; +30° to -15° latitude | helper reproduces these geometry defaults |
| Sky resolution | train 512x2048; inference 1024x4096 | manifest-driven; helper accepts both |
| Camera selection for simulation | all source images | inference uses all manifest `source` views |
| Multi-camera | front-camera training, then 100-iteration multi-camera fine-tuning reported | generic camera manifests are supported; no dataset-specific 100-iter wrapper is hard-coded |

## Why `gsplat` instead of the paper's Taichi renderer?

The publication used the open-source Taichi 3D Gaussian Splatting implementation. `gsplat` exposes differentiable means/scales/quaternions/opacities/colors with current PyTorch support and makes the learned-optimizer loop easier to run on recent CUDA stacks. This should preserve the G3R algorithmic experiment, but rasterization and gradient details can differ enough to affect exact metrics.

If exact metric reproduction is the goal, the most important next substitution is to replace `G3RRenderer.render()` with the paper's Taichi renderer while keeping the rest of the pipeline unchanged.

## Most important hidden variables to sweep

1. Sparse voxel size.
2. Gamma schedule amplitude/shape.
3. Random state initialization scale, especially latent feature standard deviation.
4. Flatness regularizer epsilon.
5. Whether to initialize RGB randomly or from scaffold colors.
6. Source/target view-selection definition for each dataset.
7. Exact image preprocessing/cropping and camera-intrinsic adjustment.
8. Truncated vs longer learned-optimizer unrolling.

These are more likely to explain a metric gap than small changes to the one-layer Gaussian decoder.
