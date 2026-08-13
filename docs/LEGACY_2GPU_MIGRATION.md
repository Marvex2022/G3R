# Legacy four-scene batch on two GPUs

This worktree is pinned to the pre-corrected G3R implementation (`20f26b3`).
It resumes the original PandaSet run with four distributed workers while keeping
the effective scene batch at four:

```text
old: 4 GPUs x 1 process/GPU x 1 scene/rank = 4 scenes/update
new: 2 GPUs x 2 processes/GPU x 1 scene/rank = 4 scenes/update
```

Each worker reconstructs one scene. Two independent CUDA processes share each
physical GPU, so four scenes can make progress concurrently before one global
gradient average and optimizer update. Independent processes keep PyTorch
autograd, TorchSparse, and gsplat state isolated; Python threads are not used.

The global sample seed is based on `outer_iteration * effective_scene_batch +
global_sample_index`, so the two-GPU run consumes the same four sample seeds per
outer iteration as the old four-GPU run.

Resume on physical GPUs 2 and 5:

```bash
cd /home/user/longjun/g3r_repro_code_legacy_2gpu
CUDA_VISIBLE_DEVICES=2,5 \
G3R_DIST_BACKEND=gloo \
G3R_PROCESSES_PER_GPU=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc-per-node=4 \
  train.py \
  --config configs/pandaset_legacy_2gpu.yaml \
  --device cuda \
  --resume /home/user/longjun/g3r_repro_code/runs/g3r_pandaset_4gpu/latest.pt
```

The run continues to write checkpoints and metrics to the original output
directory. `checkpoint_pre_2gpu_0413.pt` is the immutable migration snapshot.
