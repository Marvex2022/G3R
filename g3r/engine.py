from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from tqdm import tqdm

from .data import SceneState, SceneTemplate, select_training_views
from .gaussian import GaussianDecoder, NeuralGaussianState
from .gradient_lift import SceneGradients, lift_source_images_to_gradients
from .losses import LPIPSLoss, flat_gaussian_regularizer, mse_loss
from .math_utils import cosine_ddim_gamma
from .networks import build_optimizer_network, SkyG3RNet
from .renderer import GSplatRenderer


@dataclass
class G3RModules:
    decoder: GaussianDecoder
    static_net: nn.Module
    dynamic_net: nn.Module
    sky_net: SkyG3RNet

    def modules(self) -> list[nn.Module]:
        return [self.decoder, self.static_net, self.dynamic_net, self.sky_net]

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def train(self, mode=True):
        for m in self.modules():
            m.train(mode)
        return self

    def state_dict(self) -> dict:
        return {
            "decoder": self.decoder.state_dict(),
            "static_net": self.static_net.state_dict(),
            "dynamic_net": self.dynamic_net.state_dict(),
            "sky_net": self.sky_net.state_dict(),
        }

    def load_state_dict(self, d: dict, strict: bool = True):
        self.decoder.load_state_dict(d["decoder"], strict=strict)
        self.static_net.load_state_dict(d["static_net"], strict=strict)
        self.dynamic_net.load_state_dict(d["dynamic_net"], strict=strict)
        if "sky_net" in d:
            self.sky_net.load_state_dict(d["sky_net"], strict=strict)


def build_modules(cfg: dict) -> G3RModules:
    net_cfg = cfg.get("network", {})
    return G3RModules(
        decoder=GaussianDecoder(),
        static_net=build_optimizer_network(net_cfg),
        dynamic_net=build_optimizer_network(net_cfg),
        sky_net=SkyG3RNet(
            time_dim=int(net_cfg.get("time_dim", 32)),
            hidden=int(net_cfg.get("sky_hidden", 96)),
        ),
    )


def _gamma(cfg: dict, step: int, total_steps: int) -> float:
    scfg = cfg.get("scheduler", {})
    kind = scfg.get("type", "cosine_ddim")
    if kind == "constant":
        return float(scfg.get("value", 0.3))
    if kind == "cosine":
        import math
        return float(scfg.get("floor", 0.0)) + (1.0 - float(scfg.get("floor", 0.0))) * 0.5 * (1.0 + math.cos(math.pi * step / max(total_steps - 1, 1)))
    if kind == "cosine_ddim":
        return cosine_ddim_gamma(
            step,
            total_steps,
            s=float(scfg.get("s", 0.008)),
            floor=float(scfg.get("floor", 0.0)),
        )
    raise ValueError(f"Unknown scheduler: {kind}")


def update_scene_state(
    state: SceneState,
    grads: SceneGradients,
    modules: G3RModules,
    step: int,
    total_steps: int,
    gamma: float,
    network_dtype: torch.dtype | None = None,
    amp_sky: bool = False,
) -> SceneState:
    # The lifted gradient is treated as a stop-gradient input; state values are also
    # detached between iterative optimizer steps (truncated unrolling), matching the
    # supplement's "update network at every reconstruction step" memory regime.
    bg_s = state.background.features.detach()
    bg_g = grads.background.detach()
    if network_dtype is None:
        bg_delta = modules.static_net(bg_s, bg_g, step, total_steps, positions=bg_s[:, :3])
    else:
        with torch.autocast(device_type="cuda", dtype=network_dtype):
            bg_delta = modules.static_net(
                bg_s.to(network_dtype),
                bg_g.to(network_dtype),
                step,
                total_steps,
                positions=bg_s[:, :3],
            )
        bg_delta = bg_delta.to(bg_s.dtype)
    bg_next = NeuralGaussianState(bg_s + gamma * bg_delta, "background")

    actors_next = {}
    if state.actors:
        actor_ids = list(state.actors)
        actor_states = [state.actors[aid].features.detach() for aid in actor_ids]
        actor_grads = [grads.actors[aid].detach() for aid in actor_ids]
        sizes = [len(x) for x in actor_states]
        s_all = torch.cat(actor_states, dim=0)
        g_all = torch.cat(actor_grads, dim=0)
        batch_indices = torch.cat(
            [torch.full((size,), i, device=s_all.device, dtype=torch.int32) for i, size in enumerate(sizes)]
        )
        if network_dtype is None:
            delta_all = modules.dynamic_net(
                s_all,
                g_all,
                step,
                total_steps,
                positions=s_all[:, :3],
                batch_indices=batch_indices,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=network_dtype):
                delta_all = modules.dynamic_net(
                    s_all.to(network_dtype),
                    g_all.to(network_dtype),
                    step,
                    total_steps,
                    positions=s_all[:, :3],
                    batch_indices=batch_indices,
                )
            delta_all = delta_all.to(s_all.dtype)
        for aid, s, delta in zip(actor_ids, actor_states, delta_all.split(sizes)):
            actors_next[aid] = NeuralGaussianState(s + gamma * delta, "actor", actor_id=aid)

    sky_next = None
    if state.sky is not None:
        s = state.sky.features.detach()
        g = grads.sky.detach()
        if state.sky.grid_shape is None:
            raise ValueError("Sky component needs grid_shape=(H,W) for the 2D optimizer")
        with torch.autocast(
            device_type="cuda",
            dtype=network_dtype or torch.float16,
            enabled=amp_sky and network_dtype is not None,
        ):
            delta = modules.sky_net(s, g, step, total_steps, state.sky.grid_shape)
        delta = delta.to(s.dtype)
        sky_next = NeuralGaussianState(s + gamma * delta, "sky", grid_shape=state.sky.grid_shape)

    return SceneState(bg_next, actors_next, sky_next)


def _supervision_backward(
    state_next: SceneState,
    scene: SceneTemplate,
    view_indices: list[int],
    modules: G3RModules,
    renderer: GSplatRenderer,
    lpips_loss: LPIPSLoss,
    device: str | torch.device,
    cfg: dict,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype | None,
    amp_lpips: bool,
    loss_weight: float = 1.0,
) -> dict[str, float]:
    if not view_indices:
        raise ValueError("No supervision views")
    lcfg = cfg.get("loss", {})
    lambda_lpips = float(lcfg.get("lambda_lpips", 0.01))
    lambda_reg = float(lcfg.get("lambda_reg", 0.01))
    reg_eps = float(lcfg.get("reg_epsilon", 0.01))
    n = len(view_indices)

    mse_acc = 0.0
    lpips_acc = 0.0
    for j, vidx in enumerate(view_indices):
        view = scene.views[vidx]
        target = view.load_image(device=device, dtype=state_next.background.features.dtype)
        out = renderer.render(state_next, modules.decoder, view)
        lmse = mse_loss(out.rgb, target)
        if lambda_lpips > 0:
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.float16,
                enabled=amp_lpips and amp_dtype is not None,
            ):
                llp = lpips_loss(out.rgb, target)
            llp = llp.float()
        else:
            llp = lmse.new_zeros(())
        loss = (lmse + lambda_lpips * llp) / n
        is_last = j == n - 1
        if is_last and lambda_reg > 0:
            loss = loss + lambda_reg * flat_gaussian_regularizer(state_next, modules.decoder, reg_eps)
        scaler.scale(loss * loss_weight).backward(retain_graph=not is_last)
        mse_acc += float(lmse.detach()) / n
        lpips_acc += float(llp.detach()) / n

    return {"mse": mse_acc, "lpips": lpips_acc}


def _optimizer_step(
    modules: G3RModules,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """Apply one synchronized update after all local scenes contributed gradients."""
    scaler.unscale_(optimizer)
    gradients_finite = _average_gradients(modules)
    optimizer_skipped = not gradients_finite
    if gradients_finite:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad(set_to_none=True)
        scaler.update(new_scale=max(float(scaler.get_scale()) * 0.5, 1.0))
    return {
        "grad_scale": float(scaler.get_scale()),
        "optimizer_skipped": float(optimizer_skipped),
    }


def _average_gradients(modules: G3RModules) -> bool:
    """Average all model gradients when launched through torchrun.

    Manual gradient averaging is used instead of wrapping the sparse networks in
    DDP because a scene can contain no dynamic actors. All ranks still enter the
    same collective with zero gradients for components unused by their scene.
    """
    params = [p for module in modules.modules() for p in module.parameters() if p.requires_grad]
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = _flatten_dense_tensors(grads)
    finite_device = "cpu" if dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo" else flat.device
    finite = torch.tensor([int(torch.isfinite(flat).all())], device=finite_device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return True
    # Gloo in this project is a compatibility fallback for machines whose NCCL
    # fabric service is unavailable; it communicates a CPU copy of the gradients.
    comm = flat.cpu() if dist.get_backend() == "gloo" and flat.is_cuda else flat
    dist.all_reduce(comm, op=dist.ReduceOp.SUM)
    comm.div_(dist.get_world_size())
    if comm.data_ptr() != flat.data_ptr():
        flat.copy_(comm)
    for param, synced in zip(params, _unflatten_dense_tensors(flat, grads)):
        if param.grad is None:
            param.grad = synced.clone()
        else:
            param.grad.copy_(synced)
    return True


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """Write a checkpoint atomically so an interrupted save cannot corrupt it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_hardlink(source: Path, destination: Path) -> None:
    """Keep a numbered snapshot without writing the large checkpoint twice."""
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.unlink(missing_ok=True)
    os.link(source, tmp)
    os.replace(tmp, destination)


_METRIC_FIELDS = [
    "iteration",
    "global_step",
    "mse",
    "psnr_db",
    "lpips",
    "active_steps",
    "scenes_per_iter",
    "learning_rate",
    "iteration_seconds",
    "grad_scale",
    "optimizer_skipped",
]


def _prepare_metrics_csv(path: Path, resume_iteration: int) -> None:
    """Create the CSV or discard rows newer than the resumed checkpoint."""
    kept: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["iteration"]) <= resume_iteration:
                        kept.append(row)
                except (KeyError, TypeError, ValueError):
                    continue
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(tmp, path)


def _append_metrics_csv(path: Path, row: dict[str, float | int]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METRIC_FIELDS)
        writer.writerow(row)


def train_g3r(
    cfg: dict,
    scenes: list[SceneTemplate],
    output_dir: str | Path,
    device: str | torch.device = "cuda",
    resume: str | Path | None = None,
) -> G3RModules:
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    output_dir = Path(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # Every rank must start from identical weights. Scene/view sampling below is
    # intentionally rank-specific so one optimizer update sees world_size scenes.
    seed = int(cfg.get("training", {}).get("seed", 0))
    torch.manual_seed(seed)
    modules = build_modules(cfg).to(device).train(True)

    params = []
    for m in modules.modules():
        params.extend(list(m.parameters()))
    tcfg = cfg.get("training", {})
    optimizer = torch.optim.Adam(params, lr=float(tcfg.get("lr", 1e-4)))
    renderer = GSplatRenderer(**cfg.get("renderer", {}))
    lpips_loss = LPIPSLoss(
        enabled=float(cfg.get("loss", {}).get("lambda_lpips", 0.01)) > 0,
        max_side=cfg.get("loss", {}).get("lpips_max_side"),
        net=str(cfg.get("loss", {}).get("lpips_net", "vgg")),
    ).to(device)

    outer_iters = int(tcfg.get("scene_iterations", 1000))
    full_steps = int(tcfg.get("steps", 24))
    warmup_outer = int(tcfg.get("warmup_scene_iterations", 100))
    point_limit = int(tcfg.get("point_limit", 800_000))
    n_source = int(tcfg.get("source_views", 10))
    n_target = int(tcfg.get("target_views", 10))
    selection_mode = tcfg.get("view_selection", "frame")
    checkpoint_every = int(tcfg.get("checkpoint_every", 25))
    latest_every = int(tcfg.get("latest_every", 1))
    scenes_per_rank = int(tcfg.get("scenes_per_rank", 1))
    mixed_precision = bool(tcfg.get("mixed_precision", False))
    amp_dtype_name = str(tcfg.get("amp_dtype", "float16")).lower()
    if amp_dtype_name not in {"float16", "fp16"}:
        raise ValueError(
            f"Unsupported training.amp_dtype={amp_dtype_name!r}; this TorchSparse build only supports float16"
        )
    amp_dtype = torch.float16 if mixed_precision else None
    amp_sky = mixed_precision and bool(tcfg.get("amp_sky", True))
    amp_lpips = mixed_precision and bool(tcfg.get("amp_lpips", True))
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    if checkpoint_every <= 0 or latest_every <= 0:
        raise ValueError("checkpoint_every and latest_every must be positive")
    if scenes_per_rank <= 0:
        raise ValueError("training.scenes_per_rank must be positive")
    scenes_per_iter = world_size * scenes_per_rank

    start_outer = 0
    global_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        modules.load_state_dict(ckpt["modules"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if mixed_precision and "grad_scaler" in ckpt:
            scaler.load_state_dict(ckpt["grad_scaler"])
        start_outer = int(ckpt.get("outer_iteration", 0))
        global_step = int(ckpt.get("global_step", 0))
        if start_outer >= outer_iters:
            raise ValueError(
                f"Checkpoint is already at iteration {start_outer}, but config only requests {outer_iters}"
            )
        if rank == 0:
            print(f"Resuming {resume} from scene iteration {start_outer}, global step {global_step}")

    metrics_path = output_dir / "metrics.csv"
    tb_writer = None
    if rank == 0:
        _prepare_metrics_csv(metrics_path, start_outer)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("TensorBoard logging requires `pip install tensorboard`.") from exc
        tb_writer = SummaryWriter(
            log_dir=str(output_dir / "tensorboard"),
            purge_step=start_outer + 1 if start_outer > 0 else None,
        )

    pbar = tqdm(
        range(start_outer, outer_iters),
        desc="G3R scene iterations",
        initial=start_outer,
        total=outer_iters,
        disable=rank != 0,
    )
    for outer in pbar:
        iteration_started = time.perf_counter()
        # Keep the same four per-iteration sample seeds when migrating from
        # 4 GPUs x 1 scene/rank to 2 GPUs x 2 scenes/rank.
        local_batches = []
        for local_index in range(scenes_per_rank):
            sample_index = rank * scenes_per_rank + local_index
            sample_seed = seed + outer * scenes_per_iter + sample_index
            rng = random.Random(sample_seed)
            scene = rng.choice(scenes)
            state = scene.instantiate(
                device=device,
                point_limit=point_limit,
                seed=sample_seed,
            )
            src_idx, tgt_idx = select_training_views(
                scene, n_source, n_target, selection_mode, rng
            )
            local_batches.append([scene, state, src_idx, tgt_idx])
        if warmup_outer > 0:
            frac = min(1.0, (outer + 1) / warmup_outer)
            active_steps = max(1, int(round(1 + frac * (full_steps - 1))))
        else:
            active_steps = full_steps

        last_stats = {}
        for step in range(active_steps):
            optimizer.zero_grad(set_to_none=True)
            mse_acc = 0.0
            lpips_acc = 0.0
            gamma = _gamma(cfg, step, full_steps)
            for batch in local_batches:
                scene, state, src_idx, tgt_idx = batch
                # Each scene is processed sequentially, so activation memory is
                # close to the old one-scene-per-rank run. Gradients accumulate.
                leaf = state.detach().requires_grad_(True)
                grads = lift_source_images_to_gradients(
                    leaf, scene, src_idx, modules.decoder, renderer, device,
                    normalize=bool(tcfg.get("normalize_gradients", True)),
                )
                state_next = update_scene_state(
                    leaf,
                    grads,
                    modules,
                    step,
                    full_steps,
                    gamma,
                    network_dtype=amp_dtype,
                    amp_sky=amp_sky,
                )
                scene_stats = _supervision_backward(
                    state_next,
                    scene,
                    src_idx + tgt_idx,
                    modules,
                    renderer,
                    lpips_loss,
                    device,
                    cfg,
                    scaler,
                    amp_dtype,
                    amp_lpips,
                    loss_weight=1.0 / scenes_per_rank,
                )
                mse_acc += scene_stats["mse"] / scenes_per_rank
                lpips_acc += scene_stats["lpips"] / scenes_per_rank
                batch[1] = state_next.detach()
            last_stats = {"mse": mse_acc, "lpips": lpips_acc}
            last_stats.update(_optimizer_step(modules, optimizer, scaler))
            global_step += 1

        if distributed:
            stats_device = "cpu" if dist.get_backend() == "gloo" else device
            stats = torch.tensor(
                [last_stats.get("mse", 0.0), last_stats.get("lpips", 0.0)],
                device=stats_device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats.div_(world_size)
            # Preserve GradScaler diagnostics from the synchronized optimizer
            # step while replacing only the metrics that need rank averaging.
            last_stats["mse"] = float(stats[0])
            last_stats["lpips"] = float(stats[1])
        completed_outer = outer + 1
        mse = float(last_stats.get("mse", 0.0))
        lpips_value = float(last_stats.get("lpips", 0.0))
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        iteration_seconds = time.perf_counter() - iteration_started
        if rank == 0:
            pbar.set_postfix(
                scenes_per_iter=scenes_per_iter,
                T=active_steps,
                mse=f"{mse:.4f}",
                psnr=f"{psnr:.2f}",
            )
            metric_row = {
                "iteration": completed_outer,
                "global_step": global_step,
                "mse": mse,
                "psnr_db": psnr,
                "lpips": lpips_value,
                "active_steps": active_steps,
                "scenes_per_iter": scenes_per_iter,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "iteration_seconds": iteration_seconds,
                "grad_scale": float(last_stats.get("grad_scale", scaler.get_scale())),
                "optimizer_skipped": int(last_stats.get("optimizer_skipped", 0.0)),
            }
            _append_metrics_csv(metrics_path, metric_row)
            assert tb_writer is not None
            tb_writer.add_scalar("train/mse", mse, completed_outer)
            tb_writer.add_scalar("train/psnr_db", psnr, completed_outer)
            tb_writer.add_scalar("train/lpips", lpips_value, completed_outer)
            tb_writer.add_scalar("train/active_steps", active_steps, completed_outer)
            tb_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], completed_outer)
            tb_writer.add_scalar("train/grad_scale", metric_row["grad_scale"], completed_outer)
            tb_writer.add_scalar("train/optimizer_skipped", metric_row["optimizer_skipped"], completed_outer)
            tb_writer.add_scalar("time/iteration_seconds", iteration_seconds, completed_outer)
            tb_writer.flush()
        keep_numbered = completed_outer % checkpoint_every == 0 or completed_outer == outer_iters
        update_latest = completed_outer % latest_every == 0 or keep_numbered
        if rank == 0 and update_latest:
            ckpt = {
                "modules": modules.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": scaler.state_dict(),
                "outer_iteration": completed_outer,
                "global_step": global_step,
                "world_size": world_size,
                "scenes_per_rank": scenes_per_rank,
                "effective_scene_batch": scenes_per_iter,
                "config": cfg,
            }
            latest_path = output_dir / "latest.pt"
            _atomic_torch_save(ckpt, latest_path)
            if keep_numbered:
                _atomic_hardlink(latest_path, output_dir / f"checkpoint_{completed_outer:04d}.pt")
    if tb_writer is not None:
        tb_writer.close()
    return modules


@torch.no_grad()
def _network_update_inference(state: SceneState, grads: SceneGradients, modules: G3RModules, step: int, total_steps: int, gamma: float) -> SceneState:
    return update_scene_state(state, grads, modules, step, total_steps, gamma).detach()


def reconstruct_scene(
    cfg: dict,
    scene: SceneTemplate,
    modules: G3RModules,
    device: str | torch.device = "cuda",
    turbo: bool = False,
) -> SceneState:
    modules.to(device).train(False)
    icfg = cfg.get("inference", {})
    full_steps = int(icfg.get("steps_turbo", 12) if turbo else icfg.get("steps", 24))
    point_limit = int(icfg.get("point_limit_turbo", 1_500_000) if turbo else icfg.get("point_limit", 3_000_000))
    state = scene.instantiate(device=device, point_limit=point_limit, seed=int(icfg.get("seed", 0)))
    renderer = GSplatRenderer(**cfg.get("renderer", {}))
    src_idx = scene.source_indices()

    for step in tqdm(range(full_steps), desc=f"Reconstruct {scene.name}"):
        # Gradient lifting itself needs autograd wrt S even in inference.
        with torch.enable_grad():
            leaf = state.detach().requires_grad_(True)
            grads = lift_source_images_to_gradients(
                leaf, scene, src_idx, modules.decoder, renderer, device,
                normalize=bool(icfg.get("normalize_gradients", True)),
            )
        gamma = _gamma(cfg, step, full_steps)
        with torch.no_grad():
            state = _network_update_inference(leaf, grads, modules, step, full_steps, gamma)
    return state
