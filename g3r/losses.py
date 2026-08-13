from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .data import SceneState
from .gaussian import GaussianDecoder


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


class LPIPSLoss(nn.Module):
    def __init__(self, enabled: bool = True, max_side: int | None = None, net: str = "vgg"):
        super().__init__()
        self.enabled = enabled
        self.max_side = max_side
        self.backbone = net
        if enabled:
            import lpips
            self.net = lpips.LPIPS(net=net)
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
        else:
            self.net = None

    def forward(self, pred_hwc: torch.Tensor, target_hwc: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return pred_hwc.new_zeros(())
        pred = pred_hwc.permute(2, 0, 1)[None] * 2 - 1
        target = target_hwc.permute(2, 0, 1)[None] * 2 - 1
        if self.max_side is not None:
            h, w = pred.shape[-2:]
            scale = min(1.0, self.max_side / max(h, w))
            if scale < 1.0:
                nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
                pred = F.interpolate(pred, (nh, nw), mode="bilinear", align_corners=False)
                target = F.interpolate(target, (nh, nw), mode="bilinear", align_corners=False)
        return self.net(pred, target).mean()


def flat_gaussian_regularizer(state: SceneState, decoder: GaussianDecoder, epsilon: float) -> torch.Tensor:
    """Eq. (7): sum_i max(0, d_i^min - epsilon). Averaged for point-count invariance."""
    vals = []
    for comp in state.components():
        p = decoder(comp.features)
        dmin = p.scales.amin(dim=-1)
        vals.append(F.relu(dmin - epsilon).mean())
    if not vals:
        return torch.tensor(0.0)
    return torch.stack(vals).mean()
