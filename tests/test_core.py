from pathlib import Path
import sys

# Allow `python tests/test_core.py` from the repository root without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from g3r.gaussian import initialize_neural_gaussians, GaussianDecoder, STATE_DIM
from g3r.math_utils import cosine_ddim_gamma
from g3r.networks import PointMLPOptimizer


def test_state_and_decoder():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    s = initialize_neural_gaussians(xyz, device="cpu")
    assert s.features.shape == (4, STATE_DIM)
    p = GaussianDecoder()(s.features)
    assert p.means.shape == (4, 3)
    assert torch.all(p.scales > 0)
    assert torch.all((p.colors >= 0) & (p.colors <= 1))
    assert torch.all((p.opacities >= 0) & (p.opacities <= 1))


def test_point_optimizer_shape():
    x = torch.randn(8, STATE_DIM)
    g = torch.randn_like(x)
    net = PointMLPOptimizer(hidden=64, time_dim=16)
    y = net(x, g, step=0, total_steps=24)
    assert y.shape == x.shape
    assert y.abs().max() <= 1.0 + 1e-6


def test_point_optimizer_accepts_actor_batches():
    x = torch.randn(7, STATE_DIM)
    g = torch.randn_like(x)
    batches = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    net = PointMLPOptimizer(hidden=32, time_dim=16)
    y = net(x, g, step=0, total_steps=24, batch_indices=batches)
    assert y.shape == x.shape


def test_gamma_decays():
    vals = [cosine_ddim_gamma(i, 24) for i in range(24)]
    assert vals[0] > vals[-1]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


if __name__ == "__main__":
    test_state_and_decoder()
    test_point_optimizer_shape()
    test_point_optimizer_accepts_actor_batches()
    test_gamma_decays()
    print("core tests: OK")
