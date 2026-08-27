"""The in-training eval draws follow the seed rule (spec §2).

Sample ``i``'s noise and class come from ``--seed + i`` alone, so the eval set is
the same at any ``--gpus``, on every tick, and any subset reproduces in isolation.
Before this the latents came from the ambient per-rank RNG.
"""

import torch

from diffit.metrics import _eval_draw


def test_eval_draw_is_a_pure_function_of_seed_and_index():
    z1, c1 = _eval_draw(42, 7, 8, 3)
    z2, c2 = _eval_draw(42, 7, 8, 3)
    assert torch.equal(z1, z2) and c1 == c2
    assert z1.shape == (4, 8, 8) and 0 <= c1 < 3


def test_eval_draw_differs_across_indices_and_seeds():
    z_a, _ = _eval_draw(42, 7, 8, 3)
    z_b, _ = _eval_draw(42, 8, 8, 3)
    z_c, _ = _eval_draw(43, 7, 8, 3)
    assert not torch.equal(z_a, z_b) and not torch.equal(z_a, z_c)
