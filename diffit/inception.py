"""Shared InceptionV3 feature extractor for FID / IS / sFID metrics.

Both the inline training-time metrics (``diffit/metrics.py``) and the standalone
evaluator (``evaluator.py``) need identical Inception pool/spatial/logit
features. This module is the single source of truth for that extraction so the
two paths can never drift apart.
"""

import torch
import torch.nn.functional as F

from diffit.constants import INCEPTION_MEAN, INCEPTION_STD


def load_inception_model(device):
    """Load InceptionV3 (torchvision DEFAULT weights) in eval mode on ``device``."""
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model


class InceptionFeatureExtractor(torch.nn.Module):
    """Extract pool (2048-d), spatial and logit features from InceptionV3.

    Input ``x`` is an NCHW float tensor in ``[0, 1]``; it is resized to 299x299
    and normalized with ImageNet statistics before being run through the
    network's internal blocks.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor(INCEPTION_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(INCEPTION_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        m = self.model
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        spatial = x  # spatial features for sFID
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)

        pool = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # (N, 2048)
        logits = m.fc(pool)
        spatial = spatial.mean(dim=[-2, -1])  # (N, spatial_dim)
        return {"pool": pool, "spatial": spatial, "logits": logits}
