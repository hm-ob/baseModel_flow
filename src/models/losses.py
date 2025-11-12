import torch
import torch.nn as nn
from configs.base import Config

class MSELoss(nn.Module):
    def __init__(self, cfg: Config, **kwargs):
        super(MSELoss, self).__init__()
        self.cfg = cfg
        self.loss_fn = nn.MSELoss(**kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input[0] if isinstance(input, (tuple, list)) else input
        return self.loss_fn(out, target)