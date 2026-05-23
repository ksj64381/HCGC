"""
hcgc/_config.py -- Shared global config for HCGC.
"""

import types
import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_CFG = types.SimpleNamespace(
    node_types  = None,
    target_type = None,
    num_classes = None,
    dataset     = None,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def set_device(device):
    global DEVICE
    DEVICE = torch.device(device) if isinstance(device, str) else device
