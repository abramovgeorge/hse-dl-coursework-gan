import pickle

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.io_utils import ROOT_PATH


class MinMaxScaler(nn.Module):
    """
    Transform scaling each column to [0, 1]
    """

    def __init__(self):
        super().__init__()

        path = ROOT_PATH / "transforms_data" / "minmax"

        # nn.Parameter allows for automatic transfer to device when calling .to(device) on the transform object
        with open(path / "min", "rb") as f:
            self._min = nn.Parameter(pickle.load(f))
        with open(path / "max", "rb") as f:
            self._max = nn.Parameter(pickle.load(f))

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): transformed tensor.
        """
        return (x - self._min.expand_as(x)) / ((self._max - self._min).expand_as(x))

    def inverse(self, x):
        """
        Args:
            x (Tensor): transformed tensor.
        Returns:
            x (Tensor): original tensor.
        """
        return x * (self._max - self._min).expand_as(x) + self._min.expand_as(x)
