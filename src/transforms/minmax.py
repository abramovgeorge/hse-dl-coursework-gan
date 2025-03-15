import torch
import torch.nn.functional as F
from torch import nn


class MinMaxScaler(nn.Module):
    """
    Transform scaling each column to [0, 1]
    """

    def __init__(self, data, device, config):
        """
        Args:
            data (TableDataset): table dataset.
            device (str): device of dataset
            config (DictConfig): hydra experiment config.
        """
        super().__init__()

        self._min = (
            torch.tensor(
                [data.table.iloc[:, i].min() for i in range(data.table.shape[1])]
            )
            .to(device)
            .float()
        )
        self._max = (
            torch.tensor(
                [data.table.iloc[:, i].max() for i in range(data.table.shape[1])]
            )
            .to(device)
            .float()
        )

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
