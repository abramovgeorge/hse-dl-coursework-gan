import torch
import torch.nn.functional as F
from torch import nn


class OneHot(nn.Module):
    """
    Transform wrapper of torch.nn.functional.one_hot
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): number of classes in data.
        """
        super().__init__()

        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): transformed tensor.
        """
        return F.one_hot(x, num_classes=self.num_classes)
