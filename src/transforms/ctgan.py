import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
from torch import nn

from src.utils.io_utils import ROOT_PATH


class ContinuousTransform:
    """
    Transform for the continuous columns, wrapper around ClusterBasedNormalizer with onehot encoding for the components
    """

    def __init__(self, max_clusters):
        self._gm = ClusterBasedNormalizer(max_clusters=max_clusters)
        self._onehot = OneHotEncoder()

    def fit(self, data, column):
        self._gm.fit(data, column)
        self._onehot.fit(self._gm.transform(data), column=f"{column}.component")
        self.dummies = self._onehot.dummies

    def transform(self, data):
        return self._onehot.transform(self._gm.transform(data))

    def reverse_transform(self, data):
        return self._gm.reverse_transform(self._onehot.reverse_transform(data))


class CTGANTransform(nn.Module):
    """
    Transform for the data according to CTGAN paper
    """

    def __init__(self):
        super().__init__()

        path = ROOT_PATH / "transforms_data" / "ctgan"

        with open(path / "columns", "rb") as f:
            self._columns = pickle.load(f)
        with open(path / "transformed_columns", "rb") as f:
            self._transformed_columns = pickle.load(f)
        with open(path / "transforms", "rb") as f:
            self._transforms = pickle.load(f)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): transformed tensor.
        """
        tmp = pd.DataFrame(x.cpu().numpy(), columns=self._columns)
        for _, transform in self._transforms.items():
            tmp = transform.transform(tmp)
        return torch.Tensor(tmp.to_numpy()).to(x.device)

    def inverse(self, x):
        """
        Args:
            x (Tensor): transformed tensor.
        Returns:
            x (Tensor): original tensor.
        """
        tmp = pd.DataFrame(x.cpu().detach().numpy(), columns=self._transformed_columns)
        for _, transform in self._transforms.items():
            tmp = transform.reverse_transform(tmp)
        return torch.Tensor(tmp.to_numpy()).to(x.device)
