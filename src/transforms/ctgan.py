import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import open_dict
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
from torch import nn


class CTGANTransform(nn.Module):
    """
    Transform for the data according to CTGAN paper
    """

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

    def __init__(self, data, device, config, max_clusters=10):
        """
        Args:
            data (TableDataset): table dataset.
            device (str): device of dataset.
            config (DictConfig): hydra experiment config.
            max_clusters (int): max clusters for gaussian mixture transform.
        """
        super().__init__()

        self._device = device

        self._columns = data.table.columns.to_list()
        self._transforms = dict()
        transforms_info = dict()
        cond_len = 0
        idx = 0
        for column_name in data.table.columns:
            if column_name in data.discrete_columns:
                transform = OneHotEncoder()
                transform.fit(data.table, column=column_name)
                cond_len += len(transform.dummies)
                transforms_info[column_name] = dict()
                transforms_info[column_name]["rle"] = [idx, len(transform.dummies)]
                transforms_info[column_name]["map"] = dict(
                    (str(transform.dummies[i]), i)
                    for i in range(len(transform.dummies))
                )
                idx += transforms_info[column_name]["rle"][1]
            else:
                transform = self.ContinuousTransform(max_clusters=max_clusters)
                transform.fit(data.table, column=column_name)
                transforms_info[column_name] = dict()
                transforms_info[f"{column_name}.component"] = dict()
                transforms_info[column_name]["rle"] = [
                    idx,
                    1,
                ]  # normalized value from gm, continuous
                transforms_info[f"{column_name}.component"]["rle"] = [
                    idx + 1,
                    len(transform.dummies),
                ]  # onehot
                transforms_info[f"{column_name}.component"]["map"] = dict(
                    (str(transform.dummies[i]), i)
                    for i in range(len(transform.dummies))
                )
                idx += transforms_info[f"{column_name}.component"]["rle"][1] + 1
            self._transforms[column_name] = transform

        tmp = pd.DataFrame([data.table.iloc[0]], columns=data.table.columns)
        for _, transform in self._transforms.items():
            tmp = transform.transform(tmp)
        self._transformed_columns = tmp.columns.to_list()

        # set parameters for other modules
        with open_dict(config):
            config.model.n_feats = len(self._transformed_columns)
            config.model.cond_len = cond_len
            config.model.transforms_info = transforms_info
            config.dataloader.transforms_info = transforms_info
            if "loss_function" in config:  # inference does not have a loss function
                config.loss_function.transforms_info = transforms_info
                config.loss_function.discrete_columns = data.discrete_columns

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
        return torch.Tensor(tmp.to_numpy()).to(self._device)

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
        return torch.Tensor(tmp.to_numpy()).to(self._device)
