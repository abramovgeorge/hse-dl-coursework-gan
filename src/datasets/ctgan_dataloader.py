from copy import deepcopy
from math import log

import numpy as np
import torch
from numpy.random import choice, randint


class CTGANDataloader:
    """
    Dataloader for CTGAN
    """

    def __init__(
        self,
        dataset,
        batch_size,
        transforms_info,
        collate_fn,
        drop_last,
        shuffle,
        worker_init_fn,
    ):
        """
        Args:
            dataset (TableDataset): table dataset.
            batch_size (int): size of batch
            transforms_info (dict[dict]): info about transformed columns
            collate_fn (func): collate function for dataset items
            drop_last: unused, included for naming cohesion in data_utils.py
            shuffle: unused, included for naming cohesion in data_utils.py
            worker_init_fn: unused, included for naming cohesion in data_utils.py
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._transforms_info = transforms_info

        self._idx = dict()
        self._cond_start = (
            dict()
        )  # starting indices of discrete features in cond vector
        prev = 0
        for discrete_column in self._dataset.discrete_columns:
            _, length = self._transforms_info[discrete_column]["rle"]
            self._idx[discrete_column] = [[] for _ in range(length)]
            self._cond_start[discrete_column] = prev
            prev += length
        self._cond_len = prev
        for i in range(len(self._dataset)):
            for discrete_column in self._dataset.discrete_columns:
                value = self._dataset.original_table.iloc[i][discrete_column]
                d_map = self._transforms_info[discrete_column]["map"]
                self._idx[discrete_column][d_map[str(int(value))]].append(i)

        self._log_frequencies = dict()
        for discrete_column in self._dataset.discrete_columns:
            tmp = deepcopy(self._idx[discrete_column])
            for i in range(len(tmp)):
                tmp[i] = (
                    log(len(tmp[i])) if len(tmp[i]) != 0 else 0
                )  # should not happen
            sm = sum(tmp)
            for i in range(len(tmp)):
                tmp[i] /= sm
            self._log_frequencies[discrete_column] = tmp

    def __iter__(self):
        return self

    def get_cond_vector(self, column, value):
        """
        Get conditional vector satisfying the condition column = value

        Args:
            column (str): name of the discrete column
            value (int): value of said column
        Returns:
            cond (Tensor): conditional vector
        """
        return self._get_cond_vector(
            column, self._transforms_info[column]["map"][str(int(value))]
        )

    def _get_cond_vector(self, column, value):
        """
        Get conditional vector satisfying the condition column = encoded value

        Args:
            column (str): name of the discrete column
            value (int): value of said column
        Returns:
            cond (Tensor): conditional vector
        """
        cond = torch.zeros(self._cond_len)
        cond[self._cond_start[column] + value] = 1
        return cond

    def _get_object(self, d_i):
        """
        Get single object using training-by-sampling technique from CTGAN paper

        Args:
            d_i (str): index of the chosen discrete column
        Returns:
            index (int): index of the chosen item from dataset
            v (int): value in this column
        """
        d = self._dataset.discrete_columns[d_i]
        v = choice(
            range(self._transforms_info[d]["rle"][1]), p=self._log_frequencies[d]
        )
        tmp = self._idx[d][v]
        # this is ~10x faster than choice
        index = tmp[randint(0, len(tmp))]
        return index, v

    def __next__(self):
        items, cond_infos = [], []
        d_is = choice(len(self._dataset.discrete_columns), self._batch_size)
        for d_i in d_is:
            index, v = self._get_object(int(d_i))
            items.append(index)
            cond_infos.append((d_i, v))
        batch_item = self._dataset[np.array(items)]
        cond = torch.zeros(self._batch_size, self._cond_len)
        mask = torch.zeros(self._batch_size, len(self._dataset.discrete_columns))
        mask[np.arange(self._batch_size), d_is] = 1
        for i, (d_i, v) in enumerate(cond_infos):
            cond[i, self._cond_start[self._dataset.discrete_columns[int(d_i)]] + v] = 1
        batch_item["cond"] = cond
        batch_item["mask"] = mask
        return batch_item
