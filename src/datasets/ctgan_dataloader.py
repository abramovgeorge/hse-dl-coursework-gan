from copy import deepcopy
from math import log

import torch
from numpy.random import choice


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
                value = self._dataset.table.iloc[i][discrete_column]
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
            value (str): value of said column
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
            value (str): value of said column
        Returns:
            cond (Tensor): conditional vector
        """
        cond = torch.zeros(self._cond_len)
        cond[self._cond_start[column] + value] = 1
        return cond

    def _get_object(self):
        """
        Get single object using training-by-sampling technique from CTGAN paper

        Returns:
            item (dict): dict, containing item from dataset
            cond_info (dict): dict, containing he cond vector info
        """
        d_i = choice(len(self._dataset.discrete_columns))
        d = self._dataset.discrete_columns[d_i]
        v = choice(
            range(self._transforms_info[d]["rle"][1]), p=self._log_frequencies[d]
        )
        item = self._dataset[choice(self._idx[d][v])]
        mask = torch.zeros(len(self._dataset.discrete_columns))
        mask[d_i] = 1
        cond_info = {"cond": self._get_cond_vector(d, v), "mask": mask}
        return item, cond_info

    def _collate_cond_fn(self, cond_infos):
        """
        Converts individual cond_info into a batch.

        Args:
            cond_infos (list[dict]): list of cond_info objects
        Returns:
            result_batch (dict[Tensor]): dict, containing batch-version
                of the cond_info objects.
        """
        result_batch = {}

        result_batch["cond"] = torch.vstack([elem["cond"] for elem in cond_infos])
        result_batch["mask"] = torch.vstack([elem["mask"] for elem in cond_infos])

        return result_batch

    def __next__(self):
        items, cond_infos = [], []
        for _ in range(self._batch_size):
            item, cond_info = self._get_object()
            items.append(item)
            cond_infos.append(cond_info)
        batch_item = self._collate_fn(items)
        batch_cond_info = self._collate_cond_fn(cond_infos)
        batch_item.update(batch_cond_info)
        return batch_item
