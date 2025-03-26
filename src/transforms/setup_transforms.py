import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import open_dict
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
from torch import nn

from src.transforms.ctgan import ContinuousTransform
from src.utils.io_utils import ROOT_PATH


def setup_minmax(dataset, config):
    """
    Args:
        dataset (TableDataset): table dataset.
        config (DictConfig): hydra experiment config.
    """
    min_t = torch.tensor(
        [dataset.table.iloc[:, i].min() for i in range(dataset.table.shape[1])]
    ).float()
    max_t = torch.tensor(
        [dataset.table.iloc[:, i].max() for i in range(dataset.table.shape[1])]
    ).float()

    path = ROOT_PATH / "transforms_data" / "minmax"
    path.mkdir(exist_ok=True, parents=True)

    with open(path / "min", "wb") as f:
        pickle.dump(min_t, f)
    with open(path / "max", "wb") as f:
        pickle.dump(max_t, f)


def setup_ctgan(dataset, config, max_clusters):
    """
    Args:
        dataset (TableDataset): table dataset.
        config (DictConfig): hydra experiment config.
        max_clusters (int): max clusters for gaussian mixture transform.
    """

    dataset_name = config.datasets["train"]["path"].split("/")[-1][:-4]

    path = ROOT_PATH / "transforms_data" / "ctgan" / dataset_name

    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

        columns = dataset.table.columns.to_list()
        transforms = dict()
        transforms_info = dict()
        cond_len = 0
        idx = 0
        for column_name in dataset.table.columns:
            if column_name in dataset.discrete_columns:
                transform = OneHotEncoder()
                transform.fit(dataset.table, column=column_name)
                cond_len += len(transform.dummies)
                transforms_info[column_name] = dict()
                transforms_info[column_name]["rle"] = [idx, len(transform.dummies)]
                transforms_info[column_name]["map"] = dict(
                    (str(transform.dummies[i]), i)
                    for i in range(len(transform.dummies))
                )
                idx += transforms_info[column_name]["rle"][1]
            else:
                transform = ContinuousTransform(max_clusters=max_clusters)
                transform.fit(dataset.table, column=column_name)
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
            transforms[column_name] = transform

        dataset.original_table = dataset.table
        for _, transform in transforms.items():
            dataset.table = transform.transform(dataset.table)
        transformed_columns = dataset.table.columns.to_list()

        with open(path / "columns", "wb") as f:
            pickle.dump(columns, f)
        with open(path / "transformed_columns", "wb") as f:
            pickle.dump(transformed_columns, f)
        with open(path / "transforms", "wb") as f:
            pickle.dump(transforms, f)
        with open(path / "cond_len", "wb") as f:
            pickle.dump(cond_len, f)
        with open(path / "transforms_info", "wb") as f:
            pickle.dump(transforms_info, f)
        dataset.original_table.to_csv(path / "original_table", index=False)
        dataset.table.to_csv(path / "table", index=False)
    else:
        with open(path / "columns", "rb") as f:
            columns = pickle.load(f)
        with open(path / "transformed_columns", "rb") as f:
            transformed_columns = pickle.load(f)
        with open(path / "transforms", "rb") as f:
            transforms = pickle.load(f)
        with open(path / "cond_len", "rb") as f:
            cond_len = pickle.load(f)
        with open(path / "transforms_info", "rb") as f:
            transforms_info = pickle.load(f)
        dataset.original_table = pd.read_csv(path / "original_table")
        dataset.table = pd.read_csv(path / "table")

    with open(path.parent / "columns", "wb") as f:
        pickle.dump(columns, f)
    with open(path.parent / "transformed_columns", "wb") as f:
        pickle.dump(transformed_columns, f)
    with open(path.parent / "transforms", "wb") as f:
        pickle.dump(transforms, f)

    # set parameters for other modules
    with open_dict(config):
        config.model.n_feats = len(transformed_columns)
        config.model.cond_len = cond_len
        config.model.transforms_info = transforms_info
        config.dataloader.transforms_info = transforms_info
        if "loss_function" in config:  # inference does not have a loss function
            config.loss_function.transforms_info = transforms_info
            config.loss_function.discrete_columns = dataset.discrete_columns


def setup_transforms(dataset, config):
    """
    Args:
    data (TableDataset): table dataset.
    config (DictConfig): hydra experiment config.
    """
    setup_fns = {
        "minmax": setup_minmax,
        "ctgan": setup_ctgan,
    }
    setup_names = config.transforms.setup
    for name, args in setup_names.items():
        setup_fns[name](dataset, config, **args)
