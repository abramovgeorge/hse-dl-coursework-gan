import torch
from numpy.random import randint
from torch import nn

from src.loss.ctgan import CTGANLoss
from src.loss.tgan import TGANLoss


class CTABGANLoss(CTGANLoss, TGANLoss):
    """
    CTABGANLoss loss functions.
    """

    def __init__(self, transforms_info, discrete_columns):
        """
        transforms_info (dict[dict]): info about transformed columns
        discrete_columns (list[str]): names of discrete_columns in original dataset
        """
        super().__init__(transforms_info, discrete_columns)
        self._cond_criterion = nn.CrossEntropyLoss(reduction="none")
        self._transforms_info = transforms_info
        self._discrete_columns = discrete_columns
