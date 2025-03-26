import torch
from numpy.random import randint
from torch import nn

from src.loss.ctgan import CTGANLoss
from src.loss.tgan import TGANLoss


class CTABGANLoss(CTGANLoss, TGANLoss):
    """
    CTABGANLoss loss functions.
    """

    def __init__(self, transforms_info, discrete_columns, w_info=0.1):
        """
        Args:
            transforms_info (dict[dict]): info about transformed columns
            discrete_columns (list[str]): names of discrete_columns in original dataset
            w_info (float): weight of the information loss
        """
        super().__init__(
            transforms_info=transforms_info,
            discrete_columns=discrete_columns,
            w_info=w_info,
        )
