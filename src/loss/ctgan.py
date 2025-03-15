import torch
from numpy.random import randint
from torch import nn

from src.loss.gan import GANLoss


class CTGANLoss(GANLoss):
    """
    CTGANLoss loss functions.
    """

    def __init__(self, transforms_info, discrete_columns):
        """
        transforms_info (dict[dict]): info about transformed columns
        discrete_columns (list[str]): names of discrete_columns in original dataset
        """
        super().__init__()
        self._cond_criterion = nn.CrossEntropyLoss(reduction="none")
        self._transforms_info = transforms_info
        self._discrete_columns = discrete_columns

    def conditional(self, cond, mask, fake_data_logits, fake_data, **batch):
        """
        Conditional loss function calculation logic.
        Args:
            cond (Tensor): conditional vector.
            mask (Tensor): mask matrix, a[i][j] = 1, if ith object in batch has jth discrete feature selected in cond.
            fake_data_logits (Tensor): generated data before activations.
        Returns:
            conditional_loss (dict): dict containing calculated conditional loss function.
        """
        losses = []
        cond_idx = 0
        for discrete_column in self._discrete_columns:
            start, length = self._transforms_info[discrete_column]["rle"]
            logits = fake_data_logits[:, start : start + length]
            labels = torch.argmax(cond[:, cond_idx : cond_idx + length], dim=1)
            losses.append(self._cond_criterion(logits, labels))
            cond_idx += length
        loss = torch.stack(losses, dim=1)
        loss *= mask

        return {"conditional_loss": loss.sum(dim=1).mean()}
