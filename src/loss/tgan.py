import torch
from torch import nn

from src.loss.gan import GANLoss


class TGANLoss(GANLoss):
    """
    TGAN loss functions
    """

    def __init__(self, w_info=0.1, **kwargs):
        """
        Args:
            w_info (float): weight of the information loss
            kwargs (dict): other arguments for cooperative inheritance with other losses
        """
        super().__init__(**kwargs)
        self.w_info_ = w_info

    def classifier(self, c_logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Classifier loss function calculation logic.
        Args:
            c_logits (Tensor): classifier outputs.
            labels (Tensor): data labels.
        Returns:
            classifier_loss (dict): dict containing calculated classifier loss function.
        """
        return {
            "classifier_loss": self.criterion(
                c_logits, labels.max(axis=1).indices.reshape(-1, 1).type(torch.float)
            )
        }

    def classification(
        self, fake_c_logits: torch.Tensor, generated_labels: torch.Tensor, **batch
    ):
        """
        Classification loss function calculation logic.
        Args:
            fake_c_logits (Tensor): classifier outputs on fake data.
            generated_labels (Tensor): generated data labels.
        Returns:
            classification_loss (dict): dict containing calculated classification loss function for discriminator.
        """
        return {
            "classification_loss": self.criterion(
                torch.clip(fake_c_logits.reshape(-1), min=1e-5, max=1 - 1e-5),
                torch.clip(generated_labels.type(torch.float), min=1e-5, max=1 - 1e-5),
            )
        }

    def information(self, real_features, fake_features, **batch):
        """
        Information loss function calculation logic.
        Args:
            real_features (Tensor): flattened features of real data from discriminator
            fake_features (Tensor): flattened features of fake data from discriminator
        Returns:
            information_loss (dict): dict containing calculated information loss function for discriminator.
        """
        real_mean = torch.mean(real_features, dim=0)
        real_sd = torch.std(real_features, dim=0)
        fake_mean = torch.mean(fake_features, dim=0)
        fake_sd = torch.std(fake_features, dim=0)
        return {
            "information_loss": (
                torch.norm(real_mean - fake_mean, p=2)
                + torch.norm(real_sd - fake_sd, p=2)
            )
            * self.w_info_
        }
