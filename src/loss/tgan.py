import torch
from torch import nn

from src.loss.gan import GANLoss


class TGANLoss(GANLoss):
    """
    TGAN loss functions
    """

    def __init__(self):
        super().__init__()

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
                fake_c_logits.reshape(-1),
                torch.clip(generated_labels.type(torch.float), min=0, max=1),
            )
        }

    def information(
        self,
        real_mean: torch.Tensor,
        real_sd: torch.Tensor,
        fake_mean: torch.Tensor,
        fake_sd: torch.Tensor,
        **batch
    ):
        """
        Information loss function calculation logic.
        Args:
            real_mean (Tensor): mean of features on real data
            real_sd (Tensor): standard derivation of features on real data
            fake_mean (Tensor): mean of features on fake data
            fake_sd (Tensor): standard derivation of features on fake data
        Returns:
            information_loss (dict): dict containing calculated information loss function for discriminator.
        """
        return {
            "information_loss": torch.norm(real_mean - fake_mean, p=2)
            + torch.norm(real_sd - fake_sd, p=2)
        }
