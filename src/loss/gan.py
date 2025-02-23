import torch
from torch import nn


class GANLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def discriminator(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor, **batch
    ):
        """
        Discriminator loss function calculation logic.
        Args:
            real_logits (Tensor): discriminator outputs on the real data.
            fake_logits (Tensor): discriminator outputs on the fake data.
        Returns:
            discriminator_loss (dict): dict containing calculated discriminator loss function.
        """
        real_labels = torch.ones(real_logits.shape).to(real_logits.device)
        fake_labels = torch.zeros(fake_logits.shape).to(fake_logits.device)
        return {
            "discriminator_loss": self.criterion(real_logits, real_labels)
            + self.criterion(fake_logits, fake_labels)
        }

    def generator(self, fake_logits: torch.Tensor, **batch):
        """
        Generator loss function calculation logic.
        Args:
            fake_logits (Tensor): discriminator outputs on the fake data.
        Returns:
            generator_loss (dict): dict containing calculated generator loss function.
        """
        real_labels = torch.ones(fake_logits.shape).to(fake_logits.device)
        return {"generator_loss": self.criterion(fake_logits, real_labels)}
