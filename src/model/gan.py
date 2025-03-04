import torch
from torch import nn
from torch.nn import Sequential


class GAN(nn.Module):
    """
    Conditional Generative Adversarial Networks model
    """

    def __init__(self, n_feats, n_class, noise_dim, fc_hidden=256):
        """
        Args:
            n_feats (int): number of features
            n_class (int): number of classes.
            noise_dim (int): dimensionality of noise vector
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.n_class = n_class

        self._n_feats = n_feats

        self.discriminator_net = Sequential(
            nn.Linear(in_features=self._n_feats + self.n_class, out_features=fc_hidden),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden, out_features=1),
            nn.Sigmoid(),
        )

        self.generator_net = Sequential(
            nn.Linear(
                in_features=self.noise_dim + self.n_class, out_features=fc_hidden
            ),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden, out_features=self._n_feats),
            nn.Tanh(),
        )

    def discriminator(self, data: torch.Tensor, cond: torch.Tensor, **batch):
        """
        GAN discriminator.

        Args:
            data (Tensor): input vector.
            cond (Tensor): conditional vector (i.e. one-hot encoding of class)
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.discriminator_net(torch.cat((data, cond), 1))}

    def generator(self, noise: torch.Tensor, cond: torch.Tensor, **batch):
        """
        GAN generator.

        Args:
            noise (Tensor): input latent noise.
            cond (Tensor): conditional vector (i.e. one-hot encoding of class)
        Returns:
            output (Tensor): fake data object.
        """
        return {"fake_data": self.generator_net(torch.cat((noise, cond), 1))}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
