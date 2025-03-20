import torch
from torch import nn
from torch.nn import Sequential
from torch.nn.functional import gumbel_softmax, tanh


class CTGAN(nn.Module):
    """
    Conditional Generative Adversarial Networks model
    """

    class Residual(nn.Module):
        """
        Residual FC module for conditional generator
        """

        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.bn = nn.BatchNorm1d(out_features)
            self.relu = nn.ReLU()

        def forward(self, x):
            output = x
            output = self.linear(output)
            output = self.bn(output)
            output = self.relu(output)
            return torch.cat((x, output), dim=1)

    class Activations(nn.Module):
        """
        Activations module for conditional generator
        """

        def __init__(self, transforms_info):
            super().__init__()
            self._transforms_info = transforms_info

        def forward(self, x):
            output = torch.zeros(x.shape).to(x.device)
            for info in self._transforms_info.values():
                if "map" not in info.keys():  # continuous
                    i = info["rle"][0]
                    output[:, i] = tanh(x[:, i])
                else:
                    i = info["rle"][0]
                    length = info["rle"][1]
                    output[:, i : i + length] = gumbel_softmax(x[:, i : i + length])
            return output

    def __init__(
        self, n_feats, cond_len, transforms_info, noise_dim, fc_hidden=256, pac=1
    ):
        """
        Args:
            n_feats (int): number of features
            cond_len (int): length of the conditional vector.
            transforms_info (dict[dict]): info about transformed columns
            noise_dim (int): dimensionality of noise vector
            fc_hidden (int): number of hidden features.
            pac (int): number of data objects stacked in discriminator
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.n_class = 2  # used for consistency in GANTrainer.py

        self._pac = pac

        self.discriminator_net = Sequential(
            nn.Linear(
                in_features=self._pac * (n_feats + cond_len), out_features=fc_hidden
            ),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden, out_features=1),
            nn.Sigmoid(),
        )

        self.generator_net = Sequential(
            self.Residual(
                in_features=self.noise_dim + cond_len, out_features=fc_hidden
            ),
            self.Residual(
                in_features=self.noise_dim + cond_len + fc_hidden,
                out_features=fc_hidden,
            ),
            nn.Linear(
                in_features=self.noise_dim + cond_len + 2 * fc_hidden,
                out_features=n_feats,
            ),
            self.Activations(transforms_info=transforms_info),
        )

    def discriminator(self, data: torch.Tensor, cond: torch.Tensor, **batch):
        """
        CTGAN discriminator, batch is reshaped according to PacGAN architecture.

        Args:
            data (Tensor): input vector.
            cond (Tensor): conditional vector (i.e. one-hot encoding of class)
        Returns:
            output (dict): output dict containing logits.
        """
        net_input = torch.cat((data, cond), 1)
        batch_size = net_input.shape[0]
        assert (
            batch_size % self._pac == 0
        ), "The batch size should be divisible by pac parameter"
        net_input = net_input.reshape((batch_size // self._pac, -1))
        return {"logits": self.discriminator_net(net_input)}

    def generator(self, noise: torch.Tensor, cond: torch.Tensor, **batch):
        """
        CTGAN generator.

        Args:
            noise (Tensor): input latent noise.
            cond (Tensor): conditional vector (i.e. one-hot encoding of class)
        Returns:
            output (dict): dict, containing fake data object before activation layer and after.
        """
        logits = self.generator_net[:-1](torch.cat((noise, cond), 1))
        fake_data = self.generator_net[-1](logits)

        return {"fake_data": fake_data, "fake_data_logits": logits}

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
