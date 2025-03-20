from itertools import count

import torch
from torch import nn
from torch.nn import Sequential


class TGAN(nn.Module):
    """
    Table-GAN model
    """

    def __init__(self, n_feats, n_class, noise_dim, n_channels=64):
        """
        Args:
            n_feats (int): number of features
            n_class (int): number of classes
            noise_dim (int): dimensionality of noise vector
            n_channels (int): number of starting channels
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.n_class = n_class

        self._n_feats = n_feats

        # features are reshaped into square matrix (padded with zeros)
        # powers of two are used so that deconvolutions will work correctly
        for k in count(2):
            side = 2**k
            if side * side >= self._n_feats + self.n_class:
                self._d_side = side
                break

        d_dims = [(1, self._d_side), (n_channels, self._d_side // 2)]

        while d_dims[-1][1] != 1:
            d_dims.append([d_dims[-1][0] * 2, d_dims[-1][1] // 2])

        d_layers = []
        for i in range(len(d_dims) - 1):
            d_layers.append(nn.Conv2d(d_dims[i][0], d_dims[i + 1][0], 4, 2, 1))
            d_layers.append(nn.BatchNorm2d(d_dims[i + 1][0]))
            d_layers.append(nn.LeakyReLU())
        # add last convolution layer to condense channels to 1 scalar with kernel equal to side
        d_layers.append(nn.Conv2d(d_dims[-1][0], 1, d_dims[-1][1], 1, 0))
        d_layers.append(nn.Sigmoid())

        self.discriminator_net = Sequential(*d_layers)

        for k in count(2):
            side = 2**k
            if side * side >= self._n_feats:
                self._g_side = side
                break

        g_dims = [(1, self._g_side), (n_channels, self._g_side // 2)]

        while g_dims[-1][1] != 1:
            g_dims.append([g_dims[-1][0] * 2, g_dims[-1][1] // 2])

        # reverse of discriminator layers
        g_layers = [
            nn.ConvTranspose2d(
                self.noise_dim + self.n_class, g_dims[-1][0], g_dims[-1][1], 1, 0
            )
        ]
        for i in range(len(g_dims) - 1):
            g_layers.append(nn.BatchNorm2d(g_dims[-(i + 1)][0]))
            g_layers.append(nn.LeakyReLU())
            g_layers.append(
                nn.ConvTranspose2d(g_dims[-(i + 1)][0], g_dims[-(i + 2)][0], 4, 2, 1)
            )
        g_layers.append(nn.Tanh())

        self.generator_net = Sequential(*g_layers)

        # copy of discriminator
        for k in count(2):
            side = 2**k
            if side * side >= self._n_feats - 1:
                self._c_side = side
                break

        c_dims = [(1, self._c_side), (n_channels, self._c_side // 2)]

        while c_dims[-1][1] != 1:
            c_dims.append([c_dims[-1][0] * 2, c_dims[-1][1] // 2])

        c_layers = []
        for i in range(len(c_dims) - 1):
            c_layers.append(nn.Conv2d(c_dims[i][0], c_dims[i + 1][0], 4, 2, 1))
            c_layers.append(nn.BatchNorm2d(c_dims[i + 1][0]))
            c_layers.append(nn.LeakyReLU())
        # add last convolution layer to condense channels to 1 scalar with kernel equal to side
        c_layers.append(nn.Conv2d(c_dims[-1][0], 1, c_dims[-1][1], 1, 0))
        c_layers.append(nn.Sigmoid())

        self.classifier_net = Sequential(*c_layers)

    def discriminator(self, data: torch.Tensor, cond: torch.Tensor, **batch):
        """
        TGAN discriminator.

        Args:
            data (Tensor): input vector.
            cond (Tensor): conditional vector (i.e. one-hot encoding of class)
        Returns:
            output (dict): output dict containing logits and flattened features from penultimate layer for information loss.
        """
        # pad input with zeros and reshape into square matrices
        net_input = torch.cat((data, cond), 1)
        zeros = torch.zeros(
            net_input.shape[0], self._d_side * self._d_side - net_input.shape[1]
        ).to(net_input.device)
        net_input = torch.cat((net_input, zeros), 1)
        net_input = net_input.reshape(
            -1, 1, self._d_side, self._d_side
        )  # add channel dim
        intermediate = self.discriminator_net[:-2](net_input)
        logits = self.discriminator_net[-2:](intermediate)
        return {
            "logits": logits.reshape(-1, 1),
            "features": torch.flatten(intermediate, start_dim=1),
        }

    def generator(self, noise: torch.Tensor, cond: torch.Tensor, **batch):
        """
        TGAN generator.

        Args:
            noise (Tensor): input latent noise.
            cond (Tensor): conditional vector (i.e. one-hot encoding of class)
        Returns:
            output (Tensor): fake data object.
        """
        # pad input with zeros and reshape into square matrices
        net_input = torch.cat((noise, cond), 1)
        net_input = net_input[:, :, None, None]  # add dims for deconvs
        net_output = self.generator_net(net_input)
        net_output = net_output.reshape(-1, self._g_side * self._g_side)
        net_output = net_output[:, : self._n_feats]
        return {"fake_data": net_output}

    def classifier(self, data: torch.Tensor, **batch):
        """
        TGAN classifier.

        Args:
            data (Tensor): input vector.
        Returns:
            output (dict): output dict containing classifier logits.
        """
        # pad input with zeros and reshape into square matrices
        net_input = data
        zeros = torch.zeros(
            net_input.shape[0], self._c_side * self._c_side - net_input.shape[1]
        ).to(net_input.device)
        net_input = torch.cat((net_input, zeros), 1)
        net_input = net_input.reshape(
            -1, 1, self._c_side, self._c_side
        )  # add channel dim
        return {"c_logits": self.classifier_net(net_input).reshape(-1, 1)}

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
