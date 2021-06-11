"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size, latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels,
                                               kernel_size, **cnn_kwargs)
        # If input image is 128x128 do five convolutions
        if self.img_size[1] == self.img_size[2] == 128:
            self.convT1_128 = nn.ConvTranspose2d(hid_channels, hid_channels,
                                                 kernel_size, **cnn_kwargs)
            self.convT2_128 = nn.ConvTranspose2d(hid_channels, hid_channels,
                                                 kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels,
                                         kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels,
                                         kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size,
                                         **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        if self.img_size[1] == self.img_size[2] == 128:
            x = torch.relu(self.convT1_128(x))
            x = torch.relu(self.convT2_128(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


class DecoderClimate(nn.Module):
    def __init__(self, img_size=(1, 72, 144), latent_dim=10):
        r"""Encoder of the model proposed in [1], adapted for rectangular climate data.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. Architecture for climate data at a resolution of 2.5 degrees.

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers
            - The first 3  with 32 channels, 4 x 4 kernel, stride of 2
            - The last one with 16 channels, 3 x 3 kernel, stride of 2
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(DecoderClimate, self).__init__()

        # if img_size[1] != 72 or img_size[2] != 144:
        #     raise ValueError('Climate Decoder architecture only supported for (None, 72, 144) image size.')

        self.img_size = img_size
        n_chan = self.img_size[0]

        # Layer parameters
        hid_channels = {'first': 32, 'last': 16}
        kernel_size = {'first': 4, 'last': (3, 4)}
        hidden_dim = 256

        # Shape required to start transpose convs
        self.reshape = (hid_channels['last'], 4, 8)

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = {'first': dict(stride=2, padding=1),
                       'last': dict(stride=2, padding=0)}
        self.convT1 = nn.ConvTranspose2d(hid_channels['last'], hid_channels['first'], kernel_size['last'], **cnn_kwargs['last'])
        self.convT2 = nn.ConvTranspose2d(hid_channels['first'], hid_channels['first'], kernel_size['first'], **cnn_kwargs['first'])
        self.convT3 = nn.ConvTranspose2d(hid_channels['first'], hid_channels['first'], kernel_size['first'], **cnn_kwargs['first'])
        self.convT4 = nn.ConvTranspose2d(hid_channels['first'], n_chan,  kernel_size['first'], **cnn_kwargs['first'])

        # Default padding=0 in Conv2d as we apply SphericalPad instead
        # Not sure...
        # self.pad = SphericalPad(lat_pad=1, lon_pad=1)

    def forward(self, z):
        # B x L
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        # B x H
        x = torch.relu(self.lin2(x))
        # B x H
        x = torch.relu(self.lin3(x))
        # B x (16 · 4 · 8)

        x = x.view(batch_size, *self.reshape)
        # B x 16 x 4 x 8

        # Convolutional layers with ReLu activations
        x = torch.relu(self.convT1(x))
        # B x 32 x 9 x 18
        x = torch.relu(self.convT2(x))
        # B x 32 x 18 x 36
        x = torch.relu(self.convT3(x))
        # B x 32 x 36 x 72

        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT4(x))
        # B x C x 72 x 144

        return x
