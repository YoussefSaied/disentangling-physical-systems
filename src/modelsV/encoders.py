"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


# ALL encoders should be called Encoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size, latent_dim=10):
        r"""Encoder of the model proposed in [1].

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
            - 1 fully connected layer of latent_dim units 

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size,
                               **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size,
                               **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size,
                                     **cnn_kwargs)
        
        # If input image is 128x128 do fifth convolution
        if self.img_size[1] == self.img_size[2] == 128:
            self.conv1_128 = nn.Conv2d(hid_channels, hid_channels, kernel_size,
                                       **cnn_kwargs)
            self.conv2_128 = nn.Conv2d(hid_channels, hid_channels, kernel_size,
                                       **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, self.latent_dim )

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
        if self.img_size[1] == self.img_size[2] == 128:
            x = torch.relu(self.conv1_128(x))
            x = torch.relu(self.conv2_128(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)

        return x


class SphericalPad(nn.Module):
    def __init__(self, lat_pad=1, lon_pad=1):
        """Module wrapping the functional interface of spherical padding.

        Args:
            lat_pad (int, optional): Padding on latitude dimension. Defaults to 1.
            lon_pad (int, optional): Padding on longitude dimension. Defaults to 1.
        """
        super(SphericalPad, self).__init__()
        self.lat_pad = lat_pad
        self.lon_pad = lon_pad
    
    def forward(self, x):
        """Forward pass of padding module.

        Args:
            x (torch.Tensor): Batch of data. Shape (batch_size, n_chan, height, width)

        Height dimension corresponds to latitude. 
        In this dimension reflective padding is applied and the padded rows are rolled.
        Width dimension corresponds to longitude and circular padding is applied.

        Example:
            >>> x = torch.arange(8, dtype=torch.float).view(1, 1, 2, 4)
            >>> x
            tensor([[[[0., 1., 2., 3.],
                      [4., 5., 6., 7.]]]])
            >>> SphericalPad().forward(x)
            tensor([[[[5., 6., 7., 4., 5., 6.],
                      [3., 0., 1., 2., 3., 0.],
                      [7., 4., 5., 6., 7., 4.],
                      [1., 2., 3., 0., 1., 2.]]]])
        """
        # Add reflective padding on the latitude (height) dimension
        x = F.pad(x, (0, 0, self.lat_pad, self.lat_pad), mode='reflect')
       
        # Roll the northernmost and southernmost padded rows by the equivalent of 180 degrees of longitude
        x[..., :self.lat_pad, :] = torch.roll(x[..., :self.lat_pad, :],
                                              shifts=int(x.shape[-1] / 2),
                                              dims=-1)
        x[..., -self.lat_pad:, :] = torch.roll(x[..., -self.lat_pad:, :],
                                               shifts=int(x.shape[-1] / 2),
                                               dims=-1)
        
        # Add circular padding on the longitude (width) dimension
        x = F.pad(x, (self.lon_pad, self.lon_pad, 0, 0), mode='circular')

        return x


class EncoderClimate(nn.Module):
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
            - The last one with 16 channels, 3 x 4 kernel, stride of 2
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of latent_dim units
        """
        super().__init__()

        # if img_size[1] != 72 or img_size[2] != 144:
        #     raise ValueError('Climate Encoder architecture only supported for (None, 72, 144) image size.')

        self.img_size = img_size
        self.latent_dim = latent_dim

        # Layer parameters
        n_chan = self.img_size[0]
        hid_channels = {'first': 32, 'last': 16}
        kernel_size = {'first': 4, 'last': (3, 4)}
        hidden_dim = 256

        # Convolutional layers
        cnn_kwargs = dict(stride=2)
        # Default padding=0 in Conv2d as we apply SphericalPad instead
        self.pad = SphericalPad(lat_pad=1, lon_pad=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels['first'], kernel_size['first'], **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels['first'], hid_channels['first'], kernel_size['first'], **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels['first'], hid_channels['first'], kernel_size['first'], **cnn_kwargs)
        self.conv4 = nn.Conv2d(hid_channels['first'], hid_channels['last'],  kernel_size['last'], **cnn_kwargs)

        # Shape required to start transpose convs
        self.reshape = (hid_channels['last'], 4, 8)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, self.latent_dim)

    def forward(self, x):
        # B x C x 72 x 144
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(self.pad(x)))
        # B x 32 x 36 x 72
        x = torch.relu(self.conv2(self.pad(x)))
        # B x 32 x 18 x 36
        x = torch.relu(self.conv3(self.pad(x)))
        # B x 32 x 9 x 18
        x = torch.relu(self.conv4(x))
        # B x 16 x 4 x 8

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        # B x 512
        x = torch.relu(self.lin1(x))
        # B x 256
        x = torch.relu(self.lin2(x))
        # B x 256
        x = torch.relu(self.lin3(x))
        # B x L

        return x


class EncoderClimateVAE(nn.Module):
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
            - The last one with 16 channels, 3 x 4 kernel, stride of 2
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(EncoderClimateVAE, self).__init__()

        if img_size[1] != 72 or img_size[2] != 144:
            raise ValueError('Climate Encoder architecture only supported for (None, 72, 144) image size.')

        self.img_size = img_size
        self.latent_dim = latent_dim

        # Layer parameters
        n_chan = self.img_size[0]
        hid_channels = {'first': 32, 'last': 16}
        kernel_size = {'first': 4, 'last': (3, 4)}
        hidden_dim = 256

        # Convolutional layers
        cnn_kwargs = dict(stride=2)
        # Default padding=0 in Conv2d as we apply SphericalPad instead
        self.pad = SphericalPad(lat_pad=1, lon_pad=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels['first'], kernel_size['first'], **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels['first'], hid_channels['first'], kernel_size['first'], **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels['first'], hid_channels['first'], kernel_size['first'], **cnn_kwargs)
        self.conv4 = nn.Conv2d(hid_channels['first'], hid_channels['last'],  kernel_size['last'], **cnn_kwargs)

        # Shape required to start transpose convs
        self.reshape = (hid_channels['last'], 4, 8)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        # B x C x 72 x 144
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(self.pad(x)))
        # B x 32 x 36 x 72
        x = torch.relu(self.conv2(self.pad(x)))
        # B x 32 x 18 x 36
        x = torch.relu(self.conv3(self.pad(x)))
        # B x 32 x 9 x 18
        x = torch.relu(self.conv4(x))
        # B x 16 x 4 x 8

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        # B x 512
        x = torch.relu(self.lin1(x))
        # B x 256
        x = torch.relu(self.lin2(x))
        # B x 256

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        # B x 2L
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar




class DecoderClimateVAE(nn.Module):
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
        super(DecoderClimateVAE, self).__init__()

        if img_size[1] != 72 or img_size[2] != 144:
            raise ValueError('Climate Decoder architecture only supported for (None, 72, 144) image size.')

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
