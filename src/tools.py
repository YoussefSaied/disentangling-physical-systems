import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
# Cartopy (https://scitools.org.uk/cartopy/docs/latest/installing.html) has a dependencies on
    # GEOS (https://trac.osgeo.org/geos/)
    # PROJ (https://proj4.org/)
    
    # Also installed libgeos++-dev after the following error
    # ImportError: libproj.so.19: cannot open shared object file: No such file or directory

import netCDF4

import torch.nn.init as init

def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1],1)

    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    else:
        return "dimenional error"


def weights_init(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)




def plot_climate_data(temperature, lats, lons, timestamp, min_temp=210, max_temp=310, savefile=False, filename=None):
    
    # lats, lons = temperature.shape[0], temperature.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(16, 9),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    mesh = ax.pcolormesh(add_cyclic_point(lons), lats, temperature,
                         transform=ccrs.PlateCarree(),
                         vmin=min_temp, vmax=max_temp,
                         cmap='jet')

    title = ax.set_title(f'Max Planck Institute for Meteorology Model for {timestamp}')

    # ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.coastlines()

    ax_divider = make_axes_locatable(ax)
    ax_colorbar = ax_divider.new_horizontal(size='2%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_colorbar)

    colorbar = plt.colorbar(mesh, cax=ax_colorbar)
    colorbar.set_label('Near-Surface Air Temperature (K)')

    fig.tight_layout()
    
    if savefile:
        if filename is None:
            filename = f'{timestamp}.png'
        fig.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close(fig)



def add_lags(X_nolag, lags, freq):
    catDat = []
    start = 0
    for i in np.arange(lags-1,-1, -1):
        if i == 0:
            catDat.append(X_nolag[int(start*freq):].float())
        else:
            catDat.append(X_nolag[int(start*freq):int(-i*freq)].float())
        start += 1
    X = torch.cat(catDat,dim=2)
    del(catDat)
    X=X[:,0,:,:]
    return X


def normalize(X,norm_scheme= 2):
    if norm_scheme<0:
        X_nolag =X
    elif norm_scheme==0:
        mean = torch.mean(X,dim=0)
        X = (X-mean)
        std = torch.std(X)
        X_nolag = X/std
    elif norm_scheme==1:
        mean = torch.mean(X)
        X = (X-mean)
        std = torch.std(X)
        X_nolag = X/std
    elif norm_scheme==2: 
        X_nolag= (X-X.min())/2/(X.max()-X.min()) + 0.25
        # X_nolag= (X-150)/(200)
    elif norm_scheme ==3:
        mean = torch.mean(X,dim=0)
        X = (X-mean)
        std = torch.std(X)
        X = X/std
        X_nolag= (X-X.min())/2/(X.max()-X.min()) + 0.25
    else:
        mean = torch.mean(X,dim=0)
        X = (X-mean)
        std = torch.std(X)
        X = X/std
        X_nolag= (X-X.min())/(X.max()-X.min())

    return X_nolag
