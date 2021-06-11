import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch

from skimage.transform import rescale, resize
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy
# Cartopy (https://scitools.org.uk/cartopy/docs/latest/installing.html) has a dependencies on
    # GEOS (https://trac.osgeo.org/geos/)
    # PROJ (https://proj4.org/)
    
    # Also installed libgeos++-dev after the following error
    # ImportError: libproj.so.19: cannot open shared object file: No such file or directory

import netCDF4
import pickle

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=2.4, start_year=1870, end_year= 2100, tropical=False, freq=30, scale=1.0):
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise, theta)
    if name == "climate":
        return climate(start_year, end_year, tropical, freq, scale) 
    if name == "climate_noPC":
        return climate_noPC()
    else:
        raise ValueError('dataset {} not recognized'.format(name))


# def rescale(Xsmall, Xsmall_test):
#     #******************************************************************************
#     # Rescale data
#     #******************************************************************************
#     Xmin = Xsmall.min()
#     Xmax = Xsmall.max()
    
#     Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
#     Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

#     return Xsmall, Xsmall_test


def pendulum_lin(noise):
    
    np.random.seed(0)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    
    X = sol(anal_ts, 0.8)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
 
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate   
    Xclean = Xclean.T.dot(Q.T)     
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, Xclean, 64, 1






def pendulum(noise, theta=2.4, output_dim=2, rotate= False):
    
    np.random.seed(1)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    X = sol(anal_ts, theta)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    if rotate:
        Q = np.random.standard_normal((output_dim,2))
        Q,_ = np.linalg.qr(Q)
    else:
        Q = np.eye(2)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X, Xclean, output_dim, 1


#Use 1983 -> 2006 for Eniko's data 

def climate(start_year=1870, end_year= 2100, tropical=False, freq=30, scale=1.0):
    
    climate_file = netCDF4.Dataset('../../data/tas_day_MPI-ESM-MR_rcp85_r1i1p1_g025-002.nc')
    initial_year = 1870

    start_index = int((start_year - initial_year)*365.25)
    end_index = int((end_year - initial_year)*365.25)

    if tropical:
        X = climate_file.variables['tas'][start_index:end_index:freq,26:46].data
        m = 20
        n = 144

    else:
        X = climate_file.variables['tas'][start_index:end_index:freq].data
        m = 72
        n = 144

    if scale!=1.0:
        tmp_list =[]
        for x in X:
           tmp_list.append(rescale(x, scale))
        X = np.array(tmp_list, dtype=float)
        m, n = int(m*scale), int(n*scale)

    print(type(X))
    print(X.shape)
    Xclean = X

    return X, Xclean, m, n
    

def climate_noPC(scale=1):
    return pickle.load( open( f"../../data/noPC_5days_scale{scale}.p", "wb" ) )