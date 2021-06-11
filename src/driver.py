import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn.init as init

from read_dataset import data_from_name
from model import *
from tools import *
from train import *
from sklearn.model_selection import train_test_split

import os

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='koopmanAE', metavar='N', help='model')
#
parser.add_argument('--alpha', type=float, default='1',  help='model width')
#
parser.add_argument('--dataset', type=str, default='pendulum', metavar='N', help='dataset')
#
parser.add_argument('--theta', type=float, default=2.4,  metavar='N', help='angular displacement')
#
parser.add_argument('--noise', type=float, default=0.0,  metavar='N', help='noise level')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 0)')
#
parser.add_argument('--epochs', type=int, default=600, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=200, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='test',  help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='1',  help='balance between reconstruction and prediction loss')
#
parser.add_argument('--nu', type=float, default='1e-1',  help='tune backward loss')
#
parser.add_argument('--eta', type=float, default='0',  help='tune consistent loss')
#
parser.add_argument('--steps', type=int, default='30',  help='steps for learning forward dynamics')
#
parser.add_argument('--steps_back', type=int, default='30',  help='steps for learning backwards dynamics')
#
parser.add_argument('--bottleneck', type=int, default='6',  help='size of bottleneck layer')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 200, 400, 500], help='decrease learning rate at these epochs')
#
parser.add_argument('--lr_decay', type=float, default='0.2',  help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
#
parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')

#
parser.add_argument('--pred_steps', type=int, default='1000',  help='prediction steps')
#
parser.add_argument('--seed', type=int, default='1',  help='seed value')
#
parser.add_argument('--print_every', type=int, default='20',  help='print epochs at this rate')
#
parser.add_argument('--test_split_ratio', type=float, default='0.01',  help='test split ratio')
#
parser.add_argument('--freq', type=int, default='1',  help='frequency of dataset')
# n lags have a map to the first n-1 differences (ie derivatives)
parser.add_argument('--lags', type=int, default='1',  help='number of lags used') 
# plot projected latent dimensions
parser.add_argument('--plot_latent', type=int, default=0,  help='plot projected latent dimensions') 
#
parser.add_argument('--tropical', type=int, default=0,  help='use tropical region') 
#
parser.add_argument('--start_year', type=int, default=1900,  help='start year') 
#
parser.add_argument('--end_year', type=int, default=2010,  help='end year') 
#
parser.add_argument('--scale', type=float, default=1,  help='scale of climate data') 
#
parser.add_argument('--Burgess', type=int, default=0,  help='use Burgess architecture') 


args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)
device = get_device()



#******************************************************************************
# Create folder to save results
#******************************************************************************
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

#******************************************************************************
# Load data
#******************************************************************************
X, Xclean, m, n = data_from_name(args.dataset, noise=args.noise, theta=args.theta, start_year=args.start_year, end_year=args.end_year,
 tropical=args.tropical, freq=args.freq, scale=args.scale)
print("Data loaded")



#******************************************************************************
# Split to train and test set
#******************************************************************************
Xtrain, Xtest, Xtrain_clean, Xtest_clean = train_test_split(X, Xclean, test_size=args.test_split_ratio, random_state=42)

#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Height
#******************************************************************************
Xtrain = add_channels(Xtrain)
Xtest = add_channels(Xtest)

# transfer to tensor
Xtrain = torch.from_numpy(Xtrain).float().contiguous()
Xtest = torch.from_numpy(Xtest).float().contiguous()


#******************************************************************************
# Normalize data
#******************************************************************************

mean = torch.mean(Xtrain,dim=0) #2D mean
# mean = torch.mean(Xtrain) #1D mean
print(mean.shape)
print(mean)

Xtrain = (Xtrain-mean)
std = torch.std(Xtrain)
print(std)
Xtrain = (Xtrain)/ (std+1e-7) 

#******************************************************************************
# Create Dataloader objects
#******************************************************************************

# Add lags
if args.lags >1:
    catDat = []
    start = 0
    for i in np.arange(args.lags-1,-1, -1):
        if i == 0:
            catDat.append(Xtrain[start:].float())
        else:
            catDat.append(Xtrain[start:-i].float())
        start += 1

    Xtrain = torch.cat(catDat,dim=2)
    del(catDat)

# # Add differences
# if args.diff >0:
#     diffDat = []
#     start = 0
#     for i in np.arange(args.lags-1,-1, -1):
#         if i == 0:
#             catDat.append(Xtrain[start:].float())
#         else:
#             catDat.append(Xtrain[start:-i].float())
#         start += 1

#     Xtrain = torch.cat(diffDat,dim=1)
#     del(diffDat)


# Add steps
trainDat = []
start = 0
for i in np.arange(args.steps,-1, -1):
    if i == 0:
        trainDat.append(Xtrain[start:].float())
    else:
        trainDat.append(Xtrain[start:-i].float())
    start += 1

train_data = torch.utils.data.TensorDataset(*trainDat)
del(trainDat)

train_loader = DataLoader(dataset = train_data,
                              batch_size = args.batch,
                              shuffle = True)
print("Dataloader created")




#==============================================================================
# Model
#==============================================================================
print(Xtrain.shape)
model = koopmanAE(m, n, args.lags, args.bottleneck, args.steps, args.steps_back,
 args.alpha, args.init_scale, args.Burgess)
print('koopmanAE created')
#model = torch.nn.DataParallel(model)
model = model.to(device)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')
print(model)

#==============================================================================
# Start training
#==============================================================================
model, optimizer, epoch_hist = train(model, train_loader,
                    lr=args.lr, weight_decay=args.wd, lamb=args.lamb, num_epochs = args.epochs,
                    learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
                    b=args.bottleneck, train_data=train_data, freq=args.freq,
                    nu = args.nu, eta = args.eta, backward=args.backward, steps=args.steps, steps_back=args.steps_back, 
                    print_every=args.print_every, plot_latent =args.plot_latent)


torch.save(model.state_dict(), args.folder + '/model_backward{}'.format(args.backward)+'.pkl')
torch.save(epoch_hist, args.folder + '/model_history_backward{}'.format(args.backward)+'.pkl')




if False:
    #******************************************************************************
    # Prediction
    #******************************************************************************
    Xinput, Xtarget = Xtest[:-1], Xtest[1:]
    _, Xtarget = Xtest_clean[:-1], Xtest_clean[1:]


    snapshots_pred = []
    snapshots_truth = []


    error = []
    for i in range(30):
                error_temp = []
                init = Xinput[i].float().to(device)
                if i == 0:
                    init0 = init
                
                z = model.encoder(init) # embed data in latent space

                for j in range(args.pred_steps):
                    if isinstance(z, tuple):
                        z = model.dynamics(*z) # evolve system in time
                    else:
                        z = model.dynamics(z)
                    if isinstance(z, tuple):
                        x_pred = model.decoder(z[0])
                    else:
                        x_pred = model.decoder(z) # map back to high-dimensional space
                    target_temp = Xtarget[i+j].data.cpu().numpy().reshape(m,n)
                    error_temp.append(np.linalg.norm(x_pred.data.cpu().numpy().reshape(m,n) - target_temp) / np.linalg.norm(target_temp))
                    
                    if i == 0:
                        snapshots_pred.append(x_pred.data.cpu().numpy().reshape(m,n))
                        snapshots_truth.append(target_temp)
    
                error.append(np.asarray(error_temp))



    error = np.asarray(error)

    fig = plt.figure(figsize=(15,12))
    plt.plot(error.mean(axis=0), 'o--', lw=3, label='', color='#377eb8')
    plt.fill_between(x=range(error.shape[1]),y1=np.quantile(error, .05, axis=0), y2=np.quantile(error, .95, axis=0), color='#377eb8', alpha=0.2)

    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)

    plt.ylabel('Relative prediction error', fontsize=22)
    plt.xlabel('Time step', fontsize=22)
    plt.grid(False)
    #plt.yscale("log")
    plt.ylim([0.0,error.max()*2])
    #plt.legend(fontsize=22)
    fig.tight_layout()
    plt.savefig(args.folder +'/000prediction' +'.png')
    #plt.savefig(args.folder +'/000prediction' +'.eps')

    plt.close()

    np.save(args.folder +'/000_pred.npy', error)

    print('Average error of first pred: ', error.mean(axis=0)[0])
    print('Average error of last pred: ', error.mean(axis=0)[-1])
    print('Average error over all pred: ', np.mean(error.mean(axis=0)))


        
        
    import scipy
    save_preds = {'pred' : np.asarray(snapshots_pred), 'truth': np.asarray(snapshots_truth), 'init': np.asarray(init0.float().to(device).data.cpu().numpy().reshape(m,n))} 
    scipy.io.savemat(args.folder +'/snapshots_pred.mat', dict(save_preds), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')




    plt.close('all')
    #******************************************************************************
    # Eigenvalues
    #******************************************************************************
    model.eval()

    #if hasattr(model.dynamics, 'dynamics'):
    A =  model.dynamics.dynamics.weight.cpu().data.numpy()
    #A =  model.module.test.data.cpu().data.numpy()
    w, v = np.linalg.eig(A)
    print(np.abs(w))

    fig = plt.figure(figsize=(6.1, 6.1), facecolor="white",  edgecolor='k', dpi=150)
    plt.scatter(w.real, w.imag, c = '#dd1c77', marker = 'o', s=15*6, zorder=2, label='Eigenvalues')

    maxeig = 1.4
    plt.xlim([-maxeig, maxeig])
    plt.ylim([-maxeig, maxeig])
    plt.locator_params(axis='x',nbins=4)
    plt.locator_params(axis='y',nbins=4)

    #plt.xlabel('Real', fontsize=22)
    #plt.ylabel('Imaginary', fontsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.tick_params(axis='x', labelsize=22)
    plt.axhline(y=0,color='#636363',ls='-', lw=3, zorder=1 )
    plt.axvline(x=0,color='#636363',ls='-', lw=3, zorder=1 )

    #plt.legend(loc="upper left", fontsize=16)
    t = np.linspace(0,np.pi*2,100)
    plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c = '#636363', zorder=1 )
    plt.tight_layout()
    plt.show()
    plt.savefig(args.folder +'/000eigs' +'.png')
    plt.savefig(args.folder +'/000eigs' +'.eps')
    plt.close()

    plt.close('all')
