import torch
from torch import nn
import numpy as np

from tools import *



def train(model, train_loader, lr, weight_decay, 
          lamb, num_epochs, learning_rate_change, epoch_update, b, train_data, freq,
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, print_every=20,
          plot_latent=0):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = get_device()
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.5, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                     
        

    criterion = nn.MSELoss().to(device)


    epoch_hist = []
    loss_hist = []
    epoch_loss = []
                            
    for epoch in range(num_epochs):
        for batch_idx, data_list in enumerate(train_loader):
            model.train()
            out, out_back = model(data_list[0].to(device), mode='forward')

            loss_fwd = 0.0
            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], data_list[k+1].to(device))
                else:
                    loss_fwd += criterion(out[k], data_list[k+1].to(device))

            
            multiplicative_factor= 1 if steps<1 else steps
            loss_identity = criterion(out[-1], data_list[0].to(device)) * multiplicative_factor

            loss_bwd = 0.0
            loss_consist = 0.0

            if backward == 1:
                out, out_back = model(data_list[-1].to(device), mode='backward')
   

                for k in range(steps_back):
                    
                    if k == 0:
                        loss_bwd = criterion(out_back[k], data_list[::-1][k+1].to(device))
                    else:
                        loss_bwd += criterion(out_back[k], data_list[::-1][k+1].to(device))
                        
                               
                A = model.dynamics.dynamics.weight
                B = model.backdynamics.dynamics.weight

                K = A.shape[-1]

                for k in range(1,K+1):
                    As1 = A[:,:k]
                    Bs1 = B[:k,:]
                    As2 = A[:k,:]
                    Bs2 = B[:,:k]

                    Ik = torch.eye(k).float().to(device)

                    if k == 1:
                        loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                    else:
                        loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)

    
                
                
                
#                Ik = torch.eye(K).float().to(device)
#                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
#                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
#   
                                        
                
    
            loss = loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist 
            # loss =  loss_identity  #Youssef

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip #Youssef
            optimizer.step()           

        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)
        
        
        if (epoch) % print_every == 0:
            print('********** Epoch %s **********' %(epoch+1))
            
            print("loss identity (actual): ", loss_identity.item()/multiplicative_factor)
            print("loss identity (scaled): ", loss_identity.item())
            if backward == 1:
                print("loss backward: ", loss_bwd.item())
                print("loss consistent: ", loss_consist.item())
            if steps > 0: print("loss forward: ", loss_fwd.item())
            print("loss sum: ", loss.item())


            if hasattr(model.dynamics, 'dynamics'):
                w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
                print(np.abs(w))
            
            if plot_latent == 1:
                with torch.no_grad():
                    embedding= (model.encoder(train_data[:][0]))

                embedding = np.reshape(embedding, newshape= (-1,b))

                A =  model.dynamics.dynamics.weight.cpu().data.numpy()
                w, v = np.linalg.eig(A)

                projected = np.dot(embedding, v)
                no_years= 1
                to_plot= projected[:int(no_years*365/freq)]
                fig, ax = plt.subplots(nrows=int(b/4), ncols=4, figsize=(b,16))
                for i in range(b):
                #     print(i%3, int(i/4))
                    ax[int(i/4), i%4].plot(to_plot[:,i], 'k', linewidth=.2)
                plt.show(block=True)

                



    if backward == 1:
        loss_consist = loss_consist.item()
                
    # loss_identity.item()]          
    # loss_fwd.item()
    return model, optimizer, [epoch_hist, loss_consist] 
  



#******************************************************************************
# model with control
#******************************************************************************

def trainY(model, train_loader, lr, weight_decay, 
          lamb, num_epochs, learning_rate_change, epoch_update,
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, print_every=20, fixed_A =1):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = get_device()
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.5, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                     
        

    criterion = nn.MSELoss().to(device)


    epoch_hist = []
    loss_hist = []
    epoch_loss = []
                            
    for epoch in range(num_epochs):
        for batch_idx, data_list in enumerate(train_loader):
            model.train()

            list_x_c = []
            for k in range(steps+1):
                x_l, x_c = model.encoder(data_list[k].to(device))
                list_x_c.append(x_c)
            


            out, out_back = model(data_list[0].to(device), list_x_c, mode='forward')

            loss_fwd = 0.0
            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k], data_list[k+1].to(device))
                else:
                    loss_fwd += criterion(out[k], data_list[k+1].to(device))

            
            multiplicative_factor= 1 if steps<1 else steps
            loss_identity = criterion(out[-1], data_list[0].to(device)) * multiplicative_factor

            loss_bwd = 0.0
            loss_consist = 0.0

            if backward == 1:
                out, out_back = model(data_list[-1].to(device), list_x_c, mode='backward')

                for k in range(steps_back):
                    if k == 0:
                        #MSE(v_(t+s-1 -k),v_(t+s-(k+1)))
                        loss_bwd = criterion(out_back[k], data_list[::-1][k+1].to(device))
                    else:
                        loss_bwd += criterion(out_back[k], data_list[::-1][k+1].to(device))
                        
                               
                A = model.dynamics.dynamics_l.weight
                B = model.backdynamics.dynamics_l.weight

                K = A.shape[-1]

                for k in range(1,K+1):
                    As1 = A[:,:k]
                    Bs1 = B[:k,:]
                    As2 = A[:k,:]
                    Bs2 = B[:,:k]

                    Ik = torch.eye(k).float().to(device)

                    if k == 1:
                        loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                    else:
                        loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                         torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)

    
                
                
                
#                Ik = torch.eye(K).float().to(device)
#                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
#                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
#   
                                        
                
    
            loss = loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist 
            # loss =  loss_identity  #Youssef

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip #Youssef
            optimizer.step()           

        # schedule learning rate decay    
        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)
        
        
        if (epoch) % print_every == 0:
            print('********** Epoch %s **********' %(epoch+1))
            
            print("loss identity (actual): ", loss_identity.item()/multiplicative_factor)
            print("loss identity (scaled): ", loss_identity.item())
            if backward == 1:
                print("loss backward: ", loss_bwd.item())
                print("loss consistent: ", loss_consist.item())
            if steps > 0: print("loss forward: ", loss_fwd.item())
            print("loss sum: ", loss.item())


            if hasattr(model.dynamics, 'dynamics'):
                w, _ = np.linalg.eig(model.dynamics_l.dynamics.weight.data.cpu().numpy())
                print(np.abs(w))

                



    if backward == 1:
        loss_consist = loss_consist.item()
                
    # loss_identity.item()]          
    # loss_fwd.item()
    return model, optimizer, [epoch_hist, loss_consist] 