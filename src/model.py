from modelsV.decoders import *
from modelsV.encoders import *
from torch import nn
import torch
import math

def gaussian_init_(n_units1, n_units2=0, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units1]))
    if n_units2==0:
        Omega = sampler.sample((n_units1, n_units1))[..., 0]  
    else:
        Omega = sampler.sample((n_units1, n_units2))[..., 0]  
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, lags, b, ALPHA = 1):
        super().__init__()
        self.N = m * n * lags
        self.act = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(self.N, int(128*ALPHA))
        self.fc2 = nn.Linear(int(128*ALPHA), int(64*ALPHA))
        self.fc3 = nn.Linear(int(64*ALPHA), int(32*ALPHA))
        self.fc4 = nn.Linear(int(32*ALPHA), int(16*ALPHA))
        self.fc5 = nn.Linear(int(16*ALPHA), b)
        # self.batch2 = nn.BatchNorm1d(1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = self.batch1(x)
        x = x.view(-1, 1, self.N)
        x = self.act(self.fc1(x)) 
        x = self.act(self.fc2(x)) 
        x = self.act(self.fc3(x)) 
        x = self.act(self.fc4(x))
        # x = self.act(self.fc5(x))
        x = self.fc5(x)
        # x = self.batch2(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, lags, b, ALPHA = 1):
        super().__init__()

        self.m = m
        self.n = n
        self.b = b
        self.lags= lags

        self.act = nn.ReLU()

        self.fc1 = nn.Linear(b, int(16*ALPHA))
        self.fc2 = nn.Linear(int(16*ALPHA), int(32*ALPHA))
        self.fc3 = nn.Linear(int(32*ALPHA), int(64*ALPHA))
        self.fc4 = nn.Linear(int(64*ALPHA), int(128*ALPHA))
        self.fc5 = nn.Linear(int(128*ALPHA), m*n*self.lags)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)     

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.act (self.fc1(x)) 
        x = self.act (self.fc2(x)) 
        x = self.act (self.fc3(x)) 
        x = self.act (self.fc4(x)) 
        x = self.fc5(x)
        x = x.view(-1, 1, self.m*self.lags, self.n)
        return x
         


class encoderNetY(nn.Module):
    def __init__(self, m, n, lags, b_l, b_c, ALPHA = 1):
        super().__init__()
        self.N = m * n * lags
        self.act = nn.ReLU()
        self.act2 = nn.Tanh()


        self.batch1 = nn.BatchNorm2d(1)
        b=b_l +b_c
        self.fc1 = nn.Linear(self.N, int(128*ALPHA))
        self.fc2 = nn.Linear(int(128*ALPHA), int(64*ALPHA))
        self.fc3 = nn.Linear(int(64*ALPHA), int(32*ALPHA))
        self.fc4 = nn.Linear(int(32*ALPHA), int(16*ALPHA))
        self.fc5 = nn.Linear(int(16*ALPHA), b)
        self.fc5l = nn.Linear(b, b_l)
        self.fc5c = nn.Linear(b, b_c)
        # self.batch2 = nn.BatchNorm1d(1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = self.batch1(x)
        x = x.view(-1, 1, self.N)
        x = self.act(self.fc1(x)) 
        x = self.act(self.fc2(x)) 
        x = self.act(self.fc3(x)) 
        x = self.act2(self.fc4(x))
        # x = self.act(self.fc5(x))
        x = self.fc5(x)
        x_l, x_c = self.fc5l(x), self.fc5c(x)
        # x = self.batch2(x)
        
        return x_l, x_c


class decoderNetY(nn.Module):
    def __init__(self, m, n, lags, b_l, ALPHA = 1):
        super().__init__()

        self.m = m
        self.n = n
        self.b_l = b_l
        self.lags= lags

        self.act = nn.ReLU()
        self.act2 = nn.Tanh()

        self.fc1 = nn.Linear(b_l, int(16*ALPHA))
        self.fc2 = nn.Linear(int(16*ALPHA), int(32*ALPHA))
        self.fc3 = nn.Linear(int(32*ALPHA), int(64*ALPHA))
        self.fc4 = nn.Linear(int(64*ALPHA), int(128*ALPHA))
        self.fc5 = nn.Linear(int(128*ALPHA), m*n*self.lags)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)    

    def forward(self, x):
        x = x.view(-1, 1, self.b_l)
        x = self.act (self.fc1(x)) 
        x = self.act (self.fc2(x)) 
        x = self.act (self.fc3(x)) 
        x = self.act2 (self.fc4(x)) 
        x = self.fc5(x)
        x = x.view(-1, 1, self.m*self.lags, self.n)
        return x



class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super().__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x



class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super().__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x



class koopmanAE(nn.Module):
    def __init__(self, m, n, lags, b, steps=1, steps_back=1, alpha = 1, init_scale=1, Burgess = True, freq=5):
        super().__init__()
        self.steps = steps
        self.steps_back = steps_back

        if not Burgess:
            self.encoder = encoderNet(m, n, lags, b, ALPHA = alpha)
            self.decoder = decoderNet(m, n, lags, b, ALPHA = alpha)
        else:
            img_size = (1,m*lags,n) 
            self.encoder = EncoderBurgess(img_size= img_size, latent_dim=b)
            self.decoder = DecoderBurgess(img_size= img_size, latent_dim=b)

        self.dynamics = dynamics(b, init_scale,freq)
        self.backdynamics = dynamics_back(b, self.dynamics)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back

# model with control



class dynamicsY(nn.Module):
    def __init__(self, b_l, b_c, init_scale, fixed_A=1, freq=5):
        super().__init__()

        self.b_c =b_c
        # linear part
        if fixed_A:
            self.dynamics_l = create_rotation(freq=freq, num_harmonics=b_l/2)
            self.dynamics_l.requires_grad_(False)
        else:    
            self.dynamics_l = nn.Linear(b_l, b_l, bias=False)
            self.dynamics_l.weight.data = gaussian_init_(b_l, std=1)           
            U, _, V = torch.svd(self.dynamics_l.weight.data)
            self.dynamics_l.weight.data = torch.mm(U, V.t()) * init_scale

        # control part
        if b_c>0:
            self.dynamics_c = nn.Linear(b_c, b_l, bias=False)
            self.dynamics_c.weight.data = gaussian_init_(b_c, b_l, std=1)           
            U, _, V = torch.svd(self.dynamics_c.weight.data)
            self.dynamics_c.weight.data = torch.mm(V,U.t()) * init_scale
        
    def forward(self, x_l, x_c):
        if self.b_c>0:
            x_l = self.dynamics_l(x_l) + self.dynamics_c(x_c)
        else: 
            x_l = self.dynamics_l(x_l)

        return x_l

    def forward_c(self, x_c):
        x_c = self.dynamics_c(x_c) 
        return x_c




class dynamics_backY(nn.Module):
    def __init__(self, b_l, b_c, omega, fixed_A=1):
        super().__init__()

        # linear part
        self.dynamics_l = nn.Linear(b_l, b_l, bias=False)
        self.dynamics_l.weight.data = torch.pinverse(omega.dynamics_l.weight.data.t()) 
        if fixed_A:
            self.dynamics_l.requires_grad_(False)

        
    def forward(self, x_l, x_c):
        x_l = self.dynamics_l(x_l -x_c) 
        return x_l
     


class koopmanAEY(nn.Module):
    def __init__(self, m, n, lags, b_l, b_c, steps=1, steps_back=1, alpha = 1, init_scale=1, fixed_A=1):
        super().__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.encoder = encoderNetY(m, n, lags, b_l, b_c, ALPHA = alpha)
        self.decoder = decoderNetY(m, n, lags, b_l, ALPHA = alpha)

        self.dynamics = dynamicsY(b_l, b_c, init_scale,fixed_A=fixed_A)
        self.backdynamics = dynamics_backY(b_l, b_c, self.dynamics,fixed_A=fixed_A)


    def forward(self, x, list_x_c=None, mode='forward'):
        out = []
        out_back = []
        z, x_c = self.encoder(x.contiguous())
        x_l = z.contiguous()

        
        if mode == 'forward':
            for i in range(self.steps):
                x_c = list_x_c[i]
                x_l = self.dynamics(x_l, x_c)
                out.append(self.decoder(x_l))

            out.append(self.decoder(z)) 
            return out, out_back    

        if mode == 'backward':
            list_x_c = list_x_c[::-1]
            for i in range(self.steps):
                x_c = self.dynamics.forward_c(list_x_c[i+1])
                x_l = self.backdynamics(x_l, x_c)
                out_back.append(self.decoder(x_l))

            out_back.append(self.decoder(z)) 
            return out, out_back 
        
        if mode == 'reconstruct':
            return self.decoder(x_l)


def create_rotation2(freq=10, harmonics=1 ):
    theta = 365.25/math.pi/2*harmonics*freq
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor(((c, -s), (s, c)))
    return R

def create_rotation(freq=10,num_harmonics=12):
    R = nn.Linear(int(num_harmonics*2), int(num_harmonics*2), bias=False)
    R.weight.data = torch.zeros_like(R.weight.data) 
    for i in range(int(num_harmonics)):
        R.weight.data[2*i : 2*i+2, 2*i : 2*i+2] = create_rotation2(freq,harmonics=i)
    return R