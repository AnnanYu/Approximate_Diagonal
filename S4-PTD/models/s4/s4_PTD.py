import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import rearrange, repeat
import scipy.io as mlio
import os

dirname = os.path.dirname(__file__)

from src.models.nn import DropoutNd

class perturbation(nn.Module):
    def __init__(self,N,gamma):
        super().__init__()
        self.gamma = gamma
        self.N = N
        self.P = nn.Parameter(torch.view_as_real(torch.rand([N,N], dtype=torch.complex64) * gamma * 0.5))
       
        A = np.zeros([N,N], dtype='cfloat')
        for i in range(N):
            for j in range(i):
                A[i,j] = math.sqrt(2*i+1) * math.sqrt(2*j+1)
            A[i,i] = i+1
        self.A = torch.tensor(A, requires_grad=False)
       
    def forward(self):
        P = torch.view_as_complex(self.P)
        _, V = torch.linalg.eig(self.A + P)
        return torch.linalg.cond(V), torch.linalg.norm(P)


class perturbation_trainer():
    def __init__(self,N,lamb,lr=0.1,epochs=3000):
        self.N = N
        self.lamb = lamb
        self.lr = lr
        self.epochs = epochs

    def train_P(self):
        model = perturbation(self.N,0.2)
        opt = optim.Adam(model.parameters(),lr=self.lr,weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=3000, gamma=0.99)
        for i in range(self.epochs):
            out,out2= model()
            loss = nn.functional.mse_loss(out, torch.tensor([0],dtype=torch.float64)) + self.lamb * out2
            if i % (self.epochs // 100) == 0:
                print('Epoch: ', i, ' Condition number: ', out.item(), ' Norm: ', out2.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
        return torch.view_as_complex(model.P).detach().numpy(), out.item(), out2.item()


class S4_PTD_Kernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, delta=25, filename='', **kernel_args):
        super().__init__()
        # Generate dt
        H = d_model
        Nh = N // 2
        if 1:
            B = np.ones([Nh,H], dtype='cfloat')
            for i in range(Nh):
                B[i,:] *= math.sqrt(2*i+1)
            dic = mlio.loadmat(filename)
            lam = -dic['Lam']
            V = dic['V']
            print('Condition number of V: ',str(np.linalg.cond(V)))
            A = np.reshape(np.diag(lam), [Nh])
            B = np.linalg.solve(V,B)
            C = np.random.randn(H, Nh)
            C = np.matmul(C,V)
        else:
            A = np.ones([Nh,Nh], dtype='cfloat')
            B = np.ones([Nh,H], dtype='cfloat')
        
            for i in range(Nh):
                B[i,:] *= math.sqrt(2*i+1)
                for j in range(i):
                    A[i,j] = math.sqrt(i+1/2) * math.sqrt(j+1/2)
                for j in range(i,Nh):
                    A[i,j] = -math.sqrt(i+1/2) * math.sqrt(j+1/2)
                A[i,i] = 1/2
            lam = np.linalg.eig(A)
            V = lam[1]
            B = np.linalg.solve(V,B)
            A = np.reshape(lam[0], [Nh])
            C = np.random.randn(H, Nh)
            C = np.matmul(C,V)

        log_dt = torch.rand(H) * (
            math.log(0.1) - math.log(0.001)
        ) + math.log(0.001)

        C = torch.view_as_real(torch.tensor(C, dtype=torch.cfloat))
        self.register("log_dt", log_dt, 0)

        log_A_real = torch.log(repeat(torch.tensor(A.real), 'n -> h n', h=H))
        A_imag = repeat(torch.tensor(A.imag), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, 0)
        self.register("A_imag", A_imag, 0)
        self.register("B",torch.view_as_real(torch.tensor(B.T, dtype=torch.cfloat)),0)
        self.register("C",C,0.004)

    def getparams(self):
        return self.cond, self.norm

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        B = torch.view_as_complex(self.B) # (H N)
        A = -torch.exp(self.log_A_real) - 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = B * C * (torch.exp(dtA)-1.) / A

        K = torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real * 2

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.05}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4_PTD(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, delta=75, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.delta = delta

        # SSM Kernel
        filename = os.path.join(dirname,'../../ptbHiPPO32.mat')
        self.kernel = S4_PTD_Kernel(self.h, self.n, self.delta, filename, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def getparams(self):
        return self.cond, self.norm

    def forward(self, u, noise, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)
        k = nn.functional.pad(k,(0,L),'constant',0)
        u = nn.functional.pad(u,(0,L),'constant',0)
        noise = nn.functional.pad(noise,(0,L),'constant',0)
        k_f = torch.fft.fft(k) # (H L)
        u_f = torch.fft.fft(u+noise) # (B H L)
        y = torch.fft.ifft(u_f*k_f).real # (B H L)
        u = u[:,:,0:L]
        y = y[:,:,0:L]

        y = self.dropout(self.activation(y))
       
        y = y.float()
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified