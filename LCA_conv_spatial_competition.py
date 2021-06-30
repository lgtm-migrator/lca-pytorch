import os 
from random import shuffle

import cv2
import matplotlib.pyplot as plt 
import numpy as np
from progressbar import ProgressBar
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid


class LCADLConvSpatialComp:
    def __init__(self, n_neurons, thresh = 0.1, tau = 1000, eta = 0.01, 
                 lca_iters = 2000, update_steps = 1000, batch_size = 256,
                 kernel_size = 11, pad = 'same', stride = 1, device = None,
                 dtype = torch.float32):
        ''' Performs sparse dictionary learning via LCA (Rozell et al. 2008) '''

        self.m = n_neurons 
        self.thresh = thresh 
        self.tau = tau 
        self.charge_rate = 1.0 / self.tau
        self.eta = eta 
        self.lca_iters = lca_iters 
        self.update_steps = update_steps
        self.batch_size = batch_size
        self.kernel_size = kernel_size  
        self.pad = pad 
        self.stride = stride 
        self.device = device 
        self.dtype = dtype

        # We will store l2 reconstruction values in these to plot 
        self.error_mean, self.error_se = [], []

    def create_feat_grid(self):
        ''' Create feature grid used for determining competition over space '''
        self.grid_h = self.in_h if self.pad == 'valid' else self.in_h + (self.kernel_size - 1)
        self.grid_w = self.in_w if self.pad == 'valid' else self.in_w + (self.kernel_size - 1)

        if self.pad == 'same':
            self.n_spatial_h = int(np.ceil(self.in_h / self.stride))
            self.n_spatial_w = int(np.ceil(self.in_w / self.stride))
        elif self.pad == 'valid':
            self.n_spatial_h = int(np.ceil((self.in_h - self.kernel_size + 1) / self.stride))
            self.n_spatial_w = int(np.ceil((self.in_w - self.kernel_size + 1) / self.stride))

        self.D = torch.zeros(
            self.m, 
            self.n_spatial_h,
            self.n_spatial_w,
            self.in_c, 
            self.grid_h, 
            self.grid_w,
            device = self.device,
            dtype = self.dtype
        )

        for feat_num in range(self.m):
            kernel = torch.randn(
                self.in_c, 
                self.kernel_size, 
                self.kernel_size,
                device = self.device,
                dtype = self.dtype
            )

            for ih, h in enumerate(range(0, self.grid_h - self.kernel_size + 1, self.stride)):
                for iw, w in enumerate(range(0, self.grid_w - self.kernel_size + 1, self.stride)):
                    self.D[feat_num, ih, iw, :, h:h+self.kernel_size, w:w+self.kernel_size] = kernel

        self.D = self.D.reshape(self.m * self.n_spatial_h * self.n_spatial_w, -1)
        self.feat_mask = (self.D != 0).cpu().bool()
        self.normalize_D()

    def get_device(self, x):
        ''' Gets the device (GPU) x is on and returns None if on CPU ''' 

        return x.device.index if x.is_cuda else None

    def encode(self, x):
        ''' Computes sparse code given data vector x and dictionary matrix D '''

        # initialize membrane potentials
        u_t = torch.zeros(self.D.shape[0], x.shape[1], device = self.get_device(x), dtype = self.dtype)

        # compute inhibition matrix
        G = torch.mm(self.D, self.D.T)
        G[list(range(G.shape[0])), list(range(G.shape[1]))] = 0.0

        # compute driving inputs
        b_t = torch.mm(self.D, x) 

        for _ in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += self.charge_rate * (b_t - u_t - torch.matmul(G, a_t))

        return a_t 

    def update_D(self, x, a):
        ''' Updates the dictionary based on the reconstruction error. '''

        recon = self.compute_recon(a)
        error = x - recon
        self.track_l2_recon_error(error)
        update = torch.matmul(error, a.T)
        update = update.T.flatten()[self.feat_mask.flatten()]
        update = update.reshape(
            self.m,
            self.n_spatial_h,
            self.n_spatial_w,
            self.in_c,
            self.kernel_size,
            self.kernel_size
        )
        update = update.\
            mean(dim = (1, 2)).\
            unsqueeze(1).\
            unsqueeze(1).\
            repeat_interleave(self.n_spatial_h, dim = 1).\
            repeat_interleave(self.n_spatial_w, dim = 2).\
            reshape(self.m * self.n_spatial_h * self.n_spatial_w, -1)
        self.D.flatten()[self.feat_mask.flatten()] += update.flatten() * self.eta
        self.normalize_D()

    def normalize_D(self):
        ''' Normalizes features such at each one has unit norm '''

        scale = (self.D.norm(p = 2, dim = -1, keepdim = True) + 1e-12)
        self.D /= scale

    def soft_threshold(self, x):
        ''' Soft threshold '''
        return torch.maximum(x - self.thresh, torch.zeros_like(x))# - torch.maximum(-x - self.thresh, torch.zeros_like(x))

    def compute_recon(self, a):
        ''' Computes reconstruction given code '''
        return torch.mm(self.D.T, a)

    def track_l2_recon_error(self, error):
        ''' Keeps track of the reconstruction error over training '''
        l2_error = error.norm(p = 2, dim = 0)
        self.error_mean.append(torch.mean(l2_error).item())
        self.error_se.append(torch.std(l2_error).item() / np.sqrt(self.batch_size))

    def get_batch(self, X):
        return X[np.random.choice(X.shape[0], self.batch_size, replace = False)]

    def pad_input(self, x):
        pad_before = int(np.floor((self.kernel_size - 1) / 2))
        pad_after = int(np.ceil((self.kernel_size - 1) / 2))
        return F.pad(x, (pad_before, pad_after, pad_before, pad_after), mode = 'replicate')


    def preprocess_batch(self, x, eps = 1e-12):
        ''' Scales the values of each patch to [0, 1] and then transforms each patch to have mean 0 '''
        x = x.type(self.dtype).to(self.device)
        if self.pad == 'same': x = self.pad_input(x)
        minx = x.reshape(x.shape[0], -1).min(dim = -1)[0][:, None, None, None]
        maxx = x.reshape(x.shape[0], -1).max(dim = -1)[0][:, None, None, None]
        x = (x - minx) / (maxx - minx + eps)
        meanx = x.reshape(x.shape[0], -1).mean(dim = -1)[:, None, None, None]
        x -= meanx
        return x.reshape(x.shape[0], -1).T

    def run_model(self, X):
        self.in_c, self.in_h, self.in_w = X.shape[1:]
        self.create_feat_grid()

        for step in ProgressBar()(range(self.update_steps)):
            batch = self.get_batch(X)
            batch = self.preprocess_batch(batch)
            a = self.encode(batch)
            self.update_D(batch, a)


# loading in random images from CIFAR dataset
imgs = CIFAR10(root = 'cifar/', download = True).data.astype(np.float32)
imgs_gray = torch.from_numpy(np.mean(imgs, -1)).unsqueeze(1)


# run the model
model = LCADLConvSpatialComp(
    64, 
    lca_iters = 2000, 
    update_steps = 2000, 
    thresh = 0.1, 
    batch_size = 128,
    tau = 50, 
    stride = 2,
    pad = 'valid',
    kernel_size = 7,
    dtype = torch.float16,
    device = 1,
    eta = 0.1
)
model.run_model(imgs_gray[:-1000])


# plot error 
plt.errorbar(list(range(len(model.error_mean))), model.error_mean, yerr = model.error_se)
plt.ylabel('Reconstruction Error +/- SE')
plt.xlabel('Training Iteration')
plt.show()

# plot dictionary
feats = model.D.flatten()[model.feat_mask.flatten()]
feats = feats.reshape(
    model.m, 
    model.n_spatial_h, 
    model.n_spatial_w, 
    model.in_c, 
    model.kernel_size, 
    model.kernel_size
)
grid = make_grid(feats[:, 0, 0], nrow = int(np.sqrt(model.m)))
plt.imshow(grid[0].float().cpu().numpy(), cmap = 'gray')
plt.show()


# reconstruct new images
inputs = model.preprocess_batch(imgs_gray[-1000:])
a = model.encode(inputs)
recon = model.compute_recon(a) 


# plot inputs and recons
fig = plt.figure()
sub1 = fig.add_subplot(121)
sub2 = fig.add_subplot(122)
in_grid = make_grid(inputs.T.reshape(1000, model.in_c, model.in_h, model.in_w), nrow = 30)
rec_grid = make_grid(recon.T.reshape(1000, model.in_c, model.in_h, model.in_w), nrow = 30)
sub1.imshow(in_grid.float().cpu().numpy()[0], cmap = 'gray')
sub2.imshow(rec_grid.float().cpu().numpy()[0], cmap = 'gray')
sub1.set_title('Input')
sub2.set_title('Recon')
plt.show()