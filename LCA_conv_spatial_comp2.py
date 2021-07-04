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
    def __init__(self, n_neurons, thresh = 0.1, tau = 50, eta = 0.01, 
                 lca_iters = 2000, update_steps = 1000, batch_size = 256,
                 kernel_size = 11, stride = 1, pad = 'same', device = None,
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
        self.stride = stride 
        self.pad = pad 
        self.device = device 
        self.dtype = dtype

        # We will store l2 reconstruction values in these to plot 
        self.error_mean, self.error_se = [], []

    def create_weight_mat(self):
        self.D = torch.randn(
            self.m,
            self.in_c,
            self.kernel_size,
            self.kernel_size,
            device = self.device,
            dtype = self.dtype
        )
        self.normalize_D()

    def create_feat_grid(self):
        self.grid_size = self.kernel_size * 3 - 2
        self.n_spatial = int(np.ceil((self.grid_size - self.kernel_size + 1) / self.stride))
        self.feat_grid = torch.zeros(
            self.m,
            self.n_spatial,
            self.n_spatial,
            self.in_c,
            self.grid_size,
            self.grid_size,
            device = self.device,
            dtype = self.dtype
        )
        
        for feat_num in range(self.m):
            for ih, h in enumerate(range(0, self.grid_size - self.kernel_size + 1, self.stride)):
                for iw, w in enumerate(range(0, self.grid_size - self.kernel_size + 1, self.stride)):
                    self.feat_grid[feat_num, ih, iw, :, h:h+self.kernel_size, w:w+self.kernel_size] = 1.0

        self.feat_mask = (self.feat_grid != 0).bool().cpu()

    def compute_lateral_connectivity(self):
        self.feat_grid.flatten()[self.feat_mask.flatten()] = self.D.\
            unsqueeze(1).\
            unsqueeze(1).\
            repeat_interleave(self.n_spatial, dim = 1).\
            repeat_interleave(self.n_spatial, dim = 2).\
            flatten()
        G = torch.mm(
            self.feat_grid.reshape(self.m * self.n_spatial * self.n_spatial, -1),
            self.feat_grid.reshape(self.m * self.n_spatial * self.n_spatial, -1).T
        )
        G[list(range(G.shape[0])), list(range(G.shape[1]))] = 0.0
        G = G.reshape(
            self.m,
            self.n_spatial,
            self.n_spatial,
            self.m,
            self.n_spatial,
            self.n_spatial
        )
        return G[:, self.n_spatial // 2, self.n_spatial // 2]

    def perform_lateral_competition(self, a, G):
        return F.conv2d(
            a,
            G,
            stride = 1,
            padding = (self.n_spatial - 1) // 2
        )

    def get_device(self, x):
        ''' Gets the device (GPU) x is on and returns None if on CPU ''' 

        return x.device.index if x.is_cuda else None

    def encode(self, x):
        ''' Computes sparse code given data vector x and dictionary matrix D '''

        # input drive
        b_t = F.conv2d(
            x,
            self.D,
            stride = self.stride,
            padding = (self.kernel_size - 1) // 2 if self.pad == 'same' else 0
        )

        # initialize membrane potentials
        u_t = torch.zeros_like(b_t)

        # compute inhibition matrix
        G = self.compute_lateral_connectivity()

        for _ in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += self.charge_rate * (b_t - u_t - self.perform_lateral_competition(a_t, G))

        return a_t 

    def update_D(self, x, a):
        ''' Updates the dictionary based on the reconstruction error. '''

        recon = self.compute_recon(a)
        error = x - recon
        self.track_l2_recon_error(error)
        error = F.unfold(
            error, 
            self.kernel_size, 
            padding = (self.kernel_size - 1) // 2, 
            stride = self.stride
        )
        error = error.reshape(
            error.shape[0], 
            self.in_c, 
            self.kernel_size, 
            self.kernel_size, 
            a.shape[-2], 
            a.shape[-1]
        )
        update = torch.tensordot(a, error, dims = ([0, 2, 3], [0, -2, -1]))
        self.D += update * self.eta
        self.normalize_D()

    def normalize_D(self):
        ''' Normalizes features such at each one has unit norm '''

        scale = (self.D.norm(p = 2, dim = (1, 2, 3), keepdim = True) + 1e-12)
        self.D /= scale

    def soft_threshold(self, x):
        ''' Soft threshold '''
        return torch.maximum(x - self.thresh, torch.zeros_like(x))# - torch.maximum(-x - self.thresh, torch.zeros_like(x))

    def compute_recon(self, a):
        ''' Computes reconstruction given code '''
        return F.conv_transpose2d(
            a, 
            self.D,
            stride = self.stride,
            padding = (self.kernel_size - 1) // 2 if self.pad == 'same' else 0,
            output_padding = 0 if self.kernel_size % 2 == 0 else 1
        )

    def track_l2_recon_error(self, error):
        ''' Keeps track of the reconstruction error over training '''
        l2_error = error.norm(p = 2, dim = (1, 2, 3))
        self.error_mean.append(torch.mean(l2_error).item())
        self.error_se.append(torch.std(l2_error).item() / np.sqrt(self.batch_size))

    def preprocess_inputs(self, x, eps = 1e-12):
        ''' Scales the values of each patch to [0, 1] and then transforms each patch to have mean 0 '''
        x = x.type(self.dtype).to(self.device)
        minx = x.reshape(x.shape[0], -1).min(dim = -1, keepdim = True)[0][..., None, None]
        maxx = x.reshape(x.shape[0], -1).max(dim = -1, keepdim = True)[0][..., None, None]
        x = (x - minx) / (maxx - minx + eps)
        x -= x.mean(dim = (1, 2, 3), keepdim = True)
        return x

    def run_model(self, X):
        self.in_c, self.in_h, self.in_w = X.shape[1:]
        self.create_weight_mat()
        self.create_feat_grid()

        for step in ProgressBar()(range(self.update_steps)):
            batch = X[np.random.choice(X.shape[0], self.batch_size, replace = False)]
            batch = self.preprocess_inputs(batch)
            a = self.encode(batch)
            self.update_D(batch, a)


# loading in random images from CIFAR dataset
imgs = CIFAR10(root = 'cifar/', download = True).data.astype(np.float32)
imgs_gray = torch.from_numpy(np.mean(imgs, -1)).unsqueeze(1)


# run the model
model = LCADLConvSpatialComp(
    128, 
    lca_iters = 2000, 
    update_steps = 2000, 
    thresh = 0.1, 
    batch_size = 256,
    device = 1,
    stride = 2,
    kernel_size = 11,
    dtype = torch.float16
)
model.run_model(imgs_gray[:-1000])


# plot error 
plt.errorbar(list(range(len(model.error_mean))), model.error_mean, yerr = model.error_se)
plt.ylabel('Reconstruction Error +/- SE')
plt.xlabel('Training Iteration')
plt.show()

# plot dictionary
grid = make_grid(model.D, nrow = int(np.sqrt(model.m)))
plt.imshow(grid.float().cpu().numpy()[0], cmap = 'gray')
plt.show()


# reconstruct new images
inputs = model.preprocess_inputs(x[-1000:])
a = model.encode(inputs)
recon = model.compute_recon(a) 


# plot inputs and recons
fig = plt.figure()
sub1 = fig.add_subplot(121)
sub2 = fig.add_subplot(122)
in_grid = make_grid(inputs.T.reshape([1000, 1, PATCH_SIZE, PATCH_SIZE]), nrow = 30)
rec_grid = make_grid(recon.T.reshape([1000, 1, PATCH_SIZE, PATCH_SIZE]), nrow = 30)
sub1.imshow(in_grid.cpu().numpy()[0], cmap = 'gray')
sub2.imshow(rec_grid.cpu().numpy()[0], cmap = 'gray')
sub1.set_title('Input')
sub2.set_title('Recon')
plt.show()