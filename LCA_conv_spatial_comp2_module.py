import os 
from random import shuffle

import cv2
import matplotlib.pyplot as plt 
import numpy as np
from progressbar import ProgressBar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

BATCH_SIZE = 128
UPDATE_STEPS = 2000


class LCADLConvSpatialComp:
    def __init__(self, n_neurons, in_c, in_h, in_w, thresh = 0.1, tau = 50, 
                 eta = 0.01, lca_iters = 2000, kh = 7, kw = 7, stride = 1, 
                 pad = 'same', device = None, dtype = torch.float32, learn_dict = True,
                 nonneg = True):
        ''' Performs sparse dictionary learning via LCA (Rozell et al. 2008) '''

        self.m = n_neurons
        self.in_c = in_c 
        self.in_h = in_h 
        self.in_w = in_w 
        self.thresh = thresh 
        self.tau = tau 
        self.charge_rate = 1.0 / self.tau
        self.eta = eta 
        self.lca_iters = lca_iters 
        self.kh = kh 
        self.kw = kw 
        self.stride = stride 
        self.pad = pad 
        self.device = device 
        self.dtype = dtype
        self.learn_dict = learn_dict
        self.nonneg = nonneg
        self.input_pad = ((self.kh-1)//2, (self.kw-1)//2) if self.pad == 'same' else 0
        self.recon_output_pad = (
            self.stride - 1 if (self.kh % 2 != 0 and self.stride > 1) else 0, 
            self.stride - 1 if (self.kh % 2 != 0 and self.stride > 1) else 0
        )

        assert(self.kh > 0 and self.kw > 0)
        assert(self.kh % 2 != 0 and self.kw % 2 != 0)
        assert(self.stride >= 1)
        assert(self.eta > 0)

        # We will store l2 reconstruction values in these to plot 
        self.error_mean, self.error_se = [], []

        self.create_weight_mat()

    def create_weight_mat(self):
        self.D = torch.randn(
            self.m,
            self.in_c,
            self.kh,
            self.kw,
            device = self.device,
            dtype = self.dtype
        )
        self.normalize_D()

    def compute_lateral_connectivity(self):
        G = F.conv2d(
            self.D, 
            self.D, 
            stride = self.stride, 
            padding = (self.kh - 1, self.kw - 1)
        )
        if not hasattr(self, 'n_surround_h'):
            self.n_surround_h = int(np.ceil((G.shape[-2] - 1) / 2))
            self.n_surround_w = int(np.ceil((G.shape[-1] - 1) / 2))

        return G

    def lateral_competition(self, a, G):
        return F.conv2d(
            a,
            G,
            stride = 1,
            padding = (self.n_surround_h, self.n_surround_w)
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
            padding = self.input_pad
        )

        # initialize membrane potentials
        u_t = torch.zeros_like(b_t)

        # compute inhibition matrix
        G = self.compute_lateral_connectivity()

        for _ in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += self.charge_rate * (b_t - u_t - self.lateral_competition(a_t, G) + a_t)

        return a_t 

    def update_D(self, x, a):
        ''' Updates the dictionary based on the reconstruction error. '''

        recon = self.compute_recon(a)
        error = x - recon
        self.track_l2_recon_error(error)
        error = F.unfold(
            error, 
            (self.kh, self.kw), 
            padding = self.input_pad, 
            stride = self.stride
        )
        error = error.reshape(
            error.shape[0], 
            self.in_c, 
            self.kh, 
            self.kw, 
            a.shape[-2], 
            a.shape[-1]
        )
        update = torch.tensordot(a, error, dims = ([0, 2, 3], [0, -2, -1]))
        self.D += update * self.eta
        self.normalize_D()

    def normalize_D(self, eps = 1e-12):
        ''' Normalizes features such at each one has unit norm '''

        scale = (self.D.norm(p = 2, dim = (1, 2, 3), keepdim = True) + eps)
        self.D /= scale

    def soft_threshold(self, x):
        ''' Soft threshold '''
        if self.nonneg:
            return F.relu(x - self.thresh)
        else:
            return F.relu(x - self.thresh) - F.relu(-x - self.thresh)

    def compute_recon(self, a):
        ''' Computes reconstruction given code '''
        return F.conv_transpose2d(
            a, 
            self.D,
            stride = self.stride,
            padding = self.input_pad,
            output_padding = self.recon_output_pad
        )

    def track_l2_recon_error(self, error):
        ''' Keeps track of the reconstruction error over training '''
        l2_error = error.norm(p = 2, dim = (1, 2, 3))
        self.error_mean.append(torch.mean(l2_error).item())
        self.error_se.append(torch.std(l2_error).item() / np.sqrt(error.shape[0]))

    def preprocess_inputs(self, x, eps = 1e-12):
        ''' Scales the values of each patch to [0, 1] and then transforms each patch to have mean 0 '''
        x = x.type(self.dtype).to(self.device)
        minx = x.reshape(x.shape[0], -1).min(dim = -1, keepdim = True)[0][..., None, None]
        maxx = x.reshape(x.shape[0], -1).max(dim = -1, keepdim = True)[0][..., None, None]
        x = (x - minx) / (maxx - minx + eps)
        x -= x.mean(dim = (1, 2, 3), keepdim = True)
        return x

    def forward(self, x):
        x = self.preprocess_inputs(x)
        a = self.encode(x)

        if self.learn_dict:
            self.update_D(x, a)

        return a


# loading in random images from CIFAR dataset
imgs = CIFAR10(root = 'cifar/', download = True).data.astype(np.float32)
imgs_gray = torch.from_numpy(np.mean(imgs, -1)).unsqueeze(1)
train, test = imgs_gray[:-100], imgs_gray[-100:]


# run the model
model = LCADLConvSpatialComp(
    64, 
    1,
    32,
    32,
    lca_iters = 1500, 
    thresh = 0.1, 
    device = 1,
    stride = 2,
    kh = 9,
    kw = 9,
    nonneg = False,
    pad = 'valid',
    dtype = torch.float16
)

for step in ProgressBar()(range(UPDATE_STEPS)):
    batch = train[np.random.choice(train.shape[0], BATCH_SIZE, replace = False)]
    a = model.forward(batch)


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
inputs = model.preprocess_inputs(test)
a = model.encode(inputs)
recon = model.compute_recon(a) 


# plot inputs and recons
fig = plt.figure()
sub1 = fig.add_subplot(121)
sub2 = fig.add_subplot(122)
in_grid = make_grid(inputs, nrow = 10)
rec_grid = make_grid(recon, nrow = 10)
sub1.imshow(in_grid.float().cpu().numpy()[0], cmap = 'gray')
sub2.imshow(rec_grid.float().cpu().numpy()[0], cmap = 'gray')
sub1.set_title('Input')
sub2.set_title('Recon')
plt.show()