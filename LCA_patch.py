import os 
from random import shuffle

import cv2
import matplotlib.pyplot as plt 
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

PATCH_SIZE = 11


class LCADLPatch:
    def __init__(self, n_neurons, thresh = 0.1, tau = 1000, eta = 0.01, 
                 lca_iters = 2000, update_steps = 1000, batch_size = 256):
        ''' Performs sparse dictionary learning via LCA (Rozell et al. 2008) '''

        self.m = n_neurons 
        self.thresh = thresh 
        self.tau = tau 
        self.eta = eta 
        self.lca_iters = lca_iters 
        self.update_steps = update_steps
        self.batch_size = batch_size

        # We will store l2 reconstruction values in these to plot 
        self.error_mean, self.error_se = [], []

    def get_device(self, x):
        ''' Gets the device (GPU) x is on and returns None if on CPU ''' 

        return x.device.index if x.is_cuda else None

    def encode(self, x):
        ''' Computes sparse code given data vector x and dictionary matrix D '''

        # initialize membrane potentials
        u_t = torch.zeros(self.m, x.shape[1], device = self.get_device(x))

        # compute inhibition matrix
        G = torch.matmul(self.D.T, self.D) - torch.eye(self.m, device = self.get_device(self.D))

        # compute driving inputs
        b_t = torch.matmul(self.D.T, x) 

        for _ in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += (1 / self.tau) * (b_t - u_t - torch.matmul(G, a_t))

        return a_t 

    def update_D(self, x, a):
        ''' Updates the dictionary based on the reconstruction error. '''

        recon = self.compute_recon(a)
        error = x - recon
        self.track_l2_recon_error(error)
        self.D += torch.matmul(error, a.T) * self.eta
        self.normalize_D()

    def normalize_D(self):
        ''' Normalizes features such at each one has unit norm '''

        scale = torch.diag(1 / (self.D.norm(p = 2, dim = 0) + 1e-12))
        self.D = torch.matmul(self.D, scale)

    def soft_threshold(self, x):
        ''' Soft threshold '''
        return torch.maximum(x - self.thresh, torch.zeros_like(x))# - torch.maximum(-x - self.thresh, torch.zeros_like(x))

    def compute_recon(self, a):
        ''' Computes reconstruction given code '''
        return torch.matmul(self.D, a)

    def track_l2_recon_error(self, error):
        ''' Keeps track of the reconstruction error over training '''
        l2_error = error.norm(p = 2, dim = 0)
        self.error_mean.append(torch.mean(l2_error).item())
        self.error_se.append(torch.std(l2_error).item() / np.sqrt(self.batch_size))

    def preprocess_inputs(self, x, eps = 1e-12):
        ''' Scales the values of each patch to [0, 1] and then transforms each patch to have mean 0 '''
        x = (x - torch.min(x, 0)[0]) / (torch.max(x, 0)[0] - torch.min(x, 0)[0] + eps)
        x = x - torch.mean(x, 0)
        return x

    def run_model(self, X):
        self.n = X.shape[0]

        for step in range(self.update_steps):
            if step == 0:
                self.D = torch.randn(self.n, self.m, device = self.get_device(X)) * 0.01
                self.D = torch.maximum(self.D, torch.zeros_like(self.D))
                self.normalize_D()

            batch = X[:, np.random.choice(X.shape[1], self.batch_size, replace = False)]
            batch = self.preprocess_inputs(batch)
            a = self.encode(batch)
            self.update_D(batch, a)


# loading in random images from CIFAR dataset
imgs = CIFAR10(root = 'cifar/', download = True).data.astype(np.float32)
imgs_gray = np.mean(imgs, -1)
patches = imgs_gray[:, 16 - PATCH_SIZE // 2:16 + PATCH_SIZE // 2 + 1, 16 - PATCH_SIZE // 2:16 + PATCH_SIZE // 2 + 1]
x = patches.reshape([patches.shape[0], -1]).T
x = torch.from_numpy(x).cuda()
# x = torch.from_numpy(x)    # to use CPU instead of GPU


# run the model
model = LCADLPatch(512, lca_iters = 3000, update_steps = 2500, thresh = 0.075, batch_size = 512)
model.run_model(x[:, :-1000])


# plot error 
plt.errorbar(list(range(len(model.error_mean))), model.error_mean, yerr = model.error_se)
plt.ylabel('Reconstruction Error +/- SE')
plt.xlabel('Training Iteration')
plt.show()

# plot dictionary
grid = make_grid(model.D.T.reshape([model.m, 1, PATCH_SIZE, PATCH_SIZE]), nrow = 23)
plt.imshow(grid.cpu().numpy()[0], cmap = 'gray')
plt.show()


# reconstruct new images
inputs = model.preprocess_inputs(x[:, -1000:])
a = model.encode(inputs)
recon = model.compute_recon(a) 

# plot activations
plt.imshow(a.cpu().numpy(), cmap = 'gray')
plt.ylabel('Neuron Number')
plt.xlabel('Patch Number')
plt.show()

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