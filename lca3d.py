import os 
from random import shuffle

import cv2
import h5py
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
DEVICE = 1
DTYPE = torch.float16
UPDATE_STEPS = 1500


class LCADLConvSpatialComp:
    def __init__(self, n_neurons, in_c, thresh = 0.1, tau = 1500, 
                 eta = 0.01, lca_iters = 2000, kh = 7, kw = 7, kt = 9, stride_h = 1, stride_w = 1, 
                 stride_t = 1, pad = 'same', device = None, dtype = torch.float32, learn_dict = True,
                 nonneg = True, track_loss = True, dict_write_step = -1, 
                 act_write_step = -1, input_write_step = -1, recon_write_step = -1, 
                 result_fpath = 'LCA_results', scale_imgs = True, zero_center_imgs = True):
        ''' Performs sparse dictionary learning via LCA (Rozell et al. 2008) '''

        self.m = n_neurons
        self.in_c = in_c 
        self.thresh = thresh 
        self.tau = tau 
        self.charge_rate = 1.0 / self.tau
        self.eta = eta 
        self.lca_iters = lca_iters 
        self.kh = kh 
        self.kw = kw 
        self.kt = kt
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.stride_t = stride_t
        self.pad = pad 
        self.device = device 
        self.dtype = dtype
        self.learn_dict = learn_dict
        self.nonneg = nonneg
        self.track_loss = track_loss
        self.dict_write_step = dict_write_step 
        self.act_write_step = act_write_step
        self.input_write_step = input_write_step
        self.recon_write_step = recon_write_step
        self.result_fpath = result_fpath
        self.scale_imgs = scale_imgs 
        self.zero_center_imgs = zero_center_imgs
        self.ts = 0

        assert(self.kh % 2 != 0 and self.kw % 2 != 0)

        # We will store l2 reconstruction values in these to plot 
        if self.track_loss:
            self.recon_error_mean, self.recon_error_se = [], []
            self.l1_sparsity_mean, self.l1_sparsity_se = [], []

        # create the weight tensor and compute padding
        self.compute_padding_dims()
        self.create_weight_mat()

    def create_weight_mat(self):
        self.D = torch.randn(
            self.m,
            self.in_c,
            self.kt,
            self.kh,
            self.kw,
            device = self.device,
            dtype = self.dtype
        )
        self.normalize_D()

    def compute_padding_dims(self):
        ''' Computes padding for forward and transpose convs '''
        self.input_pad = (0, (self.kh-1)//2, (self.kw-1)//2) if self.pad == 'same' else 0
        self.recon_output_pad = (
            0,
            self.stride_h - 1 if (self.kh % 2 != 0 and self.stride_h > 1) else 0, 
            self.stride_w - 1 if (self.kw % 2 != 0 and self.stride_w > 1) else 0
        )

    def compute_lateral_connectivity(self):
        G = F.conv3d(
            self.D, 
            self.D, 
            stride = (self.stride_t, self.stride_h, self.stride_w), 
            padding = (self.kt - 1, self.kh - 1, self.kw - 1)
        )
        if not hasattr(self, 'n_surround_h'):
            self.n_surround_t = int(np.ceil((G.shape[-3] - 1) / 2))
            self.n_surround_h = int(np.ceil((G.shape[-2] - 1) / 2))
            self.n_surround_w = int(np.ceil((G.shape[-1] - 1) / 2))

        return G

    def lateral_competition(self, a, G):
        return F.conv3d(
            a,
            G,
            stride = 1,
            padding = (self.n_surround_t, self.n_surround_h, self.n_surround_w)
        )

    def get_device(self, x):
        ''' Gets the device (GPU) x is on and returns None if on CPU ''' 

        return x.device.index if x.is_cuda else None

    def encode(self, x):
        ''' Computes sparse code given data vector x and dictionary matrix D '''

        # input drive
        b_t = F.conv3d(
            x,
            self.D,
            stride = (self.stride_t, self.stride_h, self.stride_w),
            padding = self.input_pad
        )

        # initialize membrane potentials
        u_t = torch.zeros_like(b_t)

        # compute inhibition matrix
        G = self.compute_lateral_connectivity()

        for _ in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += self.charge_rate * (b_t - u_t - self.lateral_competition(a_t, G) + a_t)


        if self.track_loss: self.track_l1_sparsity(a_t)
        if self.ts % self.act_write_step == 0 and self.act_write_step != -1:
            self.append_h5(self.result_fpath, 'a_{}'.format(self.ts), a_t)

        return a_t 

    def update_D(self, x, a):
        ''' Updates the dictionary based on the reconstruction error. '''

        recon = self.compute_recon(a)

        if self.ts % self.recon_write_step == 0 and self.recon_write_step != -1:
            self.append_h5(self.result_fpath, 'recon_{}'.format(self.ts), recon)

        error = x - recon
        if self.track_loss: self.track_l2_recon_error(error)
        error = F.pad(
            error,
            (
                self.input_pad[2],
                self.input_pad[2],
                self.input_pad[1],
                self.input_pad[1],
                self.input_pad[0],
                self.input_pad[0]
            )
        ).unfold(
            -3,
            self.kt,
            self.stride_t
        ).unfold(
            -3,
            self.kh,
            self.stride_h
        ).unfold(
            -3,
            self.kw,
            self.stride_w
        )
        update = torch.tensordot(a, error, dims = ([0, 2, 3, 4], [0, 2, 3, 4]))
        self.D += update * self.eta
        self.normalize_D()

        if self.ts % self.dict_write_step == 0 and self.dict_write_step != -1:
            self.append_h5(self.result_fpath, 'D_{}'.format(self.ts), self.D)

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
        return F.conv_transpose3d(
            a, 
            self.D,
            stride = (self.stride_t, self.stride_h, self.stride_w),
            padding = self.input_pad,
            output_padding = self.recon_output_pad
        )

    def track_l2_recon_error(self, error):
        ''' Keeps track of the reconstruction error over training '''
        l2_error = error.norm(p = 2, dim = (1, 2, 3))
        self.recon_error_mean.append(torch.mean(l2_error).item())
        self.recon_error_se.append(torch.std(l2_error).item() / np.sqrt(error.shape[0]))

    def track_l1_sparsity(self, acts):
        l1_sparsity = acts.norm(p = 1, dim = (1, 2, 3))
        self.l1_sparsity_mean.append(torch.mean(l1_sparsity).item())
        self.l1_sparsity_se.append(torch.std(l1_sparsity).item() / np.sqrt(acts.shape[0]))

    def preprocess_inputs(self, x, eps = 1e-12):
        ''' Scales the values of each patch to [0, 1] and then transforms each patch to have mean 0 '''

        if self.scale_imgs:
            x = x.permute(0, 2, 1, 3, 4)
            minx = x.reshape(x.shape[0], x.shape[1], -1). \
                min(dim = -1, keepdim = True)[0][..., None, None]
            maxx = x.reshape(x.shape[0], x.shape[1], -1). \
                max(dim = -1, keepdim = True)[0][..., None, None]
            x = (x - minx) / (maxx - minx + eps)
            x = x.permute(0, 2, 1, 3, 4)
        if self.zero_center_imgs:
            x -= x.mean(dim = (1, 3, 4), keepdim = True)
            
        return x

    def append_h5(self, fpath, key, data):
        with h5py.File(fpath, 'a') as h5file:
            h5file.create_dataset(key, data = data.cpu().numpy())

    def forward(self, x):
        assert x.shape[1] == self.in_c

        x = self.preprocess_inputs(x)

        if self.ts % self.input_write_step == 0 and self.input_write_step != -1:
            self.append_h5(self.result_fpath, 'input_{}'.format(self.ts), x)

        a = self.encode(x)

        if self.learn_dict:
            self.update_D(x, a)

        self.ts += 1

        return a


# loading in random images from CIFAR dataset
imgs = CIFAR10(root = 'cifar/', download = True).data.astype(np.float32)
imgs_gray = torch.from_numpy(np.mean(imgs, -1)).unsqueeze(1)
imgs_gray = torch.stack([imgs_gray for _ in range(15)], dim = 2)
train, test = imgs_gray[:-100], imgs_gray[-100:]


# run the model
model = LCADLConvSpatialComp(
    64, 
    1,
    lca_iters = 1500, 
    thresh = 0.1, 
    device = DEVICE,
    stride_h = 2,
    stride_w = 4,
    stride_t = 2,
    kh = 7,
    kw = 9,
    kt = 5,
    nonneg = False,
    pad = 'same',
    dtype = DTYPE,
    tau = 1000
)

for step in ProgressBar()(range(UPDATE_STEPS)):
    batch = train[np.random.choice(train.shape[0], BATCH_SIZE, replace = False)]
    batch = batch.type(DTYPE).to(DEVICE)
    a = model.forward(batch)


# plot error 
plt.errorbar(
    list(range(len(model.recon_error_mean))), 
    model.recon_error_mean, 
    yerr = model.recon_error_se
)
plt.ylabel('Reconstruction Error +/- SE')
plt.xlabel('Training Iteration')
plt.show()

# plot sparsity
plt.errorbar(
    list(range(len(model.l1_sparsity_mean))),
    model.l1_sparsity_mean,
    yerr = model.l1_sparsity_se
)
plt.ylabel('L1 Sparsity +/- SE')
plt.xlabel('Training Iteration')
plt.show()

# plot dictionary
grid = make_grid(model.D, nrow = int(np.sqrt(model.m)))
plt.imshow(grid.float().cpu().numpy()[0], cmap = 'gray')
plt.show()


# reconstruct new images
inputs = model.preprocess_inputs(test)
inputs = inputs.type(DTYPE).to(DEVICE)
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