import csv
import os 
from random import randint, shuffle

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from progressbar import ProgressBar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

BATCH_SIZE = 64
DEVICE = 1
DTYPE = torch.float32
N_FRAMES_IN_TIME = 3
UPDATE_STEPS = 1500


class LCADLConvSpatialComp:
    def __init__(self, n_neurons, in_c, thresh = 0.1, tau = 1500, 
                 eta = 0.01, lca_iters = 2000, kh = 7, kw = 7, kt = 9, stride_h = 1, stride_w = 1, 
                 stride_t = 1, pad = 'same', device = None, dtype = torch.float32, learn_dict = True,
                 nonneg = True, track_metrics = True, dict_write_step = -1, recon_error_write_step = -1,
                 act_write_step = -1, input_write_step = -1, recon_write_step = -1, 
                 result_dir = 'LCA_results', scale_imgs = True, zero_center_imgs = True):
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
        self.track_metrics = track_metrics
        self.dict_write_step = dict_write_step 
        self.recon_error_write_step = recon_error_write_step
        self.act_write_step = act_write_step
        self.input_write_step = input_write_step
        self.recon_write_step = recon_write_step
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok = True)
        self.metric_fpath = os.path.join(result_dir, 'metrics.xz')
        self.tensor_write_fpath = os.path.join(result_dir, 'written_tensors.h5')
        self.scale_imgs = scale_imgs 
        self.zero_center_imgs = zero_center_imgs
        self.ts = 0

        assert(self.kh % 2 != 0 and self.kw % 2 != 0)

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
        self.input_pad = (0, (self.kh-1)//2, (self.kw-1)//2) if self.pad == 'same' else (0, 0, 0)
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
        # to avoid inhibition from future neurons to past neurons
        # if kt != input depth
        # G[:, :, (G.shape[2]-1)//2+1:, :, :] = 0.0
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

    def encode(self, x):
        ''' Computes sparse code given data vector x and dictionary matrix D '''

        if self.track_metrics:
            l1_sparsity = torch.zeros(self.lca_iters, dtype = self.dtype, device = self.device)
            l2_error = torch.zeros(self.lca_iters, dtype = self.dtype, device = self.device)
            timestep = np.zeros([self.lca_iters], dtype = np.int64)

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

        for lca_iter in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += self.charge_rate * (b_t - u_t - self.lateral_competition(a_t, G) + a_t)

            if self.track_metrics or lca_iter == self.lca_iters - 1:
                recon = self.compute_recon(a_t)
                recon_error = x - recon

            if self.track_metrics:
                l2_error[lca_iter] = self.compute_l2_recon_error(recon_error)
                l1_sparsity[lca_iter] = self.compute_l1_sparsity(a_t)
                timestep[lca_iter] = self.ts

            self.ts += 1

        if self.track_metrics:
            self.write_obj_values(timestep, l2_error, l1_sparsity)

        assert(torch.isnan(a_t).sum() == 0.0)
        return a_t, recon_error, recon

    def update_D(self, x, a, error):
        ''' Updates the dictionary based on the reconstruction error. '''

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

    def normalize_D(self, eps = 1e-12):
        ''' Normalizes features such at each one has unit norm '''

        scale = (self.D.norm(p = 2, dim = (1, 2, 3, 4), keepdim = True) + eps)
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

    def compute_l2_recon_error(self, error):
        ''' Keeps track of the reconstruction error over training '''
        l2_error_per_sample = error.norm(p = 2, dim = (1, 2, 3, 4))
        return torch.mean(l2_error_per_sample)

    def compute_l1_sparsity(self, acts):
        l1_norm_per_sample = acts.norm(p = 1, dim = (1, 2, 3, 4))
        return torch.mean(l1_norm_per_sample)

    def write_obj_values(self, timesteps, l2_error, l1_sparsity):
        obj_df = pd.DataFrame(
            {
                'Timestep': timesteps,
                'L2_Recon_Error': l2_error.float().cpu().numpy(),
                'L1_Sparsity': l1_sparsity.float().cpu().numpy()
            }
        )
        obj_df.to_csv(
            self.metric_fpath,
            header = True if not os.path.isfile(self.metric_fpath) else False,
            index = False,
            mode = 'a'
        )

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

    def append_h5(self, key, data):
        with h5py.File(self.tensor_write_fpath, 'a') as h5file:
            h5file.create_dataset(key, data = data.cpu().numpy())

    def forward(self, x):
        if self.ts % self.dict_write_step == 0 and self.dict_write_step != -1:
            self.append_h5('D_{}'.format(self.ts), self.D)

        x = self.preprocess_inputs(x)
        a, recon_error, recon = self.encode(x)

        if self.learn_dict:
            self.update_D(x, a, recon_error)

        if self.ts % self.act_write_step == 0 and self.act_write_step != -1:
            self.append_h5('a_{}'.format(self.ts), a)
        if self.ts % self.recon_write_step == 0 and self.recon_write_step != -1:
            self.append_h5('recon_{}'.format(self.ts), recon)
        if self.ts % self.input_write_step == 0 and self.input_write_step != -1:
            self.append_h5('input_{}'.format(self.ts), x)
        if self.ts % self.recon_error_write_step == 0 and self.recon_error_write_step != -1:
            self.append_h5('recon_error_{}'.format(self.ts), recon_error)

        return a


# loading in random images from CIFAR dataset
data_dir = '/media/mteti/1TB_SSD/NEMO/data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/'
vid_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if 'resized' in d]
fpaths = [[os.path.join(d, f) for f in os.listdir(d)] for d in vid_dirs]


# run the model
model = LCADLConvSpatialComp(
    128, 
    1,
    lca_iters = 3000, 
    thresh = 0.1, 
    device = DEVICE,
    stride_h = 4,
    stride_w = 4,
    stride_t = 1,
    kh = 13,
    kw = 13,
    kt = N_FRAMES_IN_TIME,
    nonneg = True,
    pad = 'same',
    dtype = DTYPE,
    tau = 1500,
    eta = 0.01,
    act_write_step = -1,
    dict_write_step = -1,
    recon_write_step = -1,
    recon_error_write_step = -1,
    input_write_step = -1,
    result_dir = 'LCA_results_run3',
    track_metrics = False
)

for step in ProgressBar()(range(UPDATE_STEPS)):
    batch_vids = [fpaths[ind] for ind in np.random.choice(len(fpaths), BATCH_SIZE, replace = False)]
    frame_inds = [randint(0, len(vid)-N_FRAMES_IN_TIME) if len(vid) >= N_FRAMES_IN_TIME else 'remove' for vid in batch_vids]
    batch_frames = [v[ind:ind+N_FRAMES_IN_TIME] for v, ind in zip(batch_vids, frame_inds) if ind != 'remove']
    batch_imgs = [[torch.from_numpy(cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE)) for img_fpath in f] for f in batch_frames]
    batch_imgs_tensor = [torch.unsqueeze(torch.stack(imgs), dim = 0) for imgs in batch_imgs]
    batch = torch.stack(batch_imgs_tensor)
    batch = batch[..., 16:48]
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
grids = [make_grid(model.D[:, :, t], nrow = int(np.sqrt(model.m))) for t in range(model.D.shape[2])]
imageio.mimwrite('lca3d_feats.gif', [grid[0].float().cpu().numpy() for grid in grids], fps = 5)

# reconstruct new images
batch = model.preprocess_inputs(batch)
a = model.encode(batch)
recon = model.compute_recon(a) 

# write out input and recon .gifs
input_grids = [make_grid(batch[:, :, t], nrow = int(np.sqrt(BATCH_SIZE))) for t in range(N_FRAMES_IN_TIME)]
recon_grids = [make_grid(recon[:, :, t], nrow = int(np.sqrt(BATCH_SIZE))) for t in range(N_FRAMES_IN_TIME)]
diff_grids = [igrid - rgrid for igrid, rgrid in zip(input_grids, recon_grids)]
imageio.mimwrite('inputs.gif', [grid[0].float().cpu().numpy() for grid in input_grids], fps = 5)
imageio.mimwrite('recons.gif', [grid[0].float().cpu().numpy() for grid in recon_grids], fps = 5)
imageio.mimwrite('diffs.gif', [grid[0].float().cpu().numpy() for grid in diff_grids], fps = 5)