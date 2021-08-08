import os
from random import randint

import cv2
import numpy as np
from progressbar import ProgressBar
import torch
import torch.nn.functional as F

from lca_base import LCAConvBase 

class LCA3DConv(LCAConvBase):
    '''
    3D Convolutional LCA Model

    Args:
        kh (int): Kernel height of the dictionary features.
        kw (int): Kernel width of the dictionary features. 
        kt (int): Kernel depth of the dictionary features.
        stride_h (int): Stride of the kernel in the vert. direction.
        stride_w (int): Stride of the kernel in the horiz. direction.
        stride_t (int): Stride of the kernel in the depth direction.
    '''
    
    def __init__(self, kh=7, kw=7, kt=3, stride_h=1, stride_w=1, stride_t=1,
                 **kwargs):
        super(LCA3DConv, self).__init__(**kwargs)
        
        self.kh = kh
        self.kt = kt
        self.kw = kw 
        self.stride_h = stride_h 
        self.stride_t = stride_t 
        self.stride_w = stride_w

        assert self.kh % 2 != 0 and self.kw % 2 != 0
        self.create_weight_tensor()
        self.compute_padding_dims()

    def compute_du_norm(self, du):
        ''' Computes the norm of du to deterimine stopping '''

        return du.norm(p=2, dim=(1,2,3,4)).mean()

    def compute_input_drive(self, x):
        return F.conv3d(
            x,
            self.D,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad
        )

    def compute_lateral_connectivity(self):
        G = F.conv3d(
            self.D, 
            self.D, 
            stride=(self.stride_t, self.stride_h, self.stride_w), 
            padding=(self.kt - 1, self.kh - 1, self.kw - 1)
        )
        # to avoid inhibition from future neurons to past neurons
        # if kt != input depth
        # G[:, :, (G.shape[2]-1)//2+1:, :, :] = 0.0
        if not hasattr(self, 'n_surround_h'):
            self.n_surround_t = int(np.ceil((G.shape[-3] - 1) / 2))
            self.n_surround_h = int(np.ceil((G.shape[-2] - 1) / 2))
            self.n_surround_w = int(np.ceil((G.shape[-1] - 1) / 2))

        return G

    def compute_l2_error(self, error):
        ''' Compute l2 norm of the recon error  '''

        l2_error_per_sample = error.norm(p=2, dim=(1, 2, 3, 4))
        return torch.mean(l2_error_per_sample)

    def compute_l1_sparsity(self, acts):
        ''' Compute l1 norm of the activations  '''

        l1_norm_per_sample = acts.norm(p=1, dim=(1, 2, 3, 4))
        return torch.mean(l1_norm_per_sample)

    def compute_padding_dims(self):
        ''' Computes padding for forward and transpose convs '''

        if self.pad == 'same':
            self.input_pad = (0, (self.kh - 1) // 2, (self.kw - 1) // 2)
        elif self.pad == 'valid':
            self.input_pad = (0, 0, 0)
        else:
            raise ValueError

        self.recon_output_pad = (
            0,
            self.stride_h - 1 if self.stride_h > 1 else 0, 
            self.stride_w - 1 if self.stride_w > 1 else 0
        )

    def compute_recon(self, a):
        ''' Computes reconstruction given code '''

        return F.conv_transpose3d(
            a, 
            self.D,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad,
            output_padding=self.recon_output_pad
        )

    def create_weight_tensor(self):
        self.D = torch.randn(
            self.n_neurons,
            self.in_c,
            self.kt,
            self.kh,
            self.kw,
            device = self.device,
            dtype = self.dtype
        )
        self.D[:, :, 1:] = 0.0
        self.normalize_D()

    def lateral_competition(self, a, G):
        return F.conv3d(
            a,
            G,
            stride=1,
            padding=(self.n_surround_t, self.n_surround_h, self.n_surround_w)
        )

    def normalize_D(self, eps=1e-12):
        ''' Normalizes features such at each one has unit norm '''

        scale = (self.D.norm(p=2, dim=(1, 2, 3, 4), keepdim=True) + eps)
        self.D *= (1.0 / scale)

    def preprocess_inputs(self, x, eps=1e-12):
        ''' Scales each batch sample to [0, 1] and 
            zero-center's each frame '''

        if self.scale_inputs:
            minx = x.reshape(x.shape[0], -1).min(dim=-1)[0]
            maxx = x.reshape(x.shape[0], -1).max(dim=-1)[0]
            minx = minx.reshape(minx.shape[0], 1, 1, 1, 1)
            maxx = maxx.reshape(maxx.shape[0], 1, 1, 1, 1)
            x = (x - minx) / (maxx - minx + eps)
        if self.zero_center_inputs:
            x -= x.mean(dim=(1, 3, 4), keepdim = True)
            
        return x

    def update_D(self, a, error):
        ''' Updates the dictionary based on the recon error '''

        error = F.pad(error, (self.input_pad[2], self.input_pad[2], 
                              self.input_pad[1], self.input_pad[1], 
                              self.input_pad[0], self.input_pad[0]
            ))
        error = error.unfold(-3, self.kt, self.stride_t)
        error = error.unfold(-3, self.kh, self.stride_h)
        error = error.unfold(-3, self.kw, self.stride_w)
        update = torch.tensordot(a, error, dims=([0, 2, 3, 4], [0, 2, 3, 4]))
        self.D += update * self.eta
        self.normalize_D()


if __name__ == '__main__':
    BATCH_SIZE = 32
    DEVICE = 1
    DTYPE = torch.float32
    N_FRAMES_IN_TIME = 3
    UPDATE_STEPS = 1500

    # get paths to videos from imagenet video dataset
    # this is just a directory with subdirectories which contain images taken from videos
    data_dir = '/media/mteti/1TB_SSD/NEMO/data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/'
    vid_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if 'resized' in d]
    fpaths = [[os.path.join(d, f) for f in os.listdir(d)] for d in vid_dirs]

    model = LCA3DConv(
        kh=9,
        kw=9,
        kt=N_FRAMES_IN_TIME,
        stride_h=4,
        stride_w=4,
        stride_t=1,
        in_c=1,
        cudnn_benchmark=True,
        n_neurons=128,
        result_dir='LCA_Test',
        tau=1000,
        eta=1e-3,
        lca_tol=1e-3,
        lca_iters=4000,
        tau_decay_factor=7e-4,
        device=DEVICE,
        dtype=DTYPE,
        nonneg=True,
        act_write_step=200,
        dict_write_step=200,
        recon_write_step=200,
        recon_error_write_step=200,
        input_write_step=200,
        track_metrics=True
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