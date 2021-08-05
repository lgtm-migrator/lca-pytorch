import numpy as np
import torch
import torch.nn.functional as F

from lca_base import LCAConvBase 

class LCA3DConv(LCAConvBase):
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

    def compute_l2_recon_error(self, error):
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

        scale = (self.D.norm(p=2, dim=(1, 2, 3, 4), keepdim = True) + eps)
        self.D *= (1.0 / scale)

    def preprocess_inputs(self, x, eps=1e-12):
        ''' Scales each batch sample to [0, 1] and 
            zero-center's each frame '''

        if self.scale_imgs:
            minx = x.reshape(x.shape[0], -1).min(dim=-1)[0]
            maxx = x.reshape(x.shape[0], -1).max(dim=-1)[0]
            minx = minx.reshape(minx.shape[0], 1, 1, 1, 1)
            maxx = maxx.reshape(maxx.shape[0], 1, 1, 1, 1)
            x = (x - minx) / (maxx - minx + eps)
        if self.zero_center_imgs:
            x -= x.mean(dim=(1, 3, 4), keepdim = True)
            
        return x