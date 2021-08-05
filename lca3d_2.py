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

    def normalize_D(self, eps=1e-12):
        ''' Normalizes features such at each one has unit norm '''

        scale = (self.D.norm(p = 2, dim = (1, 2, 3, 4), keepdim = True) + eps)
        self.D *= (1.0 / scale)