import torch

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