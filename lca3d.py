from copy import deepcopy
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
        kt (int): Kernel depth of the dictionary features. Currently
            only supports kt equal to input depth. 
        stride_h (int): Stride of the kernel in the vert. direction.
        stride_w (int): Stride of the kernel in the horiz. direction.
        stride_t (int): Stride of the kernel in the depth direction.
            Currently supports only stride_t equal to 1. 
    '''
    def __init__(self, kh=7, kw=7, kt=3, stride_h=1, stride_w=1, stride_t=1,
                 **kwargs):
        super(LCA3DConv, self).__init__(**kwargs)

        assert ((kh % 2 != 0 and kw % 2 != 0) or 
                (kh % 2 == 0 and kw % 2 == 0)), (
                'kh and kw should either both be even or both be odd numbers')
        assert stride_h == 1 or stride_h % 2 == 0
        assert stride_w == 1 or stride_w % 2 == 0
        assert stride_t == 1
        
        self.kernel_odd = True if kh % 2 != 0 else False
        self.kh = kh
        self.kt = kt
        self.kw = kw 
        self.stride_h = stride_h 
        self.stride_t = stride_t 
        self.stride_w = stride_w
        self.write_params(deepcopy(vars(self)))

        self.init_weight_tensor()
        self.compute_input_pad()
        self.compute_inhib_pad()
        self.compute_recon_pad()

    def compute_inhib_pad(self):
        ''' Computes padding for compute_lateral_connectivity '''
        self.lat_conn_pad = [0]

        if self.kernel_odd or self.stride_h == 1:
            self.lat_conn_pad.append(self.kh - 1)
        else:
            self.lat_conn_pad.append(self.kh - self.stride_h)

        if self.kernel_odd or self.stride_w == 1:
            self.lat_conn_pad.append(self.kw - 1)
        else:
            self.lat_conn_pad.append(self.kw - self.stride_w)
        
        self.lat_conn_pad = tuple(self.lat_conn_pad)        

    def compute_input_drive(self, x):
        assert x.shape[2] == self.kt
        return F.conv3d(
            x,
            self.D,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad
        )

    def compute_input_pad(self):
        ''' Computes padding for forward convolution '''
        if self.pad == 'same':
            if self.kernel_odd:
                self.input_pad = (0, (self.kh - 1) // 2, (self.kw - 1) // 2)
            else:
                raise NotImplementedError(
                    "Even kh and kw implemented only for 'valid' padding.")
        elif self.pad == 'valid':
            self.input_pad = (0, 0, 0)
        else:
            raise ValueError

    def compute_lateral_connectivity(self):
        G = F.conv3d(
            self.D, 
            self.D, 
            stride=(self.stride_t, self.stride_h, self.stride_w), 
            padding=self.lat_conn_pad
        )
        if not hasattr(self, 'surround'):
            self.compute_n_surround(G)

        return G

    def compute_recon(self, a):
        ''' Computes reconstruction given code '''
        return F.conv_transpose3d(
            a, 
            self.D,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad,
            output_padding=self.recon_output_pad
        )

    def compute_recon_pad(self):
        ''' Computes output padding for recon conv transpose '''
        if self.kernel_odd:
            self.recon_output_pad = (0, self.stride_h - 1, self.stride_w - 1)
        else:
            self.recon_output_pad = (0, 0, 0)

    def compute_update(self, a, error):
        error = F.pad(error, (self.input_pad[2], self.input_pad[2], 
                              self.input_pad[1], self.input_pad[1], 
                              self.input_pad[0], self.input_pad[0]
            ))
        error = error.unfold(-3, self.kt, self.stride_t)
        error = error.unfold(-3, self.kh, self.stride_h)
        error = error.unfold(-3, self.kw, self.stride_w)

        return torch.tensordot(a, error, dims=([0, 2, 3, 4], [0, 2, 3, 4]))

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
        return F.conv3d(a, G, stride=1, padding=self.surround)