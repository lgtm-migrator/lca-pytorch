import os

import numpy as np
import torch 
import torch.nn.functional as F


class LCAConvBase:
    def __init__(self, n_neurons, in_c, result_dir, thresh=0.1, tau=1500, 
                 eta=1e-3, lca_iters=3000, pad='same', device=None, 
                 dtype=torch.float32, nonneg=False, learn_dict=True, 
                 track_metrics=True, scale_inputs=True, 
                 zero_center_inputs=True, dict_write_step=-1, 
                 recon_write_step=-1, act_write_step=-1, 
                 recon_error_write_step=-1, input_write_step=-1, 
                 tau_decay_factor=0.0):

        self.act_write_step = act_write_step 
        self.device = device 
        self.dict_write_step = dict_write_step
        self.dtype = dtype 
        self.eta = eta 
        self.in_c = in_c 
        self.input_write_step = input_write_step 
        self.lca_iters = lca_iters 
        self.learn_dict = learn_dict
        self.n_neurons = n_neurons 
        self.nonneg = nonneg 
        self.pad = pad
        self.recon_error_write_step = recon_error_write_step
        self.recon_write_step = recon_write_step
        self.result_dir = result_dir 
        self.scale_inputs = scale_inputs 
        self.tau = tau 
        self.tau_decay_factor = tau_decay_factor
        self.thresh = thresh 
        self.track_metrics = track_metrics
        self.ts = 0
        self.zero_center_inputs = zero_center_inputs

        os.makedirs(self.result_dir, exist_ok = True)
        self.metric_fpath = os.path.join(result_dir, 'metrics.xz')
        self.tensor_write_fpath = os.path.join(result_dir, 'tensors.h5')