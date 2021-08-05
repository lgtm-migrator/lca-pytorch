import os

import h5py
import numpy as np
import pandas as pd
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
        self.metric_fpath = os.path.join(self.result_dir, 'metrics.xz')
        self.tensor_write_fpath = os.path.join(self.result_dir, 'tensors.h5')

    def create_trackers(self):
        l1_sparsity = torch.zeros(self.lca_iters, dtype=self.dtype, 
                                  device=self.device)
        l2_error = torch.zeros(self.lca_iters, dtype=self.dtype, 
                               device=self.device)
        timestep = np.zeros([self.lca_iters], dtype=np.int64)
        tau_vals = np.zeros([self.lca_iters], dtype=np.float32)

        return timestep, tau_vals, l1_sparsity, l2_error

    def encode(self, x):
        ''' Computes sparse code given data x and dictionary D '''

        if self.track_metrics:
            timestep, tau_vals, l1_sparsity, l2_error = self.create_trackers()

        # input drive
        b_t = self.compute_input_drive(x)

        # initialize membrane potentials
        u_t = torch.zeros_like(b_t)

        # compute inhibition matrix
        G = self.compute_lateral_connectivity()

        # initialize time constant
        tau = self.tau

        for lca_iter in range(self.lca_iters):
            a_t = self.soft_threshold(u_t)
            u_t += (1 / tau) * (b_t 
                                - u_t 
                                - self.lateral_competition(a_t, G) 
                                + a_t)

            if self.track_metrics or lca_iter == self.lca_iters - 1:
                recon = self.compute_recon(a_t)
                recon_error = x - recon

            if self.track_metrics:
                l2_error[lca_iter] = self.compute_l2_recon_error(recon_error)
                l1_sparsity[lca_iter] = self.compute_l1_sparsity(a_t)
                timestep[lca_iter] = self.ts
                tau_vals[lca_iter] = tau

            tau -= tau * self.tau_decay_factor
            self.ts += 1

        if self.track_metrics:
            self.write_obj_values(timestep, l2_error, l1_sparsity, tau_vals)

        return a_t, recon_error, recon

    def forward(self, x):
        if self.ts % self.dict_write_step == 0 and self.dict_write_step != -1:
            self.write_tensors('D_{}'.format(self.ts), self.D)

        x = self.preprocess_inputs(x)
        a, recon_error, recon = self.encode(x)

        if self.learn_dict:
            self.update_D(x, a, recon_error)

        if self.ts % self.act_write_step == 0 and self.act_write_step != -1:
            self.write_tensors('a_{}'.format(self.ts), a)
        if (self.ts % self.recon_write_step == 0 
                and self.recon_write_step != -1):
            self.write_tensors('recon_{}'.format(self.ts), recon)
        if (self.ts % self.input_write_step == 0 
                and self.input_write_step != -1):
            self.write_tensors('input_{}'.format(self.ts), x)
        if (self.ts % self.recon_error_write_step == 0 
                and self.recon_error_write_step != -1):
            self.write_tensors('recon_error_{}'.format(self.ts), recon_error)

        return a

    def soft_threshold(self, x):
        ''' Soft threshold transfer function '''

        if self.nonneg:
            return F.relu(x - self.thresh)
        else:
            return F.relu(x - self.thresh) - F.relu(-x - self.thresh)

    def write_obj_values(self, timesteps, l2_error, l1_sparsity, tau_vals):
        ''' Write out objective values to file '''

        obj_df = pd.DataFrame(
            {
                'Timestep': timesteps,
                'L2_Recon_Error': l2_error.float().cpu().numpy(),
                'L1_Sparsity': l1_sparsity.float().cpu().numpy(),
                'Time_Constant_Tau': tau_vals
            }
        )
        obj_df.to_csv(
            self.metric_fpath,
            header = True if not os.path.isfile(self.metric_fpath) else False,
            index = False,
            mode = 'a'
        )

    def write_tensors(self, key, data):
        ''' Writes out tensors to a HDF5 file. '''

        with h5py.File(self.tensor_write_fpath, 'a') as h5file:
            h5file.create_dataset(key, data = data.cpu().numpy())