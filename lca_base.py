import os

import h5py
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F


class LCAConvBase:
    '''
    Base class for LCA models.

    Required Args:
        n_neurons (int): Number of neurons / dictionary elements.
        in_c (int): Number of channels / features in the input data.
        result_dir (str): Path to dir where results will be saved.

    Optional Args:
        thresh (float): Threshold for the LCA transfer function.
        tau (int): LCA time constant. 
        eta (float): Learning rate for dictionary updates.
        lca_iters (int): Number of LCA timesteps per forward pass.
        pad ('same' or 'valid'): Input padding method.
        device (int): GPU to run on. 
        dtype (torch.dtype): Data type to use.
        nonneg (bool): True for rectified activations, False for 
            non-rectified activations.
        learn_dict (bool): True to update dict features in 
            alternation with LCA, False to not update dict.
        track_metrics (bool): True to track and write out objective
            metrics over the run.
        scale_inputs (bool): If True, inputs values will be scaled to
            [0, 1] per batch sample. 
        zero_center_inputs (bool): If True, inputs will be centered 
            at 0 per batch sample and depth slice after being scaled
            to [0, 1] if scale_inputs is True.
        dict_write_step (int): How often to write out dictionary 
            features in terms of the number of forward passes 
            through the model. -1 disables writing dict to disk.
        recon_write_step (int): How often to write out recons in 
            terms of the number of forward passes through the model.
            -1 disables writing recons to disk.
        act_write_step (int): How often to write out feature maps in 
            terms of the number of forward passes through the model.
            -1 disables writing feature maps to disk.
        recon_error_write_step (int): How often to write out x-recon 
            in terms of the number of forward passes through the model.
            -1 disables writing to disk.
        input_write_step (int): How often to write out inputs in terms
            of the number of forward passes through the model. 
            -1 disables writing out inputs to disk.
        tau_decay_factor (float): Each lca loop, tau will start at tau
            and after each iteration will update according to 
            tau -= tau * tau_decay_factor. Empirically helps speed up
            convergence in most cases. Use 0.0 to use constant tau.
        lca_tol (float): Value to determine when to stop LCA loop. 
            if the norm of du across the batch is less than this,
            LCA will terminate during that forward pass. Use 0.0 to 
            disable this and run for lca_iters iterations.
    '''

    def __init__(self, n_neurons, in_c, result_dir, thresh=0.1, tau=1500, 
                 eta=1e-3, lca_iters=3000, pad='same', device=None, 
                 dtype=torch.float32, nonneg=False, learn_dict=True, 
                 track_metrics=True, scale_inputs=True, 
                 zero_center_inputs=True, dict_write_step=-1, 
                 recon_write_step=-1, act_write_step=-1, 
                 recon_error_write_step=-1, input_write_step=-1, 
                 tau_decay_factor=0.0, lca_tol = 0.0):

        self.act_write_step = act_write_step 
        self.device = device 
        self.dict_write_step = dict_write_step
        self.dtype = dtype 
        self.eta = eta 
        self.forward_pass = 0
        self.in_c = in_c 
        self.input_write_step = input_write_step 
        self.lca_iters = lca_iters 
        self.lca_tol = lca_tol
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

        os.makedirs(self.result_dir)
        self.metric_fpath = os.path.join(self.result_dir, 'metrics.xz')
        self.tensor_write_fpath = os.path.join(self.result_dir, 'tensors.h5')

    def create_trackers(self):
        ''' Create placeholders to store different metrics '''

        l1_sparsity = torch.zeros(self.lca_iters, dtype=self.dtype, 
                                  device=self.device)
        l2_error = torch.zeros(self.lca_iters, dtype=self.dtype, 
                               device=self.device)
        du_norm = torch.zeros(self.lca_iters, dtype=self.dtype,
                              device=self.device)
        timestep = np.zeros([self.lca_iters], dtype=np.int64)
        tau_vals = np.zeros([self.lca_iters], dtype=np.float32)

        return {
            'L1': l1_sparsity,
            'L2': l2_error,
            'Timestep': timestep,
            'Tau': tau_vals,
            'duNorm': du_norm
        }

    def encode(self, x):
        ''' Computes sparse code given data x and dictionary D '''

        if self.track_metrics:
            tracks = self.create_trackers()

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
            du = (1 / tau) * (b_t 
                              - u_t 
                              - self.lateral_competition(a_t, G) 
                              + a_t)
            u_t += du
            du_norm = self.compute_du_norm(du)

            if self.track_metrics:
                recon = self.compute_recon(a_t)
                recon_error = x - recon
                tracks = self.update_tracks(tracks, self.ts, tau, du_norm, 
                                            self.compute_l1_sparsity(a_t),
                                            self.compute_l2_error(recon_error), 
                                            lca_iter)

            if du_norm <= self.lca_tol:
                break

            tau = self.update_tau(tau)
            self.ts += 1

        if self.track_metrics:
            self.write_obj_values(tracks, lca_iter+1)
        else:
            recon = self.compute_recon(a_t)
            recon_error = x - recon

        return a_t, recon_error, recon

    def forward(self, x):
        # x is of shape B x C x D x H x W
        x = self.preprocess_inputs(x)
        a, recon_error, recon = self.encode(x)
        if self.learn_dict:
            self.update_D(a, recon_error)

        if self.forward_pass % self.dict_write_step == 0:
            if self.dict_write_step != -1:
                self.write_tensors('D_{}'.format(self.ts), self.D)
        if self.forward_pass % self.act_write_step == 0:
            if self.act_write_step != -1:
                self.write_tensors('a_{}'.format(self.ts), a)
        if self.forward_pass % self.recon_write_step == 0: 
            if self.recon_write_step != -1:
                self.write_tensors('recon_{}'.format(self.ts), recon)
        if self.forward_pass % self.input_write_step == 0: 
            if self.input_write_step != -1:
                self.write_tensors('input_{}'.format(self.ts), x)
        if self.forward_pass % self.recon_error_write_step == 0:
            if self.recon_error_write_step != -1:
                self.write_tensors('recon_error_{}'.format(self.ts), 
                                   recon_error)

        self.forward_pass += 1
        return a

    def soft_threshold(self, x):
        ''' Soft threshold transfer function '''

        if self.nonneg:
            return F.relu(x-self.thresh)
        else:
            return F.relu(x-self.thresh) - F.relu(-x-self.thresh)

    def update_tau(self, tau):
        ''' Update LCA time constant with given decay factor '''

        return tau - tau * self.tau_decay_factor

    def update_tracks(self, tracks, timestep, tau, du_norm, l1, l2, lca_iter):
        ''' Update dictionary that stores the metrics we're tracking '''

        tracks['L2'][lca_iter] = l2
        tracks['L1'][lca_iter] = l1
        tracks['Timestep'][lca_iter] = timestep
        tracks['Tau'][lca_iter] = tau
        tracks['duNorm'][lca_iter] = du_norm
        
        return tracks

    def write_obj_values(self, tracker, ts_cutoff):
        ''' Write out objective values to file '''

        for k,v in tracker.items():
            tracker[k] = v[:ts_cutoff]
            if k in ['L1', 'L2', 'duNorm']:
                tracker[k] = tracker[k].float().cpu().numpy()

        obj_df = pd.DataFrame(tracker)
        obj_df.to_csv(
            self.metric_fpath,
            header=True if not os.path.isfile(self.metric_fpath) else False,
            index=False,
            mode='a'
        )

    def write_tensors(self, key, data):
        ''' Writes out tensors to a HDF5 file. '''

        with h5py.File(self.tensor_write_fpath, 'a') as h5file:
            h5file.create_dataset(key, data=data.cpu().numpy())