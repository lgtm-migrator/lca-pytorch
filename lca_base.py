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
        thresh_type ('hard' or 'soft'): Hard or soft transfer function.
        samplewise_standardization (bool): If True, each sample in the
            batch will be standardized (i.e. mean zero and var 1).
        tau_decay_factor (float): Each lca loop, tau will start at tau
            and after each iteration will update according to 
            tau -= tau * tau_decay_factor. Empirically helps speed up
            convergence in most cases. Use 0.0 to use constant tau.
        lca_tol (float): Value to determine when to stop LCA loop. 
            If the value of the objective function is less than this,
            LCA will terminate during that forward pass. Use None to 
            disable this and run for lca_iters iterations.
        d_update_clip (float): Dictionary updates will be clipped to
            [-d_update_clip, d_update_clip].
        dict_load_fpath (str): Path to the tensors.h5 file with at 
            least 1 key starting with 'D_', which will be used to
            load in the dictionary tensor from the latest ckpt.
        keep_solution (bool): If True, the LCA membrane potentials 
            for training batch i will be initialized with those found
            on training batch i-1. This can sometimes allows for faster
            convergence when iterating sequentially over video data.
        lca_write_step (int): How often to write out a_t, u_t, b_t,
            recon, and recon_error within a single LCA loop. If None,
            these will not be written to disk.
        forward_write_step (int): How often to write out dictionary,
            input, and dictionary update. If None, these will not be
            written to disk.
    '''
    def __init__(self, n_neurons, in_c, result_dir, thresh=0.1, tau=1500, 
                 eta=1e-3, lca_iters=3000, pad='same', device=None, 
                 dtype=torch.float32, nonneg=False, learn_dict=True, 
                 track_metrics=True, thresh_type='hard',
                 samplewise_standardization=True, tau_decay_factor=0.0, 
                 lca_tol=None, cudnn_benchmark=False, d_update_clip=np.inf,
                 dict_load_fpath=None, keep_solution=False,
                 lca_write_step=None, forward_write_step=None):

        self.d_update_clip = d_update_clip
        self.device = device 
        self.dict_load_fpath = dict_load_fpath
        self.dtype = dtype 
        self.eta = eta 
        self.forward_pass = 0
        self.forward_write_step = forward_write_step
        self.in_c = in_c 
        self.keep_solution = keep_solution
        self.lca_iters = lca_iters 
        self.lca_tol = lca_tol
        self.lca_write_step = lca_write_step
        self.learn_dict = learn_dict
        self.n_neurons = n_neurons 
        self.nonneg = nonneg 
        self.pad = pad
        self.result_dir = result_dir 
        self.samplewise_standardization = samplewise_standardization
        self.tau = tau 
        self.tau_decay_factor = tau_decay_factor
        self.thresh = thresh 
        self.thresh_type = thresh_type
        self.track_metrics = track_metrics
        self.ts = 0

        if cudnn_benchmark and torch.backends.cudnn.enabled: 
            torch.backends.cudnn.benchmark = True

        os.makedirs(self.result_dir, exist_ok=True)
        self.metric_fpath = os.path.join(self.result_dir, 'metrics.xz')
        self.tensor_write_fpath = os.path.join(self.result_dir, 'tensors.h5')

    def compute_n_surround(self, G):
        ''' Computes the number of surround neurons for each dim '''
        G_shp = G.shape[2:]
        self.surround = tuple([int(np.ceil((dim - 1) / 2)) for dim in G_shp])

    def compute_times_active_by_feature(self, x):
        ''' Computes number of active coefficients per feature '''
        dims = list(range(len(x.shape)))
        dims.remove(1)
        times_active = (x != 0).float().sum(dim=dims) + 1
        return times_active.reshape((x.shape[1],) + (1,) * len(dims))

    def compute_l1_sparsity(self, acts):
        ''' Compute l1 sparsity term of objective function '''
        dims = tuple(range(1, len(acts.shape)))
        return self.thresh * acts.norm(p=1, dim=dims).mean()

    def compute_l2_error(self, error):
        ''' Compute l2 recon error term of objective function '''
        dims = tuple(range(1, len(error.shape)))
        return 0.5 * (error.norm(p=2, dim=dims) ** 2).mean()

    def compute_perc_change(self, curr, prev):
        ''' Computes percent change of a value from t-1 to t '''
        return ((curr - prev) / prev).abs()

    def create_trackers(self):
        ''' Create placeholders to store different metrics '''
        l1_sparsity = torch.zeros(self.lca_iters, dtype=self.dtype, 
                                  device=self.device)
        l2_error = torch.zeros(self.lca_iters, dtype=self.dtype, 
                               device=self.device)
        energy = torch.zeros(self.lca_iters, dtype=self.dtype, 
                             device=self.device)
        timestep = np.zeros([self.lca_iters], dtype=np.int64)
        tau_vals = np.zeros([self.lca_iters], dtype=np.float32)
        return {
            'L1': l1_sparsity,
            'L2': l2_error,
            'TotalEnergy': energy,
            'Timestep': timestep,
            'Tau': tau_vals
        }

    def encode(self, x):
        ''' Computes sparse code given data x and dictionary D '''
        b_t = self.compute_input_drive(x) 
        G = self.compute_lateral_connectivity()
        tau = self.tau 
        if not self.keep_solution or self.forward_pass == 0:
            self.u_t = torch.zeros_like(b_t)

        for lca_iter in range(self.lca_iters):
            a_t = self.threshold(self.u_t)
            inhib = self.lateral_competition(a_t, G)
            du = (1 / tau) * (b_t - self.u_t - inhib + a_t)
            self.u_t += du

            if (self.track_metrics 
                    or lca_iter == self.lca_iters - 1
                    or self.lca_tol is not None
                    or (self.lca_write_step is not None 
                        and lca_iter % self.lca_write_step == 0)):
                recon = self.compute_recon(a_t)
                recon_error = x - recon
                if (self.lca_write_step is not None 
                        and lca_iter % self.lca_write_step == 0):
                    self.write_tensors(
                        ['a', 'b', 'u', 'recon', 'recon_error'],
                        [a_t, b_t, self.u_t, recon, recon_error],
                        lca_iter=lca_iter
                    )
                if self.track_metrics or self.lca_tol is not None:
                    l2_rec_err = self.compute_l2_error(recon_error)
                    l1_sparsity = self.compute_l1_sparsity(a_t)
                    total_energy = l2_rec_err + l1_sparsity
                    if lca_iter == 0:
                        tracks = self.create_trackers()
                    tracks = self.update_tracks(tracks, self.ts, tau, 
                                                l1_sparsity, l2_rec_err,
                                                total_energy, lca_iter)
                    if self.lca_tol is not None:
                        if lca_iter > 100:
                            if self.stop_lca(tracks['TotalEnergy'], lca_iter):
                                break

            tau = self.update_tau(tau)
            self.ts += 1 

        if self.track_metrics:
            self.write_tracks(tracks, lca_iter + 1)
        return a_t, recon_error, recon

    def forward(self, x):
        # x is of shape B x C x D x H x W
        if self.samplewise_standardization:
            x = self.standardize_inputs(x)

        a, recon_error, recon = self.encode(x)
        if self.learn_dict:
            update = self.update_D(a, recon_error)

        if (self.forward_write_step is not None
                and self.forward_pass % self.forward_write_step == 0):
            self.write_tensors(['D', 'input', 'update'], [self.D, x, update])

        self.forward_pass += 1
        return a

    def hard_threshold(self, x):
        ''' Hard threshold transfer function '''
        if self.nonneg:
            return F.threshold(x, self.thresh, 0.0)
        else:
            return (F.threshold(x, self.thresh, 0.0) 
                    - F.threshold(-x, self.thresh, 0.0))

    def init_weight_tensor(self):
        if self.dict_load_fpath is None:
            self.create_weight_tensor()
        else:
            self.load_weight_tensor()

    def load_weight_tensor(self):
        ''' Loads in dictionary from latest ckpt in result file '''
        self.create_weight_tensor()
        with h5py.File(self.dict_load_fpath, 'r') as h5f:
            h5keys = list(h5f.keys())
            Dkeys = [key for key in h5keys if 'D_' in key]
            ckpt_nums = sorted([int(key.split('_')[-1]) for key in Dkeys])
            dict = h5f[f'D_{ckpt_nums[-1]}'][()]
            assert dict.shape == self.D.shape 
            self.D = torch.from_numpy(dict).type(self.dtype).to(self.device)
            self.normalize_D()

    def normalize_D(self, eps=1e-12):
        ''' Normalizes features such at each one has unit norm '''
        dims = tuple(range(1, len(self.D.shape)))
        scale = self.D.norm(p=2, dim=dims, keepdim=True)
        self.D = self.D / (scale + eps)

    def soft_threshold(self, x):
        ''' Soft threshold transfer function '''
        if self.nonneg:
            return F.relu(x - self.thresh)
        else:
            return F.relu(x - self.thresh) - F.relu(-x - self.thresh)

    def standardize_inputs(self, batch, eps=1e-12):
        ''' Standardize each sample in x '''
        dims = tuple(range(1, len(batch.shape)))
        batch = batch - batch.mean(dim=dims, keepdim=True)
        batch = batch / (batch.std(dim=dims, keepdim=True) + eps)
        return batch

    def stop_lca(self, energy_history, lca_iter):
        ''' Determines when to stop LCA loop early by comparing the 
            percent change between a running avg of the objective value 
            at time t and that at time t-1 and checking if it is less
            then the user-defined lca_tol value '''
        curr_avg = energy_history[lca_iter - 99 : lca_iter + 1].mean()
        prev_avg = energy_history[lca_iter - 100 : lca_iter].mean()
        perc_change = self.compute_perc_change(curr_avg, prev_avg)
        if perc_change < self.lca_tol:
            return True 
        else:
            return False

    def threshold(self, x):
        if self.thresh_type == 'soft':
            return self.soft_threshold(x)
        elif self.thresh_type == 'hard': 
            return self.hard_threshold(x)
        else:
            raise ValueError

    def update_D(self, a, recon_error):
        ''' Updates the dictionary given the computed gradient '''
        update = self.compute_update(a, recon_error)
        times_active = self.compute_times_active_by_feature(a)
        update *= (self.eta / times_active)
        update = torch.clamp(update, min=-self.d_update_clip, 
                             max=self.d_update_clip)
        self.D += update
        self.normalize_D()
        return update

    def update_tau(self, tau):
        ''' Update LCA time constant with given decay factor '''
        return tau - tau * self.tau_decay_factor

    def update_tracks(self, tracks, timestep, tau, l1, l2, energy, lca_iter):
        ''' Update dictionary that stores the tracked metrics '''
        tracks['L2'][lca_iter] = l2
        tracks['L1'][lca_iter] = l1
        tracks['TotalEnergy'][lca_iter] = energy
        tracks['Timestep'][lca_iter] = timestep
        tracks['Tau'][lca_iter] = tau
        return tracks

    def write_tracks(self, tracker, ts_cutoff):
        ''' Write out objective values to file '''
        for k,v in tracker.items():
            tracker[k] = v[:ts_cutoff]
            if k in ['L1', 'L2', 'TotalEnergy']:
                tracker[k] = tracker[k].float().cpu().numpy()

        obj_df = pd.DataFrame(tracker)
        obj_df['ForwardPass'] = self.forward_pass
        obj_df.to_csv(
            self.metric_fpath,
            header=True if not os.path.isfile(self.metric_fpath) else False,
            index=False,
            mode='a'
        )

    def write_tensors(self, keys, tensors, lca_iter=0):
        ''' Writes out tensors to a HDF5 file. '''
        with h5py.File(self.tensor_write_fpath, 'a') as h5file:
            for key, tensor in zip(keys, tensors):
                h5file.create_dataset(key + f'_{self.forward_pass}_{lca_iter}',
                                      data=tensor.cpu().numpy())