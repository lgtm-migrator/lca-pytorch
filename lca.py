from copy import deepcopy
import os
from typing import Callable, Optional, Union
import yaml

import h5py
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F


Parameter = torch.nn.parameter.Parameter
Tensor = torch.Tensor


class LCAConv(torch.nn.Module):
    '''
    Base class for LCA models.

    Required Args:
        n_neurons (int): Number of dictionary elements.
        in_c (int): Number of channels / features in the input data.
        result_dir (str): Path to dir where results will be saved.

    Optional Args:
        kh (int): Kernel height of the convolutional features.
        kw (int): Kernel width of the convolutional features.
        kt (int): Kernel depth of the convolutional features. For 1D
            (e.g. raw audio) or 2D (e.g. images) inputs, keep this at
            the default of 1.
        stride_h (int): Vertical stride of each feature.
        stride_w (int): Horizontal stride of each feature.
        stride_t (int): Stride in depth (time) of each feature.
        thresh (float): Threshold for the LCA transfer function.
        tau (int): LCA time constant. 
        eta (float): Learning rate for dictionary updates.
        lca_iters (int): Number of LCA timesteps per forward pass.
        pad ('same' or 'valid'): Input padding method.
        return_recon (bool): If True, calling forward will return code,
            recon, and input - recon. If False, will just return code.
        device (int, list, or 'cpu'): Device(s) to use.
        dtype (torch.dtype): Data type to use.
        nonneg (bool): True for rectified activations, False for 
            non-rectified activations.
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
            If a running average of the objective function changes less
            than lca_tol from one LCA iteration to the next, the LCA
            loop will be terminated. Use None to disable this and run
            for lca_iters iterations.
        d_update_clip (float): Dictionary updates will be clipped to
            [-d_update_clip, d_update_clip]. Default is no clipping.
        lr_schedule (function): Function which takes the training step
            as input and returns a value for eta.
        lca_write_step (int): How often to write out a_t, u_t, b_t,
            recon, and recon_error within a single LCA loop. If None,
            these will not be written to disk.
        req_grad (bool): If True, dictionary D will have
            requires_grad set to True. Otherwise, it will be False.
            This is useful for propagating gradient through the LCA
            layer (e.g. for adversarial attacks).
        forward_write_step (int): How often to write out dictionary,
            input, and dictionary update. If None, these will not be
            written to disk.
    '''
    def __init__(
        self,
        n_neurons: int,
        in_c: int,
        result_dir: str,
        kh: int = 7,
        kw: int = 7,
        kt: int = 1,
        stride_h: int = 1,
        stride_w: int = 1,
        stride_t: int = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 1000,
        eta: float = 0.01,
        lca_iters: int = 3000,
        pad: str = 'same',
        return_recon: bool = False,
        dtype: torch.dtype = torch.float32,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            str, Callable[[Tensor], Tensor]] = 'soft_threshold',
        samplewise_standardization: bool = True,
        tau_decay_factor: float = 0.0,
        lca_tol: Optional[float] = None,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        lca_write_step: Optional[int] = None,
        forward_write_step: Optional[int] = None,
        req_grad: bool = False,
    ) -> None:

        self.d_update_clip = d_update_clip
        self.dtype = dtype 
        self.eta = eta 
        self.forward_write_step = forward_write_step
        self.in_c = in_c 
        self.kernel_odd = True if kh % 2 != 0 else False
        self.kh = kh
        self.kt = kt
        self.kw = kw
        self.lambda_ = lambda_
        self.lca_iters = lca_iters 
        self.lca_tol = lca_tol
        self.lca_warmup = tau // 10 + 100
        self.lca_write_step = lca_write_step
        if lr_schedule is not None: assert callable(lr_schedule)
        self.lr_schedule = lr_schedule
        self.metric_fpath = os.path.join(result_dir, 'metrics.xz')
        self.n_neurons = n_neurons 
        self.nonneg = nonneg 
        self.pad = pad
        self.req_grad = req_grad
        self.result_dir = result_dir 
        self.return_recon = return_recon
        self.samplewise_standardization = samplewise_standardization
        self.stride_h = stride_h
        self.stride_t = stride_t
        self.stride_w = stride_w
        self.tau = tau 
        self.tau_decay_factor = tau_decay_factor
        self.tensor_write_fpath = os.path.join(result_dir, 'tensors.h5')
        self.track_metrics = track_metrics
        self.transfer_func = transfer_func

        self._check_conv_params()
        self._compute_padding()
        os.makedirs(self.result_dir, exist_ok=True)
        self.write_params(deepcopy(vars(self)))
        super(LCAConv, self).__init__()
        self.init_weight_tensor()
        self.register_buffer('forward_pass', torch.tensor(1))

        if cudnn_benchmark and torch.backends.cudnn.enabled: 
            torch.backends.cudnn.benchmark = True

    def _check_lca_write(self, lca_iter):
        ''' Checks whether to write LCA tensors at a given LCA iter '''
        write = False
        if self.lca_write_step is not None:
            if lca_iter % self.lca_write_step == 0:
                if self._check_forward_write():
                    write = True
        return write

    def _check_forward_write(self):
        ''' Checks whether to write non-LCA-loop variables at a given
            forward pass '''
        write = False
        if self.forward_write_step is not None:
            if self.forward_pass % self.forward_write_step == 0:
                write = True
        return write

    def _check_conv_params(self):
        assert ((self.kh % 2 != 0 and self.kw % 2 != 0)
                or (self.kh % 2 == 0 and self.kw % 2 == 0)), (
                'kh and kw should either both be even or both be odd numbers, '
                f'but kh={self.kh} and kw={self.kw}.')
        assert self.stride_h == 1 or self.stride_h % 2 == 0
        assert self.stride_w == 1 or self.stride_w % 2 == 0

    def _compute_inhib_pad(self):
        ''' Computes padding for compute_lateral_connectivity '''
        self.lat_conn_pad = [0]

        if self.kernel_odd or self.stride_h == 1:
            self.lat_conn_pad.append((self.kh - 1)
                                     // self.stride_h
                                     * self.stride_h)
        else:
            self.lat_conn_pad.append(self.kh - self.stride_h)

        if self.kernel_odd or self.stride_w == 1:
            self.lat_conn_pad.append((self.kw - 1)
                                     // self.stride_w
                                     * self.stride_w)
        else:
            self.lat_conn_pad.append(self.kw - self.stride_w)

        self.lat_conn_pad = tuple(self.lat_conn_pad)

    def _compute_input_pad(self):
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
            raise ValueError("Values for pad can either be 'same' or 'valid', "
                             f"but got {self.pad}.")

    def _compute_padding(self):
        self._compute_input_pad()
        self._compute_inhib_pad()
        self._compute_recon_pad()

    def _compute_recon_pad(self):
        ''' Computes output padding for recon conv transpose '''
        if self.kernel_odd:
            self.recon_output_pad = (0, self.stride_h - 1, self.stride_w - 1)
        else:
            self.recon_output_pad = (0, 0, 0)

    def compute_frac_active(self, acts: Tensor) -> float:
        ''' Computes the number of active neurons relative to the total
            number of neurons '''
        return (acts != 0.0).float().mean().item()

    def compute_input_drive(self, inputs: Tensor,
                            weights: Union[Tensor, Parameter]) -> Tensor:
        assert inputs.shape[2] == self.kt
        return F.conv3d(inputs, weights,
                        stride=(self.stride_t, self.stride_h, self.stride_w),
                        padding=self.input_pad)

    def compute_l1_sparsity(self, acts: Tensor) -> Tensor:
        ''' Compute l1 sparsity term of objective function '''
        dims = tuple(range(1, len(acts.shape)))
        return self.lambda_ * acts.norm(p=1, dim=dims).mean()

    def compute_l2_error(self, error: Tensor) -> Tensor:
        ''' Compute l2 recon error term of objective function '''
        dims = tuple(range(1, len(error.shape)))
        return 0.5 * (error.norm(p=2, dim=dims) ** 2).mean()

    def compute_lateral_connectivity(
            self, weights: Union[Tensor, Parameter]) -> Tensor:
        conns = F.conv3d(weights, weights,
                         stride=(self.stride_t, self.stride_h, self.stride_w),
                         padding=self.lat_conn_pad)
        if not hasattr(self, 'surround'):
            self.compute_n_surround(conns)
        return conns

    def compute_n_surround(self, conns: Tensor) -> tuple:
        ''' Computes the number of surround neurons for each dim '''
        conn_shp = conns.shape[2:]
        self.surround = tuple(
            [int(np.ceil((dim - 1) / 2)) for dim in conn_shp])

    def compute_perc_change(self, curr, prev):
        ''' Computes percent change of a value from t-1 to t '''
        return abs((curr - prev) / prev)

    def compute_recon(self, acts: Tensor,
                      weights: Union[Tensor, Parameter]) -> Tensor:
        ''' Computes reconstruction given code '''
        return F.conv_transpose3d(
            acts,
            weights,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad,
            output_padding=self.recon_output_pad)

    def compute_times_active_by_feature(self, acts: Tensor) -> Tensor:
        ''' Computes number of active coefficients per feature '''
        dims = list(range(len(acts.shape)))
        dims.remove(1)
        times_active = (acts != 0).float().sum(dim=dims)
        return times_active.reshape((acts.shape[1],) + (1,) * len(dims))

    def compute_update(self, acts, error):
        error = F.pad(error, (self.input_pad[2], self.input_pad[2],
                              self.input_pad[1], self.input_pad[1],
                              self.input_pad[0], self.input_pad[0]
            ))
        error = error.unfold(-3, self.kt, self.stride_t)
        error = error.unfold(-3, self.kh, self.stride_h)
        error = error.unfold(-3, self.kw, self.stride_w)
        return torch.tensordot(acts, error, dims=([0, 2, 3, 4], [0, 2, 3, 4]))

    def create_trackers(self):
        ''' Create placeholders to store different metrics '''
        float_tracker = np.zeros([self.lca_iters], dtype=np.float32)
        return {
            'L1' : float_tracker.copy(),
            'L2' : float_tracker.copy(),
            'TotalEnergy' : float_tracker.copy(),
            'FractionActive' : float_tracker.copy(),
            'Tau' : float_tracker.copy()
        }

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ''' Computes sparse code given data x and dictionary D '''
        input_drive = self.compute_input_drive(inputs, self.weights)
        states = torch.zeros_like(input_drive, requires_grad=self.req_grad)
        connectivity = self.compute_lateral_connectivity(self.weights)
        tau = self.tau

        for lca_iter in range(1, self.lca_iters + 1):
            acts = self.transfer(states)
            inhib = self.lateral_competition(acts, connectivity)
            states = states + (1 / tau) * (input_drive - states - inhib + acts)

            if (self.track_metrics 
                    or lca_iter == self.lca_iters
                    or self.lca_tol is not None
                    or self._check_lca_write(lca_iter)):
                recon = self.compute_recon(acts, self.weights)
                recon_error = inputs - recon
                if self._check_lca_write(lca_iter):
                    self.write_tensors({
                        'acts': acts,
                        'input_drive': input_drive,
                        'input': inputs,
                        'states': states,
                        'recon': recon,
                        'recon_error': recon_error,
                        'lateral_connectivity': connectivity
                    }, lca_iter)
                if self.track_metrics or self.lca_tol is not None:
                    if lca_iter == 1:
                        tracks = self.create_trackers()
                    tracks = self.update_tracks(tracks, lca_iter, acts,
                                                recon_error, tau)
                    if self.lca_tol is not None:
                        if lca_iter > self.lca_warmup:
                            if self.stop_lca(tracks['TotalEnergy'], lca_iter):
                                break

            tau = self.update_tau(tau) 

        if self.track_metrics:
            self.write_tracks(tracks, lca_iter, inputs.device.index)
        return acts, recon, recon_error

    def forward(self, inputs: Tensor) -> Union[
            Tensor, tuple[Tensor, Tensor, Tensor]]:
        if self.samplewise_standardization:
            inputs = self.standardize_inputs(inputs)
        acts, recon, recon_error = self.encode(inputs)
        self.forward_pass += 1
        if self.return_recon:
            return acts, recon, recon_error
        else:
            return acts

    def hard_threshold(self, x: Tensor) -> Tensor:
        ''' Hard threshold transfer function '''
        if self.nonneg:
            return F.threshold(x, self.lambda_, 0.0)
        else:
            return (F.threshold(x, self.lambda_, 0.0) 
                    - F.threshold(-x, self.lambda_, 0.0))

    def init_weight_tensor(self):
        weights = torch.randn(self.n_neurons, self.in_c, self.kt, self.kh,
                              self.kw, dtype=self.dtype)
        weights[:, :, 1:] = 0.0
        self.weights = torch.nn.Parameter(weights, requires_grad=self.req_grad)
        self.normalize_weights()

    def lateral_competition(self, acts: Tensor, conns: Tensor) -> Tensor:
        return F.conv3d(acts, conns, stride=1, padding=self.surround)

    def normalize_weights(self, eps=1e-6):
        ''' Normalizes features such at each one has unit norm '''
        with torch.no_grad():
            dims = tuple(range(1, len(self.weights.shape)))
            scale = self.weights.norm(p=2, dim=dims, keepdim=True)
            self.weights.copy_(self.weights / (scale + eps))

    def soft_threshold(self, x: Tensor) -> Tensor:
        ''' Soft threshold transfer function '''
        if self.nonneg:
            return F.relu(x - self.lambda_)
        else:
            return F.relu(x - self.lambda_) - F.relu(-x - self.lambda_)

    def standardize_inputs(self, batch: Tensor, eps: float = 1e-6) -> Tensor:
        ''' Standardize each sample in x '''
        if len(batch.shape) == 3:
            dims = -1
        elif len(batch.shape) in [4, 5]:
            dims = (-2, -1)
        else:
            raise NotImplementedError
        batch = batch - batch.mean(dim=dims, keepdim=True)
        batch = batch / (batch.std(dim=dims, keepdim=True) + eps)
        return batch

    def stop_lca(self, energy_history, lca_iter):
        ''' Determines when to stop LCA loop early by comparing the 
            percent change between a running avg of the objective value 
            at time t and that at time t-1 and checking if it is less
            then the user-defined lca_tol value '''
        curr_avg = energy_history[lca_iter - 100 : lca_iter].mean()
        prev_avg = energy_history[lca_iter - 101 : lca_iter - 1].mean()
        perc_change = self.compute_perc_change(curr_avg, prev_avg)
        if perc_change < self.lca_tol:
            return True 
        else:
            return False

    def transfer(self, x: Tensor) -> Tensor:
        if type(self.transfer_func) == str:
            if self.transfer_func == 'soft_threshold':
                return self.soft_threshold(x)
            elif self.transfer_func == 'hard_threshold':
                return self.hard_threshold(x)
            else:
                raise ValueError
        elif callable(self.transfer_func):
            return self.transfer_func(x)

    def update(self, acts: Tensor, recon_error: Tensor) -> None:
        ''' Updates the dictionary given the computed gradient '''
        with torch.no_grad():
            update = self.compute_update(acts, recon_error)
            times_active = self.compute_times_active_by_feature(acts) + 1
            update *= (self.eta / times_active)
            update = torch.clamp(update, min=-self.d_update_clip,
                                 max=self.d_update_clip)
            self.weights.copy_(self.weights + update)
            self.normalize_weights()
            if self.lr_schedule is not None:
                self.eta = self.lr_schedule(self.forward_pass)
            if self._check_forward_write():
                self.write_tensors({'weight_update': update})

    def update_tau(self, tau: Union[int, float]) -> float:
        ''' Update LCA time constant with given decay factor '''
        return tau - tau * self.tau_decay_factor

    def update_tracks(self, tracks, lca_iter, acts, recon_error, tau):
        ''' Update dictionary that stores the tracked metrics '''
        l2_rec_err = self.compute_l2_error(recon_error).item()
        l1_sparsity = self.compute_l1_sparsity(acts).item()
        tracks['L2'][lca_iter - 1] = l2_rec_err
        tracks['L1'][lca_iter - 1] = l1_sparsity
        tracks['TotalEnergy'][lca_iter - 1] = l2_rec_err + l1_sparsity
        tracks['FractionActive'][lca_iter - 1] = self.compute_frac_active(acts)
        tracks['Tau'][lca_iter - 1] = tau
        return tracks

    def write_params(self, arg_dict):
        ''' Writes model params to file '''
        arg_dict['dtype'] = str(arg_dict['dtype'])
        del arg_dict['lr_schedule']
        if callable(self.transfer_func):
            arg_dict['transfer_func'] = self.transfer_func.__name__
        with open(os.path.join(self.result_dir, 'params.yaml'), 'w') as yamlf:
            yaml.dump(arg_dict, yamlf, sort_keys=True)

    def write_tracks(self, tracker, ts_cutoff, dev):
        ''' Write out objective values to file '''
        for k,v in tracker.items():
            tracker[k] = v[:ts_cutoff]

        obj_df = pd.DataFrame(tracker)
        obj_df['LCAIter'] = np.arange(1, len(obj_df) + 1, dtype=np.int32)
        obj_df['ForwardPass'] = self.forward_pass.item()
        obj_df['Device'] = dev
        obj_df.to_csv(
            self.metric_fpath,
            header=True if not os.path.isfile(self.metric_fpath) else False,
            index=False,
            mode='a')

    def write_tensors(self, tensor_dict: dict[str, Tensor],
                      lca_iter: int = 0) -> None:
        ''' Writes out tensors to a HDF5 file. '''
        with h5py.File(self.tensor_write_fpath, 'a') as h5file:
            for name, tensor in tensor_dict.items():
                h5file.create_dataset(
                    f'{name}_{self.forward_pass}_{lca_iter}',
                    data=tensor.detach().cpu().numpy())
