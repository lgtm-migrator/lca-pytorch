from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Callable, Iterable, Optional, Union
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .activation import hard_threshold, soft_threshold
from .metric import (
    compute_frac_active,
    compute_l1_sparsity,
    compute_l2_error,
    compute_times_active_by_feature,
)
from .preproc import make_unit_var, make_zero_mean
from .util import to_3d_from_5d, to_4d_from_5d, to_5d_from_3d, to_5d_from_4d

try:
    from typing import Literal  # python >= 3.8
except ImportError:
    from typing_extensions import Literal  # python < 3.8


Parameter = torch.nn.parameter.Parameter
Tensor = torch.Tensor


class _LCAConvBase(torch.nn.Module):
    """
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
        d_update_clip (float): Dictionary updates will be clipped to
            [-d_update_clip, d_update_clip]. Default is no clipping.
        lr_schedule (function): Function which takes the training step
            as input and returns a value for eta.
        req_grad (bool): If True, dictionary D will have
            requires_grad set to True. Otherwise, it will be False.
            This is useful for propagating gradient through the LCA
            layer (e.g. for adversarial attacks).
    """

    def __init__(
        self,
        n_neurons: int,
        in_c: int,
        result_dir: str,
        kernel_size: tuple[int, int, int] = (1, 7, 7),
        stride: tuple[int, int, int] = (1, 1, 1),
        lambda_: float = 0.25,
        tau: Union[float, int] = 1000,
        eta: float = 0.01,
        lca_iters: int = 3000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        dtype: torch.dtype = torch.float32,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
        no_time_pad: bool = False,
    ) -> None:

        self.d_update_clip = d_update_clip
        self.dtype = dtype
        self.eta = eta
        self.in_c = in_c
        self.input_unit_var = input_unit_var
        self.input_zero_mean = input_zero_mean
        self.kh = kernel_size[1]
        self.kt = kernel_size[0]
        self.kw = kernel_size[2]
        self.lambda_ = lambda_
        self.lca_iters = lca_iters
        if lr_schedule is not None:
            assert callable(lr_schedule)
        self.lr_schedule = lr_schedule
        self.metric_fpath = os.path.join(result_dir, "metrics.xz")
        self.n_neurons = n_neurons
        self.no_time_pad = no_time_pad
        self.nonneg = nonneg
        self.pad = pad
        self.req_grad = req_grad
        self.result_dir = result_dir
        self.return_all_ts = return_all_ts
        self.return_vars = return_vars
        self.stride_h = stride[1]
        self.stride_t = stride[0]
        self.stride_w = stride[2]
        self.tau = tau
        self.track_metrics = track_metrics
        self.transfer_func = transfer_func
        self.return_var_names = [
            "inputs",
            "input_drives",
            "states",
            "acts",
            "recons",
            "recon_errors",
            "conns",
        ]

        self._check_return_vars()
        self._check_conv_params()
        self._compute_padding()
        os.makedirs(self.result_dir, exist_ok=True)
        self._write_params(deepcopy(vars(self)))
        super(_LCAConvBase, self).__init__()
        self._init_weight_tensor()
        self.register_buffer("forward_pass", torch.tensor(1))

        if cudnn_benchmark and torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

    def assign_weight_values(self, tensor: Tensor) -> None:
        """Manually assign weight tensor"""
        with torch.no_grad():
            assert tensor.dtype == self.weights.dtype
            tensor, _ = self._to_correct_input_shape(tensor)
            assert tensor.shape == self.weights.shape
            self.weights.copy_(tensor)

    def _check_conv_params(self) -> None:
        even_k = [ksize % 2 == 0 for ksize in [self.kt, self.kh, self.kw] if ksize != 1]
        assert all(even_k) or not any(even_k)
        self.kernel_odd = not any(even_k)

    def _check_return_vars(self) -> None:
        if type(self.return_vars) not in [list, tuple]:
            raise TypeError(
                f"return_vars should be list or tuple, but got {type(self.return_vars)}."
            )

        for var_name in self.return_vars:
            if var_name not in self.return_var_names:
                raise ValueError(
                    f"Name '{var_name}' in return_vars is not in {self.return_var_names}."
                )

    def _compute_inhib_pad(self) -> None:
        """Computes padding for compute_lateral_connectivity"""
        pad = []
        for ksize, stride in zip(
            [self.kt, self.kh, self.kw], [self.stride_t, self.stride_h, self.stride_w]
        ):
            if ksize % 2 != 0:
                pad.append((ksize - 1) // stride * stride)
            else:
                if ksize % stride == 0:
                    pad.append(ksize - stride)
                else:
                    pad.append(ksize // stride * stride)

        self.lat_conn_pad = tuple(pad)

    def _compute_input_pad(self) -> None:
        """Computes padding for forward convolution"""
        if self.pad == "same":
            assert self.kernel_odd
            self.input_pad = (
                0 if self.no_time_pad else self.kt // 2,
                self.kh // 2,
                self.kw // 2,
            )
        elif self.pad == "valid":
            self.input_pad = (0, 0, 0)
        else:
            raise ValueError(
                "Acceptable values for pad are 'same' and 'valid', but got ",
                f"{self.pad}.",
            )

    def _compute_padding(self) -> None:
        self._compute_input_pad()
        self._compute_inhib_pad()
        self._compute_recon_pad()

    def _compute_recon_pad(self) -> None:
        """Computes output padding for recon conv transpose"""
        if self.kernel_odd:
            self.recon_output_pad = (
                self.stride_t - 1,
                self.stride_h - 1,
                self.stride_w - 1,
            )
        else:
            self.recon_output_pad = (0, 0, 0)

    def compute_input_drive(
        self, inputs: Tensor, weights: Union[Tensor, Parameter]
    ) -> Tensor:
        inputs, reshape_func = self._to_correct_input_shape(inputs)
        drive = F.conv3d(
            inputs,
            weights,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad,
        )
        return reshape_func(drive)

    def compute_lateral_connectivity(self, weights: Union[Tensor, Parameter]) -> Tensor:
        conns = F.conv3d(
            weights,
            weights,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.lat_conn_pad,
        )
        if not hasattr(self, "surround"):
            self._compute_n_surround(conns)
        return conns

    def _compute_n_surround(self, conns: Tensor) -> tuple:
        """Computes the number of surround neurons for each dim"""
        conn_shp = conns.shape[2:]
        self.surround = tuple([int(np.ceil((dim - 1) / 2)) for dim in conn_shp])

    def compute_recon(self, acts: Tensor, weights: Union[Tensor, Parameter]) -> Tensor:
        """Computes reconstruction given code"""
        acts, reshape_func = self._to_correct_input_shape(acts)
        recons = F.conv_transpose3d(
            acts,
            weights,
            stride=(self.stride_t, self.stride_h, self.stride_w),
            padding=self.input_pad,
            output_padding=self.recon_output_pad,
        )
        return reshape_func(recons)

    def compute_weight_update(self, acts: Tensor, error: Tensor) -> Tensor:
        error = F.pad(
            error,
            (
                self.input_pad[2],
                self.input_pad[2],
                self.input_pad[1],
                self.input_pad[1],
                self.input_pad[0],
                self.input_pad[0],
            ),
        )
        error = error.unfold(-3, self.kt, self.stride_t)
        error = error.unfold(-3, self.kh, self.stride_h)
        error = error.unfold(-3, self.kw, self.stride_w)
        return torch.tensordot(acts, error, dims=([0, 2, 3, 4], [0, 2, 3, 4]))

    def _create_trackers(self) -> dict[str, np.ndarray]:
        """Create placeholders to store different metrics"""
        float_tracker = np.zeros([self.lca_iters], dtype=np.float32)
        return {
            "L1": float_tracker.copy(),
            "L2": float_tracker.copy(),
            "TotalEnergy": float_tracker.copy(),
            "FractionActive": float_tracker.copy(),
            "Tau": float_tracker.copy(),
        }

    def encode(self, inputs: Tensor) -> tuple[list[Tensor], ...]:
        """Computes sparse code given data x and dictionary D"""
        input_drive = self.compute_input_drive(inputs, self.weights)
        states = torch.zeros_like(input_drive, requires_grad=self.req_grad)
        connectivity = self.compute_lateral_connectivity(self.weights)
        tau = self.tau

        return_vars = tuple([[] for _ in range(len(self.return_vars))])

        for lca_iter in range(1, self.lca_iters + 1):
            acts = self.transfer(states)
            inhib = self.lateral_competition(acts, connectivity)
            states = states + (1 / tau) * (input_drive - states - inhib + acts)

            if self.track_metrics or lca_iter == self.lca_iters or self.return_all_ts:
                recon = self.compute_recon(acts, self.weights)
                recon_error = inputs - recon

                if self.return_all_ts or lca_iter == self.lca_iters:
                    for var_idx, var_name in enumerate(self.return_vars):
                        if var_name == "inputs":
                            return_vars[var_idx].append(inputs)
                        elif var_name == "input_drives":
                            return_vars[var_idx].append(input_drive)
                        elif var_name == "states":
                            return_vars[var_idx].append(states)
                        elif var_name == "acts":
                            return_vars[var_idx].append(acts)
                        elif var_name == "recons":
                            return_vars[var_idx].append(recon)
                        elif var_name == "recon_errors":
                            return_vars[var_idx].append(recon_error)
                        elif var_name == "conns":
                            return_vars[var_idx].append(connectivity)

                if self.track_metrics:
                    if lca_iter == 1:
                        tracks = self._create_trackers()
                    tracks = self._update_tracks(
                        tracks, lca_iter, acts, inputs, recon, tau
                    )

        if self.track_metrics:
            self._write_tracks(tracks, lca_iter, inputs.device.index)

        return return_vars

    def forward(self, inputs: Tensor) -> Union[Tensor, tuple[Tensor, ...]]:
        if self.input_zero_mean:
            inputs = make_zero_mean(inputs)
        if self.input_unit_var:
            inputs = make_unit_var(inputs)

        inputs, reshape_func = self._to_correct_input_shape(inputs)
        outputs = self.encode(inputs)
        self.forward_pass += 1

        if self.return_all_ts:
            outputs = tuple(
                [
                    torch.stack([reshape_func(tensor) for tensor in out], -1)
                    for out in outputs
                ]
            )

            if len(self.return_vars) == 1:
                return outputs[0]
            return outputs

        else:
            if len(self.return_vars) == 1:
                return reshape_func(outputs[0][-1])
            return tuple([reshape_func(out[-1]) for out in outputs])

    def _init_weight_tensor(self) -> None:
        weights = torch.randn(
            self.n_neurons, self.in_c, self.kt, self.kh, self.kw, dtype=self.dtype
        )
        weights[weights.abs() < 1.0] = 0.0
        self.weights = torch.nn.Parameter(weights, requires_grad=self.req_grad)
        self.normalize_weights()

    def lateral_competition(self, acts: Tensor, conns: Tensor) -> Tensor:
        return F.conv3d(acts, conns, stride=1, padding=self.surround)

    def normalize_weights(self, eps: float = 1e-8) -> None:
        """Normalizes features such at each one has unit norm"""
        with torch.no_grad():
            dims = tuple(range(1, len(self.weights.shape)))
            scale = self.weights.norm(p=2, dim=dims, keepdim=True)
            self.weights.copy_(self.weights / (scale + eps))

    def _to_correct_input_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        pass

    def transfer(self, x: Tensor) -> Tensor:
        if type(self.transfer_func) == str:
            if self.transfer_func == "soft_threshold":
                return soft_threshold(x, self.lambda_, self.nonneg)
            elif self.transfer_func == "hard_threshold":
                return hard_threshold(x, self.lambda_, self.nonneg)
            else:
                raise ValueError
        elif callable(self.transfer_func):
            return self.transfer_func(x)

    def update_weights(self, acts: Tensor, recon_error: Tensor) -> None:
        """Updates the dictionary given the computed gradient"""
        with torch.no_grad():
            acts, _ = self._to_correct_input_shape(acts)
            recon_error, _ = self._to_correct_input_shape(recon_error)
            update = self.compute_weight_update(acts, recon_error)
            times_active = compute_times_active_by_feature(acts) + 1
            update *= self.eta / times_active
            update = torch.clamp(
                update, min=-self.d_update_clip, max=self.d_update_clip
            )
            self.weights.copy_(self.weights + update)
            self.normalize_weights()
            if self.lr_schedule is not None:
                self.eta = self.lr_schedule(self.forward_pass)

    def _update_tracks(
        self,
        tracks: dict[str, np.ndarray],
        lca_iter: int,
        acts: Tensor,
        inputs: Tensor,
        recons: Tensor,
        tau: Union[int, float],
    ) -> dict[str, np.ndarray]:
        """Update dictionary that stores the tracked metrics"""
        l2_rec_err = compute_l2_error(inputs, recons).item()
        l1_sparsity = compute_l1_sparsity(acts, self.lambda_).item()
        tracks["L2"][lca_iter - 1] = l2_rec_err
        tracks["L1"][lca_iter - 1] = l1_sparsity
        tracks["TotalEnergy"][lca_iter - 1] = l2_rec_err + l1_sparsity
        tracks["FractionActive"][lca_iter - 1] = compute_frac_active(acts)
        tracks["Tau"][lca_iter - 1] = tau
        return tracks

    def _write_params(self, arg_dict: dict[str, Any]) -> None:
        """Writes model params to file"""
        arg_dict["dtype"] = str(arg_dict["dtype"])
        del arg_dict["lr_schedule"]
        if callable(self.transfer_func):
            arg_dict["transfer_func"] = self.transfer_func.__name__
        for key, val in arg_dict.items():
            if type(val) == tuple:
                arg_dict[key] = list(val)
        with open(os.path.join(self.result_dir, "params.yaml"), "w") as yamlf:
            yaml.dump(arg_dict, yamlf, sort_keys=True)

    def _write_tracks(
        self, tracker: dict[str, np.ndarray], ts_cutoff: int, dev: Union[int, None]
    ) -> None:
        """Write out objective values to file"""
        for k, v in tracker.items():
            tracker[k] = v[:ts_cutoff]

        obj_df = pd.DataFrame(tracker)
        obj_df["LCAIter"] = np.arange(1, len(obj_df) + 1, dtype=np.int32)
        obj_df["ForwardPass"] = self.forward_pass.item()
        obj_df["Device"] = dev
        obj_df.to_csv(
            self.metric_fpath,
            header=True if not os.path.isfile(self.metric_fpath) else False,
            index=False,
            mode="a",
        )


class LCAConv1D(_LCAConvBase):
    def __init__(
        self,
        n_neurons: int,
        in_c: int,
        result_dir: str,
        kernel_size: Union[int, tuple[int]] = 7,
        stride: Union[int, tuple[int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 1000,
        eta: float = 0.01,
        lca_iters: int = 3000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        dtype: torch.dtype = torch.float32,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
    ) -> None:

        kernel_size = self._transform_conv_params(kernel_size)
        stride = self._transform_conv_params(stride)

        super(LCAConv1D, self).__init__(
            n_neurons,
            in_c,
            result_dir,
            kernel_size,
            stride,
            lambda_,
            tau,
            eta,
            lca_iters,
            pad,
            return_vars,
            return_all_ts,
            dtype,
            nonneg,
            track_metrics,
            transfer_func,
            input_zero_mean,
            input_unit_var,
            cudnn_benchmark,
            d_update_clip,
            lr_schedule,
            req_grad,
            False,
        )

    def _to_correct_input_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if len(inputs.shape) == 3:
            return to_5d_from_3d(inputs), to_3d_from_5d
        elif len(inputs.shape) == 5:
            return inputs, lambda inputs: inputs
        else:
            raise ValueError(
                f"Expected 3D inputs, but got {len(inputs.shape)}D inputs."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int]]
    ) -> tuple[int, int, int]:
        if type(val) == int:
            return (val, 1, 1)
        return val + (1, 1)

    def get_weights(self) -> None:
        return to_3d_from_5d(self.weights.detach())


class LCAConv2D(_LCAConvBase):
    def __init__(
        self,
        n_neurons: int,
        in_c: int,
        result_dir: str,
        kernel_size: Union[int, tuple[int, int]] = 7,
        stride: Union[int, tuple[int, int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 1000,
        eta: float = 0.01,
        lca_iters: int = 3000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        dtype: torch.dtype = torch.float32,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
    ) -> None:

        kernel_size = self._transform_conv_params(kernel_size)
        stride = self._transform_conv_params(stride)

        super(LCAConv2D, self).__init__(
            n_neurons,
            in_c,
            result_dir,
            kernel_size,
            stride,
            lambda_,
            tau,
            eta,
            lca_iters,
            pad,
            return_vars,
            return_all_ts,
            dtype,
            nonneg,
            track_metrics,
            transfer_func,
            input_zero_mean,
            input_unit_var,
            cudnn_benchmark,
            d_update_clip,
            lr_schedule,
            req_grad,
            True,
        )

    def _to_correct_input_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if len(inputs.shape) == 4:
            return to_5d_from_4d(inputs), to_4d_from_5d
        elif len(inputs.shape) == 5:
            return inputs, lambda inputs: inputs
        else:
            raise ValueError(
                f"Expected 4D inputs, but got {len(inputs.shape)}D inputs."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int, int]]
    ) -> tuple[int, int, int]:
        if type(val) == int:
            return (1, val, val)
        return (1,) + val

    def get_weights(self) -> Tensor:
        return to_4d_from_5d(self.weights.detach())


class LCAConv3D(_LCAConvBase):
    def __init__(
        self,
        n_neurons: int,
        in_c: int,
        result_dir: str,
        kernel_size: Union[int, tuple[int, int, int]] = 7,
        stride: Union[int, tuple[int, int, int]] = 1,
        lambda_: float = 0.25,
        tau: Union[float, int] = 1000,
        eta: float = 0.01,
        lca_iters: int = 3000,
        pad: Literal["same", "valid"] = "same",
        return_vars: Iterable[
            Literal[
                "inputs",
                "input_drives",
                "states",
                "acts",
                "recons",
                "recon_errors",
                "conns",
            ]
        ] = ["acts"],
        return_all_ts: bool = False,
        dtype: torch.dtype = torch.float32,
        nonneg: bool = True,
        track_metrics: bool = False,
        transfer_func: Union[
            Literal["soft_threshold", "hard_threshold"], Callable[[Tensor], Tensor]
        ] = "soft_threshold",
        input_zero_mean: bool = True,
        input_unit_var: bool = True,
        cudnn_benchmark: bool = True,
        d_update_clip: float = np.inf,
        lr_schedule: Optional[Callable[[int], float]] = None,
        req_grad: bool = False,
        no_time_pad: bool = False,
    ) -> None:

        kernel_size = self._transform_conv_params(kernel_size)
        stride = self._transform_conv_params(stride)

        super(LCAConv3D, self).__init__(
            n_neurons,
            in_c,
            result_dir,
            kernel_size,
            stride,
            lambda_,
            tau,
            eta,
            lca_iters,
            pad,
            return_vars,
            return_all_ts,
            dtype,
            nonneg,
            track_metrics,
            transfer_func,
            input_zero_mean,
            input_unit_var,
            cudnn_benchmark,
            d_update_clip,
            lr_schedule,
            req_grad,
            no_time_pad,
        )

    def _to_correct_input_shape(
        self, inputs: Tensor
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        if len(inputs.shape) == 5:
            return inputs, lambda inputs: inputs
        else:
            raise ValueError(
                f"Expected 5D inputs, but got {len(inputs.shape)}D inputs."
            )

    def _transform_conv_params(
        self, val: Union[int, tuple[int, int, int]]
    ) -> tuple[int, int, int]:
        if type(val) == int:
            return (val,) * 3
        return val

    def get_weights(self) -> Tensor:
        return self.weights.detach()
