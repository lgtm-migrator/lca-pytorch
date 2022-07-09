import torch


Tensor = torch.Tensor


def make_zero_mean(batch: Tensor) -> Tensor:
    dims = tuple(range((len(batch.shape) - 1) * -1, 0))
    mean = batch.mean(dim=dims, keepdim=True)
    return batch - mean


def make_unit_var(batch: Tensor, eps: float = 1e-8) -> Tensor:
    dims = tuple(range((len(batch.shape) - 1) * -1, 0))
    std = batch.std(dim=dims, keepdim=True)
    return batch / (std + eps)
