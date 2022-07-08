import torch


Tensor = torch.Tensor


def zero_mean(batch: Tensor) -> Tensor:
    dims = tuple(range((len(batch.shape) - 1) * -1, 0))
    mean = batch.mean(dim=dims, keepdim=True)
    return batch - mean


def contrast_norm(batch: Tensor, eps: float = 1e-8) -> Tensor:
    dims = tuple(range((len(batch.shape) - 1) * -1, 0))
    std = batch.std(dim=dims, keepdim=True)
    return batch / (std + eps)
