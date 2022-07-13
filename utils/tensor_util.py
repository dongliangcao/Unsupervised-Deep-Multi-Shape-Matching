import numpy as np
import numbers
import torch


def _to_device(x, device):
    assert torch.is_tensor(x), f'x is not a torch.Tensor'
    x = x.to(device=device)
    return x


def to_device(x, device):
    if isinstance(x, list):
        x = [to_device(x_i, device) for x_i in x]
        return x
    elif isinstance(x, dict):
        x = {k: to_device(v, device) for (k, v) in x.items()}
        return x
    else:
        return _to_device(x, device)


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array([tensor])
    else:
        raise NotImplementedError()


def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError()


def to_number(ndarray):
    if isinstance(ndarray, torch.Tensor) or isinstance(ndarray, np.ndarray):
        return ndarray.item()
    elif isinstance(ndarray, numbers.Number):
        return ndarray
    else:
        raise NotImplementedError()
