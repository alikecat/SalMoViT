from functools import wraps
from inspect import signature, _empty
from typing import Callable

import numpy as np
import torch
from torch.nn import Conv1d, Module


class GaussFilter(Module):
    def __init__(
        self,
        sigma: float = None,
        kernel_size: int = None,
        padding="same",
        truncate=4.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        assert not (
            sigma is None and kernel_size is None
        ), "Either sigma or kernel_size must be provided"
        kernel_size = int(
            2 * np.ceil(truncate * sigma) + 1 if kernel_size is None else kernel_size
        )
        sigma = (kernel_size - 1) / 8 if sigma is None else sigma
        kernel = torch.linspace(
            -(kernel_size // 2), kernel_size // 2, kernel_size, device=device
        )
        kernel = torch.exp(-(kernel**2) / sigma**2 / 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        self.conv1d_x = Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.conv1d_y = Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.conv1d_x.weight.data = kernel
        self.conv1d_y.weight.data = kernel
        self.conv1d_x.weight.requires_grad = False
        self.conv1d_y.weight.requires_grad = False

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1d_y(x.reshape(-1, 1, w))
        x = x.reshape(b, c, h, w).transpose(2, 3)
        x = self.conv1d_x(x.reshape(-1, 1, h))
        x = x.reshape(b, c, w, h).transpose(2, 3)
        return x


def sparse_dense_elem_mul(sparse, dense):
    return torch.sparse_coo_tensor(
        sparse.indices(),
        sparse.values() * dense[tuple(sparse.indices())],
        size=sparse.shape,
    ).coalesce()


def precall_param_merge(callback: Callable[..., None]):
    def decorator(func):
        callback_signature = signature(callback)
        func_signature = signature(func)
        callback_params = callback_signature.parameters
        func_params = func_signature.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            callback_kwargs = {k: v for k, v in kwargs.items() if k in callback_params}
            func_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
            callback(**callback_kwargs)
            return func(*args, **func_kwargs)

        merged_params = [
            *func_params.values(),
            *(p for p in callback_params.values() if p.name not in func_params),
        ]
        sorted_params = [p for p in merged_params if p.default == _empty] + [
            p for p in merged_params if p.default != _empty
        ]
        wrapper.__signature__ = func_signature.replace(parameters=sorted_params)
        wrapper.__annotations__.update(callback.__annotations__)
        return wrapper

    return decorator
