import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_activation_func(activation_func_name: str = "relu"):
    if activation_func_name is "none":
        return None
    elif activation_func_name is "relu":
        return nn.ReLU()
    elif activation_func_name is "relu6":
        return nn.ReLU6()
    elif activation_func_name is "prelu":
        return nn.PReLU()
    elif activation_func_name is "elu":
        return nn.ELU()
    elif activation_func_name is "selu":
        return nn.SELU()
    elif activation_func_name is "leakyrelu":
        return nn.LeakyReLU()
    elif activation_func_name is "sigmoid":
        return nn.Sigmoid()
    elif activation_func_name is "tanh":
        return nn.Tanh()
    elif activation_func_name is "hardtanh":
        return nn.Hardtanh()
    elif activation_func_name is "tanhshrink":
        return nn.Tanhshrink()
    elif activation_func_name is "hardshrink":
        return nn.Hardshrink()
    elif activation_func_name is "softshrink":
        return nn.Softshrink()
    elif activation_func_name is "softsign":
        return nn.Softsign()
    elif activation_func_name is "softplus":
        return nn.Softplus()
    else:
        raise ValueError("[!] Invalid activation function.")
