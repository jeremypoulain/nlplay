import numpy as np
import torch
from torch import nn
from nlplay.utils.utils import human_readable_size


def set_seed(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_activation_func(activation_func_name: str = "relu"):
    if activation_func_name is "none":
        return None
    elif activation_func_name == "relu":
        return nn.ReLU()
    elif activation_func_name == "relu6":
        return nn.ReLU6()
    elif activation_func_name == "prelu":
        return nn.PReLU()
    elif activation_func_name == "elu":
        return nn.ELU()
    elif activation_func_name == "selu":
        return nn.SELU()
    elif activation_func_name == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_func_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_func_name == "tanh":
        return nn.Tanh()
    elif activation_func_name == "hardtanh":
        return nn.Hardtanh()
    elif activation_func_name == "tanhshrink":
        return nn.Tanhshrink()
    elif activation_func_name == "hardshrink":
        return nn.Hardshrink()
    elif activation_func_name == "softshrink":
        return nn.Softshrink()
    elif activation_func_name == "softsign":
        return nn.Softsign()
    elif activation_func_name == "softplus":
        return nn.Softplus()
    else:
        raise ValueError("[!] Invalid activation function.")


def embeddings_to_cosine_similarity_matrix(E):
    """
    Converts a a tensor of n embeddings to an (n, n) tensor of similarities.
    E = torch.randn(20000, 100)
    embeddings_to_cosine_similarity_matrix(E)
  """
    dot = E @ E.t()
    norm = torch.norm(E, 2, 1)
    x = torch.div(dot, norm)
    x = torch.div(x, torch.unsqueeze(norm, 0))
    return x


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


def get_gpu_info(device):

    device_name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    device_capability = "CUDA Compute Capability: {}.{}".format(major, minor)
    mem_tot = human_readable_size(torch.cuda.get_device_properties(device).total_memory)
    mem_alloc = human_readable_size(torch.cuda.memory_allocated(device))
    out = "{} - Memory: {} / {}".format(device_name, mem_alloc, mem_tot)
    return out