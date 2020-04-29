import math
import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Title    : Mish: A Self Regularized Non-Monotonic Neural Activation Function - 2019
    Authors  : Diganta Misra
    Papers   : https://arxiv.org/abs/1908.08681
    Source   : https://github.com/digantamisra98/Mish
    Notes    : Applies the mish function element-wise:
               mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))"""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class FTSwishPlus(nn.Module):
    """
    FTSwish with mean shifting added to increase performance - Less Wright

    Title    : Flatten-T Swish: a thresholded ReLU-Swish-like activation function for deep learning
    Authors  : Swalpa Kumar Roy, Suvojit Manna, Shiv Ram Dubey, Bidyut B. Chaudhuri
    Papers   : https://arxiv.org/abs/1812.06247
               https://medium.com/@lessw/comparison-of-activation-functions-for-deep-learning-initial-winner-ftswish-f13e2621847
    Source   : https://github.com/lessw2020/FTSwishPlus
    """
    def __init__(self, threshold=-0.25, mean_shift=-0.1):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x):
        x = F.relu(x) * torch.sigmoid(x) + self.threshold
        if self.mean_shift is not None:
            x.sub_(self.mean_shift)
        return x


class LightRelu(nn.Module):
    """
    Title    : LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks -2019
    Authors  : Swalpa Kumar Roy, Suvojit Manna, Shiv Ram Dubey, Bidyut B. Chaudhuri
    Papers   : https://arxiv.org/abs/1901.05894
    Source   : https://github.com/lessw2020/LightRelu

    Notes from Less Wright :
    .46 was found to shift the mean to 0 on a random distribution test
    maxv of 7.5 was from initial testing on MNIST.
    Important - cut your learning rates in half with this...
    """

    def __init__(self, sub=.46, maxv=7.5):
        super().__init__()
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        # change to lisht

        x = x * torch.tanh(x)

        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


class TRelu(nn.Module):
    """
    Title    : An improved activation function for deep learning - Threshold Relu, or TRelu
    Authors  : Less Wright
    Papers   : ---
    Source   : https://github.com/lessw2020/TRelu
    """
    def __init__(self, threshold=- .25, mean_shift=-.03):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x):
        x = F.relu(x) + self.threshold

        if self.mean_shift is not None:
            x.sub_(self.mean_shift)

        return x