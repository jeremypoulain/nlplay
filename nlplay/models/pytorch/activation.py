import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Mish activation function.
    https://github.com/digantamisra98/Mish

    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class FTSwishPlus(nn.Module):
    # original FTSwish = https://arxiv.org/abs/1812.06247
    def __init__(self, threshold=-.25, mean_shift=-.1):
        super().__init__()
        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x):
        x = F.relu(x) * torch.sigmoid(x) + self.threshold
        # note on above - why not F.sigmoid?:
        # PyTorch docs - "nn.functional.sigmoid is deprecated. Use torch.sigmoid instead."
        # apply mean shift to drive mean to 0. -.1 was tested as optimal for kaiming init
        if self.mean_shift is not None:
            x.sub_(self.mean_shift)
        return x