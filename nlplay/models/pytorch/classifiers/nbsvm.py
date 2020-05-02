"""
Title    : Baselines and Bigrams: Simple, Good Sentiment and Topic Classification- 2012
Authors  : Sida Wang and Christopher D. Manning
Papers   : https://www.aclweb.org/anthology/P12-2018.pdf
Source   : https://www.youtube.com/watch?v=XJ_waZlJU8g&feature=youtu.be&t=43m31s
           https://github.com/fastai/fastai/blob/release-1.0.61/old/fastai/nlp.py
Note     : This is the customized version of NBSVM, called NBSVM++ , made by Jeremy Howard & Stephen Merity
"""
import numpy as np
import torch
from torch import nn


class NBSVM(nn.Module):
    def __init__(self, vocab_size, n_classes, r, w_adj=0.4, r_adj=10, padding_idx=0):
        super(NBSVM, self).__init__()

        # Init w
        self.w = nn.Embedding(vocab_size + 1, 1, padding_idx=padding_idx)
        self.w.weight.data.uniform_(-0.1, 0.1)
        self.w.weight.data[padding_idx] = 0

        # Init r
        self.r = nn.Embedding(vocab_size + 1, n_classes)
        self.r.weight.data = torch.Tensor(np.concatenate([np.zeros((1, n_classes)), r]))
        self.r.weight.requires_grad = False

        self.w_adj = w_adj
        self.r_adj = r_adj

    def forward(self, feat_idx):
        w = self.w(feat_idx) + self.w_adj
        r = self.r(feat_idx)
        x = (w * r).sum(dim=1)
        x = x / self.r_adj
        return x
