import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


def embedding_dropout(embed, words, dropout=0.1, scale=None):
    """
    Title    : Regularizing and Optimizing LSTM Language Models - 2017
    Authors  : Stephen Merity 1 Nitish Shirish Keskar 1 Richard Socher 1
    Papers   : https://arxiv.org/pdf/1708.02182.pdf
    Source   :
    """
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - dropout
        ).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )
    return X


class LockedDropout(torch.nn.Module):
    """
    Title    : A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    Authors  : Yarin Gal, Zoubin Ghahramani
    Papers   : https://arxiv.org/pdf/1512.05287.pdf
    Source   : https://github.com/flairNLP/flair/blob/master/flair/nn.py
    Note     : Implementation of locked (or variational) dropout.
               Randomly drops out entire parameters in embedding space.
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


class WeightDrop(torch.nn.Module):
    """
    Title    : Regularizing and Optimizing LSTM Language Models - 2017
    Authors  : Stephen Merity 1 Nitish Shirish Keskar 1 Richard Socher 1
    Papers   : https://arxiv.org/pdf/1708.02182.pdf
    Source   : https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
    """
    def __init__(self, module, weights_to_drop, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights_to_drop = weights_to_drop
        self.dropout = dropout
        self._setup()

    def null(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null
            for name_w in self.weights:
                w = getattr(self.module, name_w)
                del self.module._parameters[name_w]
                self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights_to_drop:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class WordDropout(torch.nn.Module):
    """
    Title    : A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    Authors  : Yarin Gal, Zoubin Ghahramani
    Papers   : https://arxiv.org/pdf/1512.05287.pdf
    Source   : https://github.com/flairNLP/flair/blob/master/flair/nn.py
    Note     : Implementation of word dropout.
               Randomly drops out entire words (or characters) in embedding space.
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

