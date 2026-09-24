import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def embedding_dropout(embed: nn.Embedding, words: torch.Tensor, dropout: float = 0.1, scale=None,
                      training: bool = True) -> torch.Tensor:
    """
    Title    : Regularizing and Optimizing LSTM Language Models - 2017
    Authors  : Stephen Merity, Nitish Shirish Keskar, Richard Socher
    Papers   : https://arxiv.org/pdf/1708.02182.pdf
    Source   : https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py
    Note     : Drops entire words of the vocabulary (rows of the embedding matrix), rescaled by 1 / (1 - dropout).
    :param embed: embedding layer.
    :param words: token ids.
    :param dropout: probability of dropping a vocabulary word.
    :param scale: optional per dimension scale of the embeddings.
    :param training: apply the dropout, pass module.training.
    :returns: the embeddings of words.
    """
    weight = embed.weight
    if dropout and training:
        mask = weight.new_empty((weight.size(0), 1)).bernoulli_(1 - dropout) / (1 - dropout)
        weight = mask * weight
    if scale is not None:
        weight = scale.expand_as(weight) * weight
    # padding_idx None stays None, -1 would make the last vocabulary row the padding row
    return F.embedding(
        words, weight, embed.padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse
    )


class LockedDropout(nn.Module):
    """
    Title    : A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    Authors  : Yarin Gal, Zoubin Ghahramani
    Papers   : https://arxiv.org/pdf/1512.05287.pdf
    Source   : https://github.com/flairNLP/flair/blob/master/flair/nn.py
    Note     : Implementation of locked (or variational) dropout.
               The same embedding dimensions are dropped at every time step of a sequence.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        shape = (x.size(0), 1, x.size(2)) if self.batch_first else (1, x.size(1), x.size(2))
        mask = x.new_empty(shape).bernoulli_(1 - self.dropout_rate) / (1 - self.dropout_rate)
        return mask * x


class WeightDrop(nn.Module):
    """
    Title    : Regularizing and Optimizing LSTM Language Models - 2017
    Authors  : Stephen Merity, Nitish Shirish Keskar, Richard Socher
    Papers   : https://arxiv.org/pdf/1708.02182.pdf
    Source   : https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
    Note     : DropConnect on the given weights of the wrapped module (e.g. the hidden to hidden weights of an
               LSTM, "weight_hh_l0"), a new mask at every forward pass in training mode.
    """

    def __init__(self, module: nn.Module, weights_to_drop: list[str], dropout: float = 0.0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights_to_drop = list(weights_to_drop)
        self.dropout = dropout
        self._setup()

    @staticmethod
    def _null(*args, **kwargs):
        # Replaces flatten_parameters, the dropped weights are not cuDNN compacted parameters anymore
        return

    def _setup(self):
        if isinstance(self.module, nn.RNNBase):
            self.module.flatten_parameters = self._null
        for name_w in self.weights_to_drop:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + "_raw", Parameter(w.detach().clone()))
            setattr(self.module, name_w, w.detach().clone())

    def _setweights(self):
        for name_w in self.weights_to_drop:
            raw_w = getattr(self.module, name_w + "_raw")
            setattr(self.module, name_w, F.dropout(raw_w, p=self.dropout, training=self.training))

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)


class WordDropout(nn.Module):
    """
    Title    : A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
    Authors  : Yarin Gal, Zoubin Ghahramani
    Papers   : https://arxiv.org/pdf/1512.05287.pdf
    Source   : https://github.com/flairNLP/flair/blob/master/flair/nn.py
    Note     : Implementation of word dropout.
               Randomly drops out entire words (or characters) in embedding space, not rescaled as in flair.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x
        mask = x.new_empty((x.size(0), x.size(1), 1)).bernoulli_(1 - self.dropout_rate)
        return mask * x
