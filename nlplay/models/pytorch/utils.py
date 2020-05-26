import numpy as np
from nlplay.models.pytorch.activation import *
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
    elif activation_func_name == "gelu":
        return nn.GELU()
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
    elif activation_func_name == "mish":
        return Mish()
    elif activation_func_name == "ftswishplus":
        return FTSwishPlus()
    elif activation_func_name == "lightrelu":
        return LightRelu()
    elif activation_func_name == "trelu":
        return TRelu()
    else:
        raise ValueError("[!] Invalid activation function.")


def embeddings_to_cosine_similarity_matrix(embedding: torch.Tensor):
    """
    Title    : Converts a a tensor of n embeddings to an (n, n) tensor of similarities.
    Authors  : Dillon Erb - https://github.com/dte
    Papers   : ---
    Source   : https://gist.github.com/dte/e600bb76e72854379f4a306c1873f2c2#file-vectorized_cosine_similarities-py
    """
    dot = embedding @ embedding.t()
    norm = torch.norm(embedding, 2, 1)
    x = torch.div(dot, norm)
    x = torch.div(x, torch.unsqueeze(norm, 0))
    return x


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

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


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

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    """
    Title    : A masked softmax module to correctly implement attention in Pytorch.
    Authors  : Bilal Khan / AllenNLP
    Papers   : ---
    Source   : https://github.com/bkkaggle/pytorch_zoo/blob/master/pytorch_zoo/utils.py
               https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    A masked softmax module to correctly implement attention in Pytorch.
    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.

    Args:
        vector (torch.tensor): The tensor to softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the softmax operation.
        dim (int, optional): The dimension to softmax over.
                            Defaults to -1.
        memory_efficient (bool, optional): Whether to use a less precise, but more memory efficient implementation of masked softmax.
                                            Defaults to False.
        mask_fill_value ([type], optional): The value to fill masked values with if `memory_efficient` is `True`.
                                            Defaults to -1e32.

    Returns:
        torch.tensor: The masked softmaxed output
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector, mask, dim=-1):
    """
    Title    : A masked log-softmax module to correctly implement attention in Pytorch.
    Authors  : Bilal Khan / AllenNLP
    Papers   : ---
    Source   : https://github.com/bkkaggle/pytorch_zoo/blob/master/pytorch_zoo/utils.py
               https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    A masked log-softmax module to correctly implement attention in Pytorch.
    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    Args:
        vector (torch.tensor): The tensor to log-softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the log-softmax operation.
        dim (int, optional): The dimension to log-softmax over.
                            Defaults to -1.

    Returns:
        torch.tensor: The masked log-softmaxed output
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def get_gpu_info(device):

    device_name = torch.cuda.get_device_name(device)
    # major, minor = torch.cuda.get_device_capability(device)
    # device_capability = "CUDA Compute Capability: {}.{}".format(major, minor)
    mem_tot = human_readable_size(torch.cuda.get_device_properties(device).total_memory)
    mem_alloc = human_readable_size(torch.cuda.memory_allocated(device))
    out = "{} - Memory: {} / {}".format(device_name, mem_alloc, mem_tot)

    return out


def char_vectorizer(X, vocab, max_seq: int = 1014):
    """
    Function to transform input sentences into a one encoded matrix
    of a form [Sentence Index x Sentence Length x Vocabulary size],
    so that it can be directly fed into a Conv1D layer

    :param X: list of input sentences to be processed
    :param vocab: dict of characters to be taken into account for the vectorization
    :param max_seq: limit the max of a sentence
    :return: (nd.array): vectorized sentences
    """

    # TODO - Optimize this code as part of the Dataset/Vectorizer future refactoring
    vocab_size = len(vocab)
    output = np.zeros((len(X), max_seq, vocab_size))
    for i, sentence in enumerate(X):
        counter = 0
        sentence_vec = np.zeros((max_seq, vocab_size))
        chars = list(sentence.lower().replace(" ", ""))
        for c in chars:
            if counter >= max_seq:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in vocab.keys():
                    ix = vocab[c]
                    char_array[ix] = 1
                sentence_vec[counter, :] = char_array
                counter += 1
        output[i, :, :] = sentence_vec

    return output
