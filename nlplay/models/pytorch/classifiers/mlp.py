from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F, init
from nlplay.models.pytorch.utils import (
    get_activation_func,
    masked_max,
    masked_mean,
    padding_mask,
    reset_padding_embedding,
)


class MLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 300,
        embedding_mode: str = "avg",
        fc_hidden_sizes: Sequence[int] = (256, 128, 64),
        fc_activation_functions: Sequence[str] = ("relu", "relu", "relu"),
        fc_dropouts: Sequence[float | None] = (0.2, None, None),
        padding_idx: int = 0,
        pretrained_vec=None,
        update_embedding: bool = True,
        apply_sm: bool = True
    ):
        """
        Pooled word embeddings followed by an MLP, padding positions are ignored by the pooling.
        :param num_classes: number of classes.
        :param vocabulary_size: number of items in the vocabulary.
        :param embedding_size: size of the embeddings.
        :param embedding_mode: "avg", "max" or "concat" (avg and max pooling concatenated).
        :param fc_hidden_sizes: size of each hidden layer.
        :param fc_activation_functions: activation of each hidden layer.
        :param fc_dropouts: dropout rate after each hidden layer, None or 0 for no dropout.
        :param padding_idx: padding token id, its embedding is kept at zero.
        :param pretrained_vec: optional numpy matrix of shape (vocabulary_size, embedding_size).
        :param update_embedding: train (True) or freeze (False) the embedding layer.
        :param apply_sm: return log probabilities (for NLLLoss) instead of raw scores.
            Must be False for losses expecting raw scores, e.g. CrossEntropyLoss or ModifiedHuberLoss.
        """
        super(MLP, self).__init__()

        if embedding_mode not in ("avg", "max", "concat"):
            raise ValueError(f"Unknown embedding_mode: {embedding_mode}")
        if not len(fc_hidden_sizes) == len(fc_activation_functions) == len(fc_dropouts):
            raise ValueError("fc_hidden_sizes, fc_activation_functions and fc_dropouts must have the same length")

        self.embedding_mode = embedding_mode
        self.padding_idx = padding_idx
        self.apply_sm = apply_sm
        self.pretrained_vec = pretrained_vec
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )
        with torch.no_grad():
            if self.pretrained_vec is not None:
                pretrained = torch.as_tensor(self.pretrained_vec, dtype=self.embedding.weight.dtype)
                if tuple(pretrained.shape) != (vocabulary_size, embedding_size):
                    raise ValueError(
                        f"pretrained_vec must have shape {(vocabulary_size, embedding_size)}, "
                        f"got {tuple(pretrained.shape)}"
                    )
                self.embedding.weight.copy_(pretrained)
            else:
                init.xavier_uniform_(self.embedding.weight)
        reset_padding_embedding(self.embedding)
        self.embedding.weight.requires_grad_(update_embedding)

        in_size = embedding_size * 2 if self.embedding_mode == "concat" else embedding_size

        # Dynamic setup of MLP given the input parameters
        self.hidden_sizes = [in_size, *fc_hidden_sizes, num_classes]
        modules = []
        for size_in, size_out, activation, dropout in zip(
            self.hidden_sizes[:-2], self.hidden_sizes[1:-1], fc_activation_functions, fc_dropouts
        ):
            modules.append(nn.Linear(in_features=size_in, out_features=size_out))
            activation_func = get_activation_func(activation)
            if activation_func is not None:
                modules.append(activation_func)
            if dropout is not None and dropout > 0.0:
                modules.append(nn.Dropout(p=dropout))
        modules.append(nn.Linear(in_features=self.hidden_sizes[-2], out_features=num_classes))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        x_embedding = self.embedding(x)
        mask = padding_mask(x, self.padding_idx)

        # Pooling over the embedding, padding positions excluded
        if self.embedding_mode == "avg":
            x = masked_mean(x_embedding, mask)
        elif self.embedding_mode == "max":
            x = masked_max(x_embedding, mask)
        else:
            x = torch.cat((masked_mean(x_embedding, mask), masked_max(x_embedding, mask)), dim=1)

        # Apply each module of the MLP Layer setup
        for m in self.module_list:
            x = m(x)

        if self.apply_sm:
            return F.log_softmax(x, dim=1)
        return x
