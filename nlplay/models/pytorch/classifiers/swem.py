"""
Title    : Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms - 2018
Authors  : Dinghan Shen, Guoyin Wang, Wenlin Wang, Martin Renqiang Min, Qinliang Su, Yizhe Zhang, Chunyuan Li,
           Ricardo Henao, Lawrence Carin
Papers   : https://arxiv.org/pdf/1805.09843.pdf
"""
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


class SWEM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 300,
        hidden_size: int = 100,
        swem_mode: str = "concat",
        swem_window: int = 2,
        activation_function: str = "relu",
        drop_out: float = 0.2,
        padding_idx: int = 0,
        pretrained_vec=None,
        update_embedding: bool = True,
        apply_sm: bool = True
    ):
        """
        Args:
            num_classes (int) : number of classes
            vocabulary_size (int): number of items in the vocabulary
            embedding_size (int): size of the embeddings
            swem_mode (str): "avg", "max", "concat" or "hier"
            swem_window (int): window size of the "hier" mode local average pooling
            activation_function (str)
            drop_out (float) : default 0.2; drop out rate applied to the embedding layer
            padding_idx (int): default 0; Embedding will not use this index
            pretrained_vec (nd.array): default None : numpy matrix containing pretrained word vectors
            update_embedding (boolean) : default True : train or freeze the embedding layer

        """
        super(SWEM, self).__init__()

        self.swem_mode = swem_mode
        self.swem_window = swem_window
        self.apply_sm = apply_sm
        self.drop_out = drop_out
        self.pretrained_vec = pretrained_vec
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )

        if self.pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        else:
            init.xavier_uniform_(self.embedding.weight)
        reset_padding_embedding(self.embedding)
        self.embedding.weight.requires_grad = update_embedding

        if self.swem_mode == "concat":
            in_size = embedding_size * 2
        else:
            in_size = embedding_size

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.activation = get_activation_func(activation_function.lower())
        self.fc2 = nn.Linear(hidden_size, out_features=num_classes)

    def forward(self, x):
        mask = padding_mask(x, self.embedding.padding_idx)
        x_embedding = self.embedding(x)

        # All poolings are computed over real tokens only
        if self.swem_mode == "avg":
            # apply global average pooling only
            x_embedding = masked_mean(x_embedding, mask)

        elif self.swem_mode == "max":
            # apply global max pooling only
            x_embedding = masked_max(x_embedding, mask)

        elif self.swem_mode == "concat":
            # concat global average & max pooling
            x_embedding = torch.cat((masked_mean(x_embedding, mask), masked_max(x_embedding, mask)), dim=1)

        elif self.swem_mode == "hier":
            # Average pooling over each local window of swem_window words, real tokens only
            weights = mask.unsqueeze(1).to(x_embedding.dtype)
            window_sum = F.avg_pool1d(x_embedding.permute(0, 2, 1) * weights, self.swem_window, stride=1)
            window_count = F.avg_pool1d(weights, self.swem_window, stride=1)
            window_count = window_count.squeeze(1)
            windows = (window_sum.permute(0, 2, 1) / window_count.clamp(min=1.0 / self.swem_window).unsqueeze(2))
            # Keep full windows only, or the partial ones for sentences shorter than swem_window
            full = window_count > 1.0 - 1e-6
            valid = torch.where(full.any(dim=1, keepdim=True), full, window_count > 0)
            # Apply global max-pooling on top of the valid windows
            x_embedding = masked_max(windows, valid)

        else:
            raise ValueError(f"Unknown swem_mode: {self.swem_mode}")

        if self.drop_out > 0.0:
            x_embedding = F.dropout(x_embedding, self.drop_out, training=self.training)

        h_layer = self.fc1(x_embedding)
        h_layer = self.activation(h_layer)
        out = self.fc2(h_layer)
        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
