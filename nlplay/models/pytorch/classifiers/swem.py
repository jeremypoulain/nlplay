"""
Title    : Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms - 2018
Authors  : Dinghan Shen, Guoyin Wang, Wenlin Wang, Martin Renqiang Min, Qinliang Su, Yizhe Zhang, Chunyuan Li,
           Ricardo Henao, Lawrence Carin
Papers   : https://arxiv.org/pdf/1805.09843.pdf
"""
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from nlplay.models.pytorch.utils import get_activation_func


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
            swem_mode (str):
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
        if update_embedding:
            self.embedding.weight.requires_grad = update_embedding

        if self.swem_mode == "concat":
            in_size = embedding_size * 2
        else:
            in_size = embedding_size

        # TODO : the AdaptiveAvgPool1d only allows to use a swem_window=2 aka bigram
        self.hier_pool = nn.AdaptiveAvgPool1d(self.embedding_size)

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.activation = get_activation_func(activation_function.lower())
        self.fc2 = nn.Linear(hidden_size, out_features=num_classes)

    def forward(self, x):
        x_embedding = self.embedding(x)

        if self.swem_mode == "avg":
            # apply global average pooling only
            x_embedding = x_embedding.mean(dim=1)

        elif self.swem_mode == "max":
            # apply global max pooling only
            x_embedding, _ = torch.max(x_embedding, dim=1)

        elif self.swem_mode == "concat":
            # apply global average pooling
            x1 = x_embedding.mean(dim=1)
            # apply global max pooling
            x2, _ = torch.max(x_embedding, dim=1)
            # concat average & max pooling
            x_embedding = torch.cat((x1, x2), dim=1)

        elif self.swem_mode == "hier":
            # Rearrange the embedding shape to perform the AdaptiveAvgPool1d
            x_embedding = x_embedding.permute(0, 2, 1)
            x_embedding = self.hier_pool(x_embedding).permute(0, 2, 1)
            # Apply global max-pooling operation on top of the representations for every window
            x_embedding, _ = torch.max(x_embedding, dim=1)

        if self.drop_out > 0.0:
            x_embedding = F.dropout(x_embedding, self.drop_out)

        h_layer = self.fc1(x_embedding)
        h_layer = self.activation(h_layer)
        out = self.fc2(h_layer)
        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
