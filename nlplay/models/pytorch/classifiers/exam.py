"""
Title   : Explicit Interaction Model towards Text Classification
Author  : Cunxiao Du, Zhaozheng Chin, Fuli Feng, Lei Zhu, Tian Gan, Liqiang Nie
Papers  : https://arxiv.org/pdf/1811.09386.pdf
"""
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from nlplay.models.pytorch.utils import get_activation_func


class EXAM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 128,
        region_size: int = 7,
        max_sent_len: int = 256,
        activation_function: str = "relu",
        padding_idx: int = 0,
        drop_out: float = 0.2,
        pretrained_vec=None,
        update_embedding: bool = True,
        device: str = "cuda",
        apply_sm: bool = True
    ):
        """
        Args:
            num_classes (int) : number of classes
            vocabulary_size (int): number of items in the vocabulary
            embedding_size (int): size of the embeddings
            padding_idx (int): default 0; Embedding will not use this index
            drop_out (float) : default 0.2; drop out rate applied to the embedding layer
            pretrained_vec (nd.array): default None : numpy matrix containing pretrained word vectors
            update_embedding: bool (boolean) : default True : option to train/freeze embedding layer weights
        """
        super(EXAM, self).__init__()
        self.num_classes = num_classes
        self.max_sent_len = max_sent_len
        self.region_size = region_size
        self.region_radius = self.region_size // 2
        self.embedding_size = embedding_size
        self.drop_out = drop_out
        self.pretrained_vec = pretrained_vec
        self.device = torch.device(device)
        self.apply_sm = apply_sm

        # Embedding layers required for the region embedding (Word Context Scenario)
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=region_size * self.embedding_size,
            padding_idx=padding_idx,
        )
        self.embedding_region = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=self.embedding_size,
            padding_idx=padding_idx,
        )

        self.activation = get_activation_func(activation_function.lower())
        self.max_pool_1d = nn.AdaptiveAvgPool1d(output_size=1)

        # EXAM adds 2 extra linear layers (dense1/dense2) on top of the default region embedding models
        self.dense0 = nn.Linear(self.embedding_size, num_classes)
        self.dense1 = nn.Linear(
            self.max_sent_len - 2 * self.region_radius, self.max_sent_len * 2
        )
        self.dense2 = nn.Linear(self.max_sent_len * 2, 1)

        if self.pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        else:
            init.xavier_uniform_(self.embedding.weight)
        if update_embedding:
            self.embedding.weight.requires_grad = update_embedding

    def forward(self, x):
        # Retrieve batch size as extra parameter for data preparation
        batch_size = x.shape[0]

        # Batch data preparation for region embedding - Qiao et al. (2018) - https://openreview.net/pdf?id=BkSDMA36Z
        aligned_seq = torch.zeros(
            (self.max_sent_len - 2 * self.region_radius, batch_size, self.region_size),
            dtype=torch.int64,
            device=self.device,
        )
        for i in range(self.region_radius, self.max_sent_len - self.region_radius):
            aligned_seq[i - self.region_radius] = x[
                :, i - self.region_radius : i - self.region_radius + self.region_size
            ]
        trimed_seq = x[:, self.region_radius : self.max_sent_len - self.region_radius]
        mask = torch.repeat_interleave(
            (trimed_seq > 0).type(torch.uint8).reshape((batch_size, -1, 1)),
            repeats=self.embedding_size,
            dim=2,
        )

        # Region embedding setup
        region_aligned_seq = aligned_seq.transpose(1, 0)
        region_aligned_emb = self.embedding_region(region_aligned_seq).reshape(
            (batch_size, -1, self.region_size, self.embedding_size)
        )
        context_unit = self.embedding(trimed_seq).reshape(
            (batch_size, -1, self.region_size, self.embedding_size)
        )
        projected_emb = region_aligned_emb * context_unit

        feature = self.max_pool_1d(
            projected_emb.transpose(3, 2).reshape((batch_size, -1, self.region_size))
        ).reshape((batch_size, -1, self.embedding_size))
        feature = feature * mask

        # Exam - Feature interaction with classes
        feature = feature.reshape((-1, self.embedding_size))
        feature = (
            self.dense0(feature)
            .reshape((batch_size, -1, self.num_classes))
            .transpose(2, 1)
            .reshape((batch_size * self.num_classes, -1))
        )

        # Exam - Aggregation Layer modifications
        feature = feature.unsqueeze(1)
        residual = torch.sum(feature, axis=2).reshape((batch_size, self.num_classes))
        res = (
            self.dense2(self.activation(self.dense1(feature)))
            .reshape(batch_size * self.num_classes, 1, -1)
            .reshape((batch_size, self.num_classes))
        )

        out = res + residual
        if self.apply_sm:
            out = F.log_softmax(out, dim=1)
        return out
