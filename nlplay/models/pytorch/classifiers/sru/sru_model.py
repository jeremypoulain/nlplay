"""
Title    : Simple Recurrent Units for Highly Parallelizable Recurrence - 2017
Authors  : Tao Lei, Yu Zhang, Sida I. Wang, Hui Dai, Yoav Artzi
Papers   : https://arxiv.org/pdf/1709.02755.pdf
Source   : https://github.com/asappresearch/sru
"""
import torch
import torch.nn as nn
from torch.nn import init
from nlplay.models.pytorch.classifiers.sru.sru_functional import SRU


class SRUClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
            vocabulary_size: int,
            embedding_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            padding_idx: int = 0,
            dropout: float = 0.2,
            pretrained_vec=None,
            update_embedding: bool = True,):

        super(SRUClassifier, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.pretrained_vec = pretrained_vec
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

        self.encoder = SRU(
                embedding_size,
                hidden_size,
                num_layers,
                dropout=dropout,
            )
        self.fc1 = nn.Linear(embedding_size, num_classes)

    def forward(self, input):
        emb = self.embedding(input)
        emb = self.drop(emb)

        # Apply SRU
        output, hidden = self.encoder(emb)
        # Take the output of last RNN layer
        output = output[-1]

        output = self.drop(output)
        output = self.fc1(output)
        return output