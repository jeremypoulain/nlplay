"""
Title   : Bag of Tricks for Efficient Text Classification
Author  : Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas
Papers  : https://arxiv.org/abs/1607.01759
"""
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from nlplay.models.pytorch.utils import reset_padding_embedding


class PytorchFastText(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int,
        padding_idx: int = 0,
        drop_out: float = 0.2,
        pretrained_vec=None,
        update_embedding: bool = True,
    ):
        """
        Mean of the token embeddings followed by a linear layer, returns raw scores.
        :param num_classes: number of classes.
        :param vocabulary_size: number of items in the vocabulary.
        :param embedding_size: size of the embeddings.
        :param padding_idx: padding token id, excluded from the mean and kept at zero.
        :param drop_out: dropout rate applied to the averaged embedding, in [0, 1).
        :param pretrained_vec: optional numpy matrix of shape (vocabulary_size, embedding_size).
        :param update_embedding: train (True) or freeze (False) the embedding layer.
        """
        super(PytorchFastText, self).__init__()
        if not 0.0 <= drop_out < 1.0:
            raise ValueError("drop_out must be in [0, 1)")
        self.drop_out = drop_out
        self.pretrained_vec = pretrained_vec
        # EmbeddingBag averages without building the (batch, seq_len, embedding_size) tensor,
        # padding_idx is excluded from the mean
        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            mode="mean",
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

        self.fc1 = nn.Linear(embedding_size, out_features=num_classes)

    def forward(self, x):
        # global average pooling over real tokens, fully padded rows give a zero vector
        x_embedding = self.embedding(x)
        x_embedding = F.dropout(x_embedding, self.drop_out, training=self.training)
        return self.fc1(x_embedding)
