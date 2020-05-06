"""
Title   : Bag of Tricks for Efficient Text Classification
Author  : Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas
Papers  : https://arxiv.org/abs/1607.01759
"""
import torch
import torch.nn as nn
from torch.nn import functional as F, init


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
        Args:
            num_classes (int) : number of classes
            vocabulary_size (int): number of items in the vocabulary
            embedding_size (int): size of the embeddings
            padding_idx (int): default 0; Embedding will not use this index
            drop_out (float) : default 0.2; drop out rate applied to the embedding layer
            pretrained_vec (nd.array): default None : numpy matrix containing pretrained word vectors
            update_embedding: bool = True, (boolean) : default False : option to freeze/don't train embedding layer
        """
        super(PytorchFastText, self).__init__()
        self.drop_out = drop_out
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

        self.fc1 = nn.Linear(embedding_size, out_features=num_classes)

    def forward(self, x):
        # global average pooling
        x_embedding = self.embedding(x).mean(dim=1)
        if self.drop_out > 0.0:
            x_embedding = F.dropout(x_embedding, self.drop_out)

        out = self.fc1(x_embedding)
        out = F.log_softmax(out, dim=1)
        return out
