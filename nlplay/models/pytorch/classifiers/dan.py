import torch
import torch.nn as nn
from torch.nn import functional as F, init


class DAN2L(nn.Module):
    """
    Deep Averaging Network - DAN with 2Layers
    Deep Averaging Network - Mohit Iyyer and Varun Manjunatha and Jordan Boyd-Graber and Hal - 2015
    paper : https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
    """

    def __init__(
        self,
        num_classes: int,
        vocabulary_size=10000,
        embedding_size=300,
        hidden_size=256,
        padding_idx: int = 0,
        drop_out: float = 0.2,
        pretrained_vec=None,
        freeze_embedding: bool = False,
    ):
        super(DAN2L, self).__init__()
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
        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        self.drop_out = drop_out
        self.dropout1 = nn.Dropout(self.drop_out)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.dropout2 = nn.Dropout(self.drop_out)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        if self.drop_out > 0.0:
            x = self.dropout1(x)

        x = F.relu(self.fc1(x))
        if self.drop_out > 0.0:
            x = self.dropout2(x)

        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)

        return out


class DAN3L(nn.Module):
    """
    Deep Averaging Network - DAN with 3Layers
    Deep Averaging Network - Mohit Iyyer and Varun Manjunatha and Jordan Boyd-Graber and Hal - 2015
    paper : https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
    """

    def __init__(
        self,
        num_classes: int,
        vocabulary_size=10000,
        embedding_size=300,
        hidden_size=256,
        padding_idx: int = 0,
        drop_out: float = 0.2,
        pretrained_vec=None,
        freeze_embedding: bool = False,
    ):
        super(DAN3L, self).__init__()
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
        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        self.drop_out = drop_out
        self.dropout1 = nn.Dropout(self.drop_out)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.dropout2 = nn.Dropout(self.drop_out)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size ))
        self.dropout3 = nn.Dropout(self.drop_out)
        self.fc3 = nn.Linear(int(hidden_size ), num_classes)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        if self.drop_out > 0.0:
            x = self.dropout1(x)

        x = F.relu(self.fc1(x))
        if self.drop_out > 0.0:
            x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        if self.drop_out > 0.0:
            x = self.dropout3(x)

        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)

        return out
