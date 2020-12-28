import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class CustomRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 300,
        spatial_dropout: float = 0.1,
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = False,
        dropout:float = 0.15,
        pretrained_vec=None,
        update_embedding: bool = False,
        padding_idx: int = 0,
        apply_sm: bool = True,
    ):

        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.rnn_encoder = None
        self.spatial_dropout = spatial_dropout
        self.dropout = dropout
        self.apply_sm = apply_sm

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

        if self.rnn_type == "lstm":
            self.rnn_encoder = nn.LSTM(
                input_size=embedding_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
            )

        elif self.rnn_type == "gru":
            self.rnn_encoder = nn.GRU(
                input_size=embedding_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
            )
        else:
            raise NotImplementedError

        if rnn_bidirectional:
            h_size = rnn_hidden_size * 4
        else:
            h_size = rnn_hidden_size * 2

        self.fc1 = nn.Linear(h_size, num_classes)

    def forward(self, x):

        x = self.embedding(x)

        # Apply SpatialDropout
        x = x.permute(0, 2, 1)  # reshape to [batch, channels, time]
        x = F.dropout2d(
            x, self.spatial_dropout, training=self.training
        )
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]

        x, _ = self.rnn_encoder(x)

        # apply average and max pooling on rnn output
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        # concatenate average and max pooling
        feats = torch.cat((avg_pool, max_pool), 1)
        feats = F.dropout(feats, self.dropout)

        # pass through the output layer and return the output
        out = self.fc1(feats)

        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
