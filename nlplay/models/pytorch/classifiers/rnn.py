import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class RNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        rnn_type: str = "lstm",
        embedding_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.2,
        pretrained_vec=None,
        update_embedding: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.rnn_encoder = None
        self.drop_out = dropout

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

        if bidirectional:
            h_size = hidden_size * 2
        else:
            h_size = hidden_size

        if self.rnn_type == "lstm":
            self.rnn_encoder = nn.LSTM(
                input_size=embedding_size,
                hidden_size=h_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )

        elif self.rnn_type == "gru":
            self.rnn_encoder = nn.GRU(
                input_size=embedding_size,
                hidden_size=h_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(h_size, num_classes)

    def forward(self, x):
        embeddings = self.embedding(x)

        # from the recurrent layer, only take the activities from the last sequence step
        if self.rnn_type == "gru":
            _, rec_out = self.rnn_encoder(embeddings)
        else:
            _, (rec_out, _) = self.rnn_encoder(embeddings)

        # Take the output of last RNN layer
        out = rec_out[-1]

        # Apply dropout regularization
        if self.drop_out > 0.0:
            out = F.dropout(out, p=self.drop_out)

        # Apply SoftMax
        out = self.fc1(out)
        #out = F.log_softmax(out, dim=-1)

        return out
