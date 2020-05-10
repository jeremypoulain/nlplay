"""
Title    : Long Short Term Memory - 1997 (LSTM)
Authors  : Sepp Hochreiter, Jurgen Schmidhuber
Papers   : https://www.bioinf.jku.at/publications/older/2604.pdf

Title    : Neural architectures for named entity recognition - 2016 (BiLSTM)
Authors  : Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer
Papers   : https://arxiv.org/abs/1603.01360

Title    : Neural Machine Translation by Jointly Learning to Align and Translate - 2014 (GRU)
Authors  : Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
Papers   : https://arxiv.org/abs/1409.0473
"""
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
        update_embedding: bool = False,
        padding_idx: int = 0,
    ):
        """
        Args:
            num_classes (int) : number of classes
            vocabulary_size (int): number of items in the vocabulary
            embedding_size (int): size of the embeddings
            hidden_size: (int) :
            num_layers: (int) :
            bidirectional: (bool) :
            dropout (float) : default 0.2; drop out rate applied to the embedding layer
            pretrained_vec (nd.array): default None : numpy matrix containing pretrained word vectors
            update_embedding: bool = True, (boolean) : default False : option to freeze/don't train embedding layer
            padding_idx (int): default 0; Embedding will not use this index
        """
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
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(h_size, num_classes)

    def forward(self, x):
        embeddings = self.embedding(x)

        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero
        if self.rnn_type == "gru":
            output, h_n = self.rnn_encoder(embeddings)
        else:
            output, (h_n, c_n) = self.rnn_encoder(embeddings)

        # Take the output of last RNN layer
        out = h_n[-1]

        # Apply dropout regularization
        if self.drop_out > 0.0:
            out = F.dropout(out, p=self.drop_out)

        out = self.fc1(out)

        return out
