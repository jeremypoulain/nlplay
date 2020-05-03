"""
Title    : Convolutional Neural Networks for Sentence Classification - 2014
Authors  : Yoon Kim
Papers   : https://arxiv.org/abs/1607.01759
Source   : https://github.com/galsang/
           https://github.com/galsang/CNN-sentence-classification-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class TextCNN(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_classes,
        model_type="static",
        max_sent_len=400,
        embedding_dim=300,
        filters=[2, 3, 4],
        kernel_sizes=[50, 50, 50],
        dropout_prob=0.5,
        pretrained_vec=None,
        pad_index=0,
    ):

        super(TextCNN, self).__init__()

        self.model_type = model_type
        self.max_sent_len = max_sent_len
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout_prob
        self.in_channel = 1

        assert len(self.filters) == len(self.kernel_sizes)

        self.embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_dim, padding_idx=pad_index
        )

        if self.model_type == "rand":
            init.xavier_uniform_(self.embedding.weight)
        if (
            self.model_type == "static"
            or self.model_type == "non-static"
            or self.model_type == "multichannel"
        ):
            self.pretrained_vec = pretrained_vec
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
            if self.model_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.model_type == "multichannel":
                self.embedding2 = nn.Embedding(
                    self.vocabulary_size, self.embedding_dim, padding_idx=pad_index
                )
                self.embedding2.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2

        for i in range(len(self.filters)):
            conv = nn.Conv1d(
                self.in_channel,
                self.kernel_sizes[i],
                self.embedding_dim * self.filters[i],
                stride=self.embedding_dim,
            )
            setattr(self, f"conv_{i}", conv)

        self.fc = nn.Linear(sum(self.kernel_sizes), self.num_classes)

    def get_conv(self, i):
        return getattr(self, f"conv_{i}")

    def forward(self, x):
        x = self.embedding(x).view(-1, 1, self.embedding_dim * self.max_sent_len)
        if self.model_type == "multichannel":
            x2 = self.embedding2(x).view(-1, 1, self.embedding_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(
                F.relu(self.get_conv(i)(x)), self.max_sent_len - self.filters[i] + 1
            ).view(-1, self.kernel_sizes[i])
            for i in range(len(self.filters))
        ]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
