"""
Title    : Attentive Convolution: Equipping CNNs with RNN-style Attention Mechanisms - 2017
Authors  : Wenpeng Yin, Hinrich Schütze
Papers   : https://arxiv.org/pdf/1710.00519.pdf
Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/classification/attentive_convolution.py
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from nlplay.models.pytorch.layers.layers import (
    Highway,
    AdditiveAttention2D,
    DotProductAttention,
)
from nlplay.models.pytorch.utils import init_tensor


class AttentiveConvNet(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        margin_size: int = 3,
        attentive_conv_net_type: str = "advanced",
        attention_type: str = "bilinear",
        dropout: float = 0.3,
        pretrained_vec=None,
        update_embedding: bool = True,
        pad_index=0,
        apply_sm: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.attentive_conv_net_type = attentive_conv_net_type.upper()
        self.attention_type = attention_type.upper()
        self.margin_size = margin_size
        assert self.margin_size % 2 == 1, "AttentiveConvNet margin size should be odd!"
        self.radius = int(self.margin_size / 2)

        self.embedding_dim = embedding_dim
        self.attention_dim = self.embedding_dim
        self.apply_sm = apply_sm
        self.device = device

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=self.embedding_dim,
            padding_idx=pad_index,
        )
        self.pretrained_vec = pretrained_vec
        if self.pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        else:
            init.xavier_uniform_(self.embedding.weight)
        if update_embedding:
            self.embedding.weight.requires_grad = update_embedding

        if self.attentive_conv_net_type == "ADVANCED":
            self.attention_dim *= 2
            self.x_context_highway = self.get_highway(
                self.embedding_dim, self.margin_size
            )

            self.x_self_highway = self.get_highway(self.embedding_dim, 1)

            self.a_context_highway = self.get_highway(
                self.embedding_dim, self.margin_size
            )
            self.a_self_highway = self.get_highway(self.embedding_dim, 1)
            self.beneficiary_highway = self.get_highway(self.embedding_dim, 1)

        if self.attention_type == "DOT":
            self.dot_product_attention = DotProductAttention(1.0)
        elif self.attention_type == "BILINEAR":
            self.bilinear_matrix = init_tensor(
                torch.empty(self.attention_dim, self.attention_dim)
            ).to(device)
            self.dot_product_attention = DotProductAttention(1.0)
        elif self.attention_type == "ADDITIVE_PROJECTION":
            self.additive_projection = AdditiveAttention2D(self.attention_dim)
        else:
            raise TypeError("Unsupported AttentionType: %s." % self.attention_type)

        self.attentive_conv = init_tensor(
            torch.empty(self.attention_dim, self.embedding_dim)
        ).to(device)
        self.x_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.embedding_dim,
                self.embedding_dim,
                self.margin_size,
                padding=self.radius,
            ),
            torch.nn.Tanh(),
        )
        self.bias = torch.zeros([self.embedding_dim]).to(device)
        self.hidden1_matrix = init_tensor(
            torch.empty(self.embedding_dim, self.hidden_size)
        ).to(device)
        self.hidden2_matrix = init_tensor(
            torch.empty(self.hidden_size, self.hidden_size)
        ).to(device)

        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(
            self.embedding_dim + 2 * self.hidden_size, num_classes
        )

    @staticmethod
    def get_highway(dimension, margin_size):
        radius = int(margin_size / 2)
        transformer_gate = torch.nn.Sequential(
            torch.nn.Conv1d(dimension, dimension, margin_size, padding=radius),
            torch.nn.Sigmoid(),
        )
        transformer_forward = torch.nn.Sequential(
            torch.nn.Conv1d(dimension, dimension, margin_size, padding=radius),
            torch.nn.Tanh(),
        )
        return Highway(transformer_gate, transformer_forward)

    def forward(self, x):

        embedding = self.embedding(x)

        if self.attentive_conv_net_type == "LIGHT":
            x_multi_granularity, a_multi_granularity, x_beneficiary = (
                embedding,
                embedding,
                embedding,
            )

        elif self.attentive_conv_net_type == "ADVANCED":
            embedding = embedding.permute(0, 2, 1)
            source_context = self.x_context_highway(embedding)
            source_self = self.x_self_highway(embedding)
            x_multi_granularity = torch.cat([source_context, source_self], 1).permute(
                0, 2, 1
            )

            focus_context = self.a_context_highway(embedding)
            focus_self = self.a_self_highway(embedding)
            a_multi_granularity = torch.cat([focus_context, focus_self], 1).permute(
                0, 2, 1
            )

            x_beneficiary = self.beneficiary_highway(embedding).permute(0, 2, 1)
        else:
            raise TypeError(
                "Unsupported AttentiveConvNetType: %s." % self.attentive_conv_net_type
            )

        if self.attention_type == "DOT":
            attentive_context = self.dot_product_attention(
                x_multi_granularity, a_multi_granularity, a_multi_granularity
            )
        elif self.attention_type == "BILINEAR":
            x_trans = x_multi_granularity.matmul(self.bilinear_matrix)
            attentive_context = self.dot_product_attention(
                x_trans, a_multi_granularity, a_multi_granularity
            )
        elif self.attention_type == "ADDITIVE_PROJECTION":
            attentive_context = self.additive_projection(
                a_multi_granularity, x_multi_granularity
            )

        attentive_conv = attentive_context.matmul(self.attentive_conv)
        x_conv = self.x_conv(x_beneficiary.permute(0, 2, 1)).permute(0, 2, 1)
        attentive_convolution = torch.tanh(attentive_conv + x_conv + self.bias).permute(
            0, 2, 1
        )
        hidden = torch.nn.functional.max_pool1d(
            attentive_convolution, kernel_size=attentive_convolution.size()[-1]
        ).squeeze()
        hidden1 = hidden.matmul(self.hidden1_matrix)
        hidden2 = hidden1.matmul(self.hidden2_matrix)
        hidden_layer = torch.cat([hidden, hidden1, hidden2], 1)

        out = self.dropout(self.fc1(hidden_layer))

        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
