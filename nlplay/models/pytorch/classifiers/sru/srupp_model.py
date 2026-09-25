"""
Title    : When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute - 2021
Authors  : Tao Lei
Papers   : https://arxiv.org/abs/2102.12459
Source   : https://github.com/asappresearch/sru (branch 3.0.0-dev)
"""
import torch
from torch import nn
from torch.nn import functional as F

from nlplay.models.pytorch.classifiers.sru.sru_functional import SRUCell
from nlplay.models.pytorch.utils import masked_max, masked_mean, padding_mask, reset_padding_embedding


class SRUppProjectedLinear(nn.Module):
    """
    Low rank input projection of the SRU++ layers without attention: U = W_o LN(W_q x).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        proj_features: int,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(proj_features) if layer_norm else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.dropout.p > 0:
            self.linear2.weight.data.mul_((1 - self.dropout.p) ** 0.5)

    def forward(self, x: torch.Tensor, mask_pad: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        :param x: input of shape (seq_len, batch, in_features).
        :param mask_pad: unused, kept for the SRUCell custom module interface.
        :returns: U of shape (seq_len, batch, out_features).
        """
        z = self.linear1(x)
        if self.layer_norm is not None:
            z = self.layer_norm(z)
        return self.linear2(self.dropout(z))


class SRUppAttention(nn.Module):
    """
    Self attention replacing the SRU input projection: Q = W_q x, K = W_k Q, V = W_v Q,
    U = W_o LN(Q + alpha Attention(Q, K, V)).
    alpha starts at 0 so every layer starts as an SRU with a low rank input projection.
    The attention is not causal since the whole text is known for classification.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        proj_features: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        rezero_init_alpha: float = 0.0,
        layer_norm: bool = True,
    ):
        super().__init__()
        if proj_features % num_heads != 0:
            raise ValueError(f"proj_features ({proj_features}) must be divisible by num_heads ({num_heads})")
        self.proj_features = proj_features
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.rezero_init_alpha = rezero_init_alpha
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, proj_features, bias=False)
        self.linear2 = nn.Linear(proj_features, proj_features * 2, bias=False)
        self.linear3 = nn.Linear(proj_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.tensor([rezero_init_alpha]))
        self.layer_norm = nn.LayerNorm(proj_features) if layer_norm else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.alpha.data.fill_(self.rezero_init_alpha)
        if self.dropout.p > 0:
            self.linear3.weight.data.mul_((1 - self.dropout.p) ** 0.5)

    def forward(self, x: torch.Tensor, mask_pad: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        :param x: input of shape (seq_len, batch, in_features).
        :param mask_pad: boolean mask of shape (seq_len, batch), True for padding positions.
        :returns: U of shape (seq_len, batch, out_features).
        """
        seq_len, batch_size, _ = x.shape
        head_dim = self.proj_features // self.num_heads

        q = self.linear1(x)
        k, v = self.linear2(q).chunk(2, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(seq_len, batch_size, self.num_heads, head_dim).permute(1, 2, 0, 3)

        attn_mask = None
        if mask_pad is not None:
            # Finite additive mask so that a fully padded text gets uniform weights instead of NaN
            attn_mask = q.new_zeros(batch_size, 1, 1, seq_len).masked_fill(
                mask_pad.t().bool()[:, None, None, :], torch.finfo(q.dtype).min
            )
        attn = F.scaled_dot_product_attention(
            split_heads(q),
            split_heads(k),
            split_heads(v),
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.proj_features)

        out = q + self.alpha * attn
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return self.linear3(self.dropout(out))


class SRUpp(nn.Module):
    """
    Stack of SRU layers whose input projection is an attention module every attention_every_n_layers
    layers and a low rank projection otherwise.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        bidirectional: bool = False,
        highway_bias: float = -2.0,
        attention_every_n_layers: int = 1,
        weight_c_init: float = 1.0,
    ):
        super().__init__()
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_lst = nn.ModuleList()
        for i in range(num_layers):
            in_features = input_size if i == 0 else self.output_size
            # 3 matrices for the candidate and the 2 gates, a 4th one projects the skip term when sizes differ
            out_features = self.output_size * (3 if in_features == self.output_size else 4)
            if (i + 1) % attention_every_n_layers == 0:
                transform = SRUppAttention(
                    in_features,
                    out_features,
                    proj_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            else:
                transform = SRUppProjectedLinear(in_features, out_features, proj_size, dropout=dropout)
            self.rnn_lst.append(
                SRUCell(
                    in_features,
                    hidden_size,
                    dropout=dropout if i + 1 != num_layers else 0,
                    bidirectional=bidirectional,
                    highway_bias=highway_bias,
                    rescale=False,
                    custom_m=transform,
                    weight_c_init=weight_c_init,
                )
            )

    def forward(self, x: torch.Tensor, mask_pad: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: input of shape (seq_len, batch, input_size).
        :param mask_pad: boolean mask of shape (seq_len, batch), True for padding positions,
            padding keeps the recurrent state unchanged, outputs zeros and is ignored by the attention.
        :returns: hidden states of shape (seq_len, batch, output_size).
        """
        for rnn in self.rnn_lst:
            x, _ = rnn(x, mask_pad=mask_pad)
        return x


class SRUppClassifier(nn.Module):
    """
    Embedding → SRU++ encoder → masked pooling over the real tokens → linear classifier.
    """

    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int = 256,
        proj_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 1,
        attention_every_n_layers: int = 1,
        bidirectional: bool = True,
        pooling: str = "mean_max",
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        highway_bias: float = -2.0,
        padding_idx: int | None = 0,
        pretrained_vec=None,
        update_embedding: bool = True,
    ):
        """
        :param hidden_size: SRU state size per direction.
        :param proj_size: attention size, the paper uses a quarter of the model size.
        :param attention_every_n_layers: 1 puts attention in every layer, k in every k-th layer only.
        :param pooling: "mean", "max" or "mean_max" over the real tokens.
        :param highway_bias: initial bias of the highway gate, negative values favour the skip connection.
        """
        super().__init__()
        if pooling not in ("mean", "max", "mean_max"):
            raise ValueError(f"pooling must be 'mean', 'max' or 'mean_max', got {pooling!r}")
        self.padding_idx = padding_idx
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_idx)
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_vec))
        else:
            nn.init.xavier_uniform_(self.embedding.weight)
        reset_padding_embedding(self.embedding)
        self.embedding.weight.requires_grad = update_embedding

        self.encoder = SRUpp(
            embedding_size,
            hidden_size,
            proj_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            bidirectional=bidirectional,
            highway_bias=highway_bias,
            attention_every_n_layers=attention_every_n_layers,
        )
        pooled_size = self.encoder.output_size * (2 if pooling == "mean_max" else 1)
        self.fc1 = nn.Linear(pooled_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: token ids of shape (batch, seq_len), pre or post padded.
        :returns: logits of shape (batch, num_classes).
        """
        mask = padding_mask(x, self.padding_idx)
        emb = self.drop(self.embedding(x))

        # SRU works on (seq_len, batch, features) and expects True on padding positions
        h = self.encoder(emb.transpose(0, 1), mask_pad=(~mask).t().contiguous())
        h = h.transpose(0, 1)

        if self.pooling == "mean":
            pooled = masked_mean(h, mask)
        elif self.pooling == "max":
            pooled = masked_max(h, mask)
        else:
            pooled = torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=1)
        return self.fc1(self.drop(pooled))
