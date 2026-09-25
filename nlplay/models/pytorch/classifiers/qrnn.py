"""
Title    : Quasi-Recurrent Neural Networks - 2016
Authors  : James Bradbury, Stephen Merity, Caiming Xiong, Richard Socher
Papers   : https://arxiv.org/pdf/1611.01576
Source   : https://github.com/dreamgonfly/deep-text-classification-pytorch
"""
import torch
from torch import nn
from torch.nn import functional as F
from nlplay.models.pytorch.utils import reset_padding_embedding


class QRNNLayer(nn.Module):
    # Gates computed by the convolution for each pooling mode, z is the candidate
    POOLING_GATES = {"f": 2, "fo": 3, "ifo": 4}

    def __init__(
        self, input_size, hidden_size, kernel_size=2, pooling="fo", zoneout=0.5
    ):
        super(QRNNLayer, self).__init__()
        if pooling not in self.POOLING_GATES:
            raise ValueError(f"pooling must be 'f', 'fo' or 'ifo', got {pooling!r}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.zoneout = zoneout

        # One convolution for all the gates the pooling uses, same init as separate convolutions
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size * self.POOLING_GATES[pooling],
            kernel_size=kernel_size,
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the fusion hold one convolution per gate in z, f, o, i order
        if prefix + "conv_z.weight" in state_dict:
            gates = ["z", "f", "o", "i"][: self.POOLING_GATES[self.pooling]]
            for name in ("weight", "bias"):
                state_dict[prefix + "conv." + name] = torch.cat(
                    [state_dict[f"{prefix}conv_{g}.{name}"] for g in gates]
                )
            for g in "zfoi":
                for name in ("weight", "bias"):
                    state_dict.pop(f"{prefix}conv_{g}.{name}", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):

        # Causal padding so that the convolution at step t only sees steps <= t
        x_padded = F.pad(x, (self.kernel_size - 1, 0))

        gates = self.conv(x_padded).chunk(self.POOLING_GATES[self.pooling], dim=1)
        z = self.tanh(gates[0])
        f = self.sigmoid(gates[1])
        if self.zoneout > 0:
            # Zoneout F = 1 - dropout(1 - F) with an unscaled mask so that F stays in [0, 1],
            # the expected update is used at inference
            update = 1 - f
            if self.training:
                update = update * torch.bernoulli(torch.full_like(update, 1 - self.zoneout))
            else:
                update = update * (1 - self.zoneout)
            f = 1 - update
        o = self.sigmoid(gates[2]) if self.pooling != "f" else None
        i = self.sigmoid(gates[3]) if self.pooling == "ifo" else None

        h_list, c_list = [], []
        h_prev = x.new_zeros(x.size(0), self.hidden_size)
        c_prev = x.new_zeros(x.size(0), self.hidden_size)

        for t in range(x.size(2)):
            z_t = z[:, :, t]
            f_t = f[:, :, t]
            o_t = o[:, :, t] if o is not None else None
            i_t = i[:, :, t] if i is not None else None
            h_prev, c_prev = self.pool(h_prev, c_prev, z_t, f_t, o_t, i_t)
            h_list.append(h_prev)
            if c_prev is not None:
                c_list.append(c_prev)

        h = torch.stack(h_list, dim=2)
        if c_prev is not None:
            c = torch.stack(c_list, dim=2)
            return h, c
        else:
            return h, None

    def pool(self, h_prev, c_prev, z_t, f_t, o_t, i_t):

        if self.pooling == "f":
            c_t = None
            h_t = f_t * h_prev + (1 - f_t) * z_t
        elif self.pooling == "fo":
            c_t = f_t * c_prev + (1 - f_t) * z_t
            h_t = o_t * c_t
        elif self.pooling == "ifo":
            c_t = f_t * c_prev + i_t * z_t
            h_t = o_t * c_t

        return h_t, c_t


class QRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int,
        padding_idx: int = 0,
        drop_out: float = 0.3,
        pretrained_vec=None,
        dense=True,
        zoneout=0.5,
        pooling="fo",
        kernel_size=3,
        num_layers=3,
        hidden_size=300,
    ):

        super(QRNN, self).__init__()

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=padding_idx
        )
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_vec))
        reset_padding_embedding(self.embedding)
        self.dropout = nn.Dropout(p=drop_out)
        self.dense = dense

        qrnn_layers = []
        input_size = embedding_size
        for _ in range(num_layers):
            qrnn_layers.append(
                QRNNLayer(input_size, hidden_size, kernel_size, pooling, zoneout)
            )
            if self.dense:
                input_size += hidden_size
            else:
                input_size = hidden_size

        self.qrnn_layers = nn.ModuleList(qrnn_layers)
        self.linear = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):

        x = self.embedding(x).transpose(1, 2)  # batch_size, channels, timesteps
        for qrnn_layer in self.qrnn_layers:
            residual = x
            h, c = qrnn_layer(x)
            x = self.dropout(h)
            if self.dense:
                x = torch.cat([x, residual], dim=1)

        last_timestep = x[:, :, -1]
        out = self.linear(last_timestep)


        return out
