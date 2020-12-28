"""
Title    : Quasi-Recurrent Neural Networks - 2016
Authors  : James Bradbury, Stephen Merity, Caiming Xiong, Richard Socher
Papers   : https://arxiv.org/pdf/1611.01576
Source   : https://github.com/dreamgonfly/deep-text-classification-pytorch
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class QRNNLayer(nn.Module):
    def __init__(
        self, input_size, hidden_size, kernel_size=2, pooling="fo", zoneout=0.5
    ):
        super(QRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.zoneout = zoneout

        self.conv_z = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.conv_f = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.conv_o = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.conv_i = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=zoneout)

    def forward(self, x):

        zero_padding = Variable(
            torch.zeros(x.size(0), self.input_size, self.kernel_size - 1),
            requires_grad=False,
        )
        if x.is_cuda:
            zero_padding = zero_padding.cuda()
        x_padded = torch.cat([zero_padding, x], dim=2)

        z = self.tanh(self.conv_z(x_padded))
        if self.zoneout > 0:
            f = 1 - self.dropout(1 - self.sigmoid(self.conv_f(x_padded)))
        else:
            f = self.sigmoid(self.conv_f(x_padded))
        o = self.sigmoid(self.conv_o(x_padded))
        i = self.sigmoid(self.conv_i(x_padded))

        h_list, c_list = [], []
        h_prev = Variable(torch.zeros(x.size(0), self.hidden_size), requires_grad=False)
        c_prev = Variable(torch.zeros(x.size(0), self.hidden_size), requires_grad=False)
        if x.is_cuda:
            h_prev = h_prev.cuda()
            c_prev = c_prev.cuda()

        for t in range(x.size(2)):
            z_t = z[:, :, t]
            f_t = f[:, :, t]
            o_t = o[:, :, t]
            i_t = i[:, :, t]
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
        apply_sm: bool = True
    ):

        super(QRNN, self).__init__()

        self.apply_sm = apply_sm
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=padding_idx
        )
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        self.dropout = nn.Dropout(p=drop_out)
        self.dense = dense

        qrnn_layers = []
        input_size = embedding_size
        for _ in range(num_layers - 1):
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
            else:
                x = x

        last_timestep = x[:, :, -1]
        out = self.linear(last_timestep)

        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
