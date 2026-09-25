import math
import torch
import torch.nn as nn
from nlplay.models.pytorch.utils import init_tensor

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # PyTorch < 2.1
    from torch.nn.utils import weight_norm


class Highway(torch.nn.Module):
    """
    Title    : Highway Networks - 2015
               Training Very Deep Networks - 2015
    Authors  : Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    Papers   : https://arxiv.org/pdf/1505.00387.pdf
               https://arxiv.org/pdf/1507.06228.pdf
    Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/layers.py#L150
    Notes    : For now we don't limit the type of the gate and forward.
               Caller should init Highway with transformer and carry and guarantee the embedding_dim
               to be matching.
    """

    def __init__(self, transformer_gate, transformer_forward):
        super(Highway, self).__init__()
        self.transformer_forward = transformer_forward
        self.transformer_gate = transformer_gate

    def forward(self, x, gate_input=None, forward_input=None):
        if gate_input is None:
            gate_input = x
        if forward_input is None:
            forward_input = x
        gate = self.transformer_gate(gate_input)
        forward = self.transformer_forward(forward_input)
        return gate * forward + (1 - gate) * x


class Chomp1d(nn.Module):
    """
    Title    : An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling - 2018
    Authors  : Shaojie Bai, J. Zico Kolter, Vladlen Koltun
    Papers   : https://arxiv.org/pdf/1803.01271
    Source   : https://github.com/locuslab/TCN
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Title    : An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling - 2018
    Authors  : Shaojie Bai, J. Zico Kolter, Vladlen Koltun
    Papers   : https://arxiv.org/pdf/1803.01271
    Source   : https://github.com/locuslab/TCN
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        # Init before weight_norm: it recomputes weight from its magnitude and
        # direction at every forward, so writing weight afterwards is lost
        conv1.weight.data.normal_(0, 0.01)
        conv2.weight.data.normal_(0, 0.01)

        self.conv1 = weight_norm(conv1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(conv2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # conv1 and conv2 are initialized in __init__, before weight_norm
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Title    : An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling - 2018
    Authors  : Shaojie Bai, J. Zico Kolter, Vladlen Koltun
    Papers   : https://arxiv.org/pdf/1803.01271
    Source   : https://github.com/locuslab/TCN
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SumAttention(torch.nn.Module):
    """
    Reference: Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, input_dimension, attention_dimension, device, dropout=0):
        super(SumAttention, self).__init__()
        self.attention_matrix = torch.nn.Linear(
            input_dimension, attention_dimension
        ).to(device)
        self.attention_vector = torch.nn.Linear(attention_dimension, 1, bias=False).to(
            device
        )
        init_tensor(self.attention_matrix.weight)
        init_tensor(self.attention_vector.weight)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        if inputs.size(1) == 1:
            return self.dropout(inputs.squeeze(1))
        u = torch.tanh(self.attention_matrix(inputs))
        v = self.attention_vector(u)
        alpha = torch.nn.functional.softmax(v, 1).transpose(1, 2)
        return self.dropout(torch.matmul(alpha, inputs).squeeze(1))


class AdditiveAttention(torch.nn.Module):
    """
    Title    : Neural machine translation by jointly learning to align and translate
    Authors  : Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
    Papers   : https://arxiv.org/abs/1409.0473
    Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/layers.py#L45
    Notes    : Also known as Soft Attention or Bahdanau Attention
    """

    def __init__(self, dim, dropout=0):
        super(AdditiveAttention, self).__init__()
        self.w_attention_matrix = nn.Parameter(init_tensor(torch.empty(dim, dim)))
        self.u_attention_matrix = nn.Parameter(init_tensor(torch.empty(dim, dim)))
        self.v_attention_vector = nn.Parameter(init_tensor(torch.empty(dim, 1)))

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, s, h):
        raise NotImplementedError


class AdditiveAttention1D(AdditiveAttention):
    """
    Title    : Neural machine translation by jointly learning to align and translate
    Authors  : Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
    Papers   : https://arxiv.org/abs/1409.0473
    Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/layers.py#L63
    Notes    : Input shape is: [batch, embedding_dim] and [batch, seq_len, embedding_dim]
               Output is same with the first input
    """

    def forward(self, s, h):
        s_attention = s.matmul(self.w_attention_matrix).unsqueeze(1)
        h_attention = h.matmul(self.u_attention_matrix)
        attention = torch.tanh(s_attention + h_attention)
        attention = attention.matmul(self.v_attention_vector).squeeze(-1)
        attention_weight = torch.nn.functional.softmax(attention, -1)
        return self.dropout(attention_weight.unsqueeze(1).matmul(h).squeeze(1))


class AdditiveAttention2D(AdditiveAttention):
    """
    Title    : Neural machine translation by jointly learning to align and translate
    Authors  : Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
    Papers   : https://arxiv.org/abs/1409.0473
    Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/layers.py#L78
    Notes    : Input shape is: [batch, seq_len, embedding_dim] and [batch, seq_len, embedding_dim]
               Output is same with the first input
    """

    def forward(self, s, h):
        # [batch, s_len, 1, dim] + [batch, 1, h_len, dim] -> [batch, s_len, h_len, dim]
        s_attention = s.matmul(self.w_attention_matrix).unsqueeze(2)
        h_attention = h.matmul(self.u_attention_matrix).unsqueeze(1)
        attention = torch.tanh(s_attention + h_attention)
        attention = attention.matmul(self.v_attention_vector).squeeze(-1)
        attention_weight = torch.nn.functional.softmax(attention, -1)
        return self.dropout(attention_weight.matmul(h))


class DotProductAttention(torch.nn.Module):
    """
    Title    : Attention Is All You Need - 2017
    Authors  : Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
               Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    Papers   : https://arxiv.org/abs/1706.03762
    Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/layers.py#L95
    Notes    : Input shape is: [batch, seq_len, dim_k] and [batch, seq_len, dim_k]
               [batch, seq_len, dim_v]
               Output is same with the third input
    """

    def __init__(self, scaling_factor=None, dropout=0):
        super(DotProductAttention, self).__init__()
        self.scaling_factor = scaling_factor
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        # Scores are divided by scaling_factor, sqrt(dim_k) by default
        scaling_factor = self.scaling_factor
        if scaling_factor is None:
            scaling_factor = math.sqrt(q.size(2))
        e = q.matmul(k.permute(0, 2, 1)) / scaling_factor
        attention_weight = torch.nn.functional.softmax(e, -1)
        return self.dropout(attention_weight.matmul(v))


class MultiHeadAttention(torch.nn.Module):
    """
    Title    : Attention Is All You Need - 2017
    Authors  : Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
               Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    Papers   : https://arxiv.org/abs/1706.03762
    Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/06a8ed74215cf0e8bf0ee0e2d875f91ed93da06b/model/layers.py#L116
    """

    def __init__(self, dimension, dk, dv, head_number, scaling_factor, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.dk = dk
        self.dv = dv
        self.head_number = head_number
        self.q_linear = torch.nn.Linear(dimension, head_number * dk)
        self.k_linear = torch.nn.Linear(dimension, head_number * dk)
        self.v_linear = torch.nn.Linear(dimension, head_number * dv)
        self.scaling_factor = scaling_factor
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        def _reshape_permute(x, d, head_number):
            x = x.view(x.size(0), x.size(1), head_number, d)
            return x.permute(0, 2, 1, 3)

        q_trans = _reshape_permute(self.q_linear(q), self.dk, self.head_number)
        k_trans = _reshape_permute(self.k_linear(k), self.dk, self.head_number)
        v_trans = _reshape_permute(self.v_linear(v), self.dv, self.head_number)

        e = q_trans.matmul(k_trans.permute(0, 1, 3, 2)) / self.scaling_factor
        attention_weight = torch.nn.functional.softmax(e, -1)
        output = attention_weight.matmul(v_trans).permute(0, 2, 1, 3)
        output = output.reshape(
            output.size(0), output.size(1), output.size(2) * output.size(3)
        )
        return self.dropout(output)
