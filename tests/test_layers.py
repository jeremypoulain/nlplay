import math

import pytest
import torch
from torch import nn

from nlplay.models.pytorch.classifiers.att_conv_net import AttentiveConvNet
from nlplay.models.pytorch.layers.layers import (
    AdditiveAttention1D,
    AdditiveAttention2D,
    Chomp1d,
    DotProductAttention,
    MultiHeadAttention,
    SumAttention,
    TemporalBlock,
    TemporalConvNet,
)


def test_additive_attention_weights_are_parameters():
    att = AdditiveAttention2D(8)
    names = {name for name, _ in att.named_parameters()}
    assert names == {"w_attention_matrix", "u_attention_matrix", "v_attention_vector"}
    assert set(att.state_dict()) == names


@pytest.mark.parametrize("batch, s_len, h_len", [(4, 7, 7), (5, 5, 5), (1, 6, 6), (3, 2, 9)])
def test_additive_attention_2d_matches_loop(batch, s_len, h_len):
    torch.manual_seed(0)
    dim = 6
    att = AdditiveAttention2D(dim).eval()
    s, h = torch.randn(batch, s_len, dim), torch.randn(batch, h_len, dim)
    out = att(s, h)
    assert out.shape == (batch, s_len, dim)
    for b in range(batch):
        for i in range(s_len):
            scores = torch.tanh(s[b, i] @ att.w_attention_matrix + h[b] @ att.u_attention_matrix)
            weights = torch.softmax((scores @ att.v_attention_vector).squeeze(-1), -1)
            assert torch.allclose(out[b, i], weights @ h[b], atol=1e-5)


@pytest.mark.parametrize("batch", [1, 3])
def test_additive_attention_1d_and_sum_attention_batch_of_one(batch):
    torch.manual_seed(0)
    s, h = torch.randn(batch, 6), torch.randn(batch, 5, 6)
    assert AdditiveAttention1D(6).eval()(s, h).shape == (batch, 6)
    sum_att = SumAttention(6, 4, "cpu").eval()
    assert sum_att(h).shape == (batch, 6)
    assert sum_att(h[:, :1]).shape == (batch, 6)


def test_sum_attention_matches_weighted_sum():
    torch.manual_seed(0)
    att = SumAttention(6, 4, "cpu").eval()
    x = torch.randn(3, 5, 6)
    scores = att.attention_vector(torch.tanh(att.attention_matrix(x)))
    expected = (torch.softmax(scores, 1) * x).sum(1)
    assert torch.allclose(att(x), expected, atol=1e-6)


def test_dot_product_attention_default_scaling():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 16), torch.randn(2, 4, 16), torch.randn(2, 4, 5)
    att = DotProductAttention()
    expected = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(16), -1) @ v
    assert torch.allclose(att(q, k, v), expected, atol=1e-6)
    # The default is computed per call, not frozen at the first input size
    q4, k4 = torch.randn(2, 3, 4), torch.randn(2, 4, 4)
    expected4 = torch.softmax(q4 @ k4.transpose(1, 2) / 2.0, -1) @ v
    assert torch.allclose(att(q4, k4, v), expected4, atol=1e-6)
    assert att.scaling_factor is None


def test_multi_head_attention_output_shape():
    att = MultiHeadAttention(dimension=12, dk=4, dv=5, head_number=3, scaling_factor=2.0)
    q = torch.randn(2, 7, 12)
    assert att(q, q, q).shape == (2, 7, 15)


def test_chomp_zero_keeps_input():
    x = torch.randn(2, 3, 5)
    assert torch.equal(Chomp1d(0)(x), x)
    assert Chomp1d(2)(x).shape == (2, 3, 3)


def test_temporal_block_init_survives_weight_norm():
    torch.manual_seed(0)
    block = TemporalBlock(8, 16, kernel_size=3, stride=1, dilation=1, padding=2)
    block(torch.randn(2, 8, 10))
    # Default Conv1d init has std ~0.1 for fan_in 24, the TCN init is 0.01
    assert block.conv1.weight.std() < 0.02
    assert block.conv2.weight.std() < 0.02


def test_temporal_conv_net_kernel_size_one_keeps_length():
    net = TemporalConvNet(4, [6, 6], kernel_size=1)
    assert net(torch.randn(2, 4, 9)).shape == (2, 6, 9)


@pytest.mark.parametrize("net_type", ["light", "advanced"])
@pytest.mark.parametrize("attention_type", ["dot", "bilinear", "additive_projection"])
def test_attentive_conv_net_attention_types(net_type, attention_type):
    torch.manual_seed(0)
    model = AttentiveConvNet(
        vocabulary_size=50,
        num_classes=3,
        embedding_dim=8,
        attentive_conv_net_type=net_type,
        attention_type=attention_type,
    )
    x = torch.randint(1, 50, (4, 11))
    out = model(x)
    assert out.shape == (4, 3)
    out.sum().backward()
    if attention_type == "additive_projection":
        assert model.additive_projection.w_attention_matrix.grad is not None
