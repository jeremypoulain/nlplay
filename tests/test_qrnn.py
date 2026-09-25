import pytest
import torch
from torch import nn

from nlplay.models.pytorch.classifiers.qrnn import QRNN, QRNNLayer


def paper_pooling(layer, x):
    """
    Reference QRNN layer written from the paper equations, one convolution per gate.
    :param layer: fused layer whose weights are split per gate.
    :param x: input of shape (batch, channels, seq_len).
    :returns: hidden states of shape (batch, hidden, seq_len).
    """
    n = layer.POOLING_GATES[layer.pooling]
    weights, biases = layer.conv.weight.chunk(n), layer.conv.bias.chunk(n)
    x = nn.functional.pad(x, (layer.kernel_size - 1, 0))
    pre = [nn.functional.conv1d(x, w, b) for w, b in zip(weights, biases)]
    z = torch.tanh(pre[0])
    # Zoneout at inference, F = 1 - (1 - p)(1 - sigmoid)
    f = 1 - (1 - torch.sigmoid(pre[1])) * (1 - layer.zoneout)
    h = c = torch.zeros_like(z[:, :, 0])
    out = []
    for t in range(z.size(2)):
        if layer.pooling == "f":
            h = f[:, :, t] * h + (1 - f[:, :, t]) * z[:, :, t]
        else:
            if layer.pooling == "fo":
                c = f[:, :, t] * c + (1 - f[:, :, t]) * z[:, :, t]
            else:
                c = f[:, :, t] * c + torch.sigmoid(pre[3][:, :, t]) * z[:, :, t]
            h = torch.sigmoid(pre[2][:, :, t]) * c
        out.append(h)
    return torch.stack(out, dim=2)


@pytest.mark.parametrize("pooling", ["f", "fo", "ifo"])
def test_layer_matches_paper_equations(pooling):
    torch.manual_seed(0)
    layer = QRNNLayer(6, 5, kernel_size=3, pooling=pooling, zoneout=0.3).eval()
    x = torch.randn(2, 6, 9)
    h, c = layer(x)
    torch.testing.assert_close(h, paper_pooling(layer, x))
    assert (c is None) == (pooling == "f")


@pytest.mark.parametrize("pooling, gates", [("f", 2), ("fo", 3), ("ifo", 4)])
def test_every_parameter_is_used(pooling, gates):
    torch.manual_seed(0)
    model = QRNN(2, 30, 8, hidden_size=5, num_layers=2, pooling=pooling)
    model(torch.randint(1, 30, (3, 7))).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.qrnn_layers[0].conv.out_channels == 5 * gates


def test_invalid_pooling():
    with pytest.raises(ValueError):
        QRNN(2, 30, 8, pooling="xx")


@pytest.mark.parametrize("pooling", ["f", "fo", "ifo"])
def test_loads_checkpoints_with_one_convolution_per_gate(pooling):
    torch.manual_seed(0)
    layer = QRNNLayer(6, 5, kernel_size=3, pooling=pooling).eval()
    n = layer.POOLING_GATES[pooling]
    old = {}
    for g, w, b in zip("zfoi", layer.conv.weight.chunk(n), layer.conv.bias.chunk(n)):
        old[f"conv_{g}.weight"], old[f"conv_{g}.bias"] = w.clone(), b.clone()
    # Old layers always had the 4 convolutions
    for g in "zfoi"[n:]:
        old[f"conv_{g}.weight"], old[f"conv_{g}.bias"] = torch.randn(5, 6, 3), torch.randn(5)
    x = torch.randn(2, 6, 9)
    expected, _ = layer(x)

    fresh = QRNNLayer(6, 5, kernel_size=3, pooling=pooling).eval()
    fresh.load_state_dict(old)
    torch.testing.assert_close(fresh(x)[0], expected)
